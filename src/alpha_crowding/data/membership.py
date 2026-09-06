"""Point-in-time index-membership schedule construction."""

from __future__ import annotations

import pandas as pd


def weekly_last_sessions(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the final trading session in each Friday-ending calendar week."""

    required = {"calendar_date", "is_trading_day"}
    missing = required - set(calendar.columns)
    if missing:
        raise KeyError(f"missing calendar columns: {sorted(missing)}")
    dates = pd.to_datetime(calendar["calendar_date"], errors="raise").dt.normalize()
    trading = pd.to_numeric(calendar["is_trading_day"], errors="raise").eq(1)
    sessions = pd.Series(dates[trading].drop_duplicates().sort_values().to_numpy())
    if sessions.empty:
        raise ValueError("calendar contains no trading sessions")
    weekly = sessions.groupby(sessions.dt.to_period("W-FRI")).max()
    return pd.DatetimeIndex(weekly.to_numpy()).sort_values()


def validate_constituent_snapshot(
    frame: pd.DataFrame,
    *,
    index: str,
    requested_date: object,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Reject incomplete or future-dated CSI300/CSI500 responses."""

    expected = {"CSI300": 300, "CSI500": 500}
    key = index.upper()
    if key not in expected:
        raise ValueError("index must be CSI300 or CSI500")
    required = {"index", "requested_date", "updateDate", "code"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing constituent columns: {sorted(missing)}")
    required_rows = expected[key] if expected_rows is None else expected_rows
    if required_rows < 1:
        raise ValueError("expected_rows must be positive")
    if len(frame) != required_rows:
        raise ValueError(f"{key} expected {required_rows} rows; found {len(frame)}")
    if frame["code"].duplicated().any():
        raise ValueError(f"{key} contains duplicate codes")
    requested = pd.Timestamp(requested_date).normalize()
    updates = pd.to_datetime(frame["updateDate"], errors="raise").dt.normalize()
    if (updates > requested).any():
        raise ValueError(f"{key} contains future updateDate")
    if not frame["index"].eq(key).all():
        raise ValueError("snapshot index column mismatch")
    if not pd.to_datetime(frame["requested_date"]).dt.normalize().eq(requested).all():
        raise ValueError("snapshot requested_date column mismatch")
    return {
        "index": key,
        "requested_date": requested.date().isoformat(),
        "rows": len(frame),
        "source_update_date": updates.max().date().isoformat(),
    }


def repair_merger_membership_gaps(
    frame: pd.DataFrame, events: list[dict[str, object]]
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Repair provider timing gaps using dated official replacement announcements."""

    required = {"index", "requested_date", "updateDate", "code", "code_name"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing constituent repair columns: {sorted(missing)}")
    out = frame.copy()
    out["requested_date"] = pd.to_datetime(out["requested_date"]).dt.normalize()
    out["updateDate"] = pd.to_datetime(out["updateDate"]).dt.normalize()
    out["membership_source"] = "baostock"
    out["availableDate"] = out["updateDate"]
    out["effectiveDate"] = out["updateDate"]
    out["source_url"] = ""
    repairs: list[dict[str, str]] = []
    additions = []
    for event in events:
        index = str(event["index"])
        gap_start = pd.Timestamp(event["provider_gap_start"]).normalize()
        visible = pd.Timestamp(event["provider_successor_visible"]).normalize()
        effective = pd.Timestamp(event["official_successor_effective"]).normalize()
        available = pd.Timestamp(event["announcement_available"]).normalize()
        predecessor = str(event["predecessor_code"])
        successor = str(event["successor_code"])
        dates = sorted(
            out.loc[
                out["index"].eq(index)
                & out["requested_date"].ge(gap_start)
                & out["requested_date"].lt(visible),
                "requested_date",
            ].unique()
        )
        for date_value in dates:
            date = pd.Timestamp(date_value).normalize()
            group_mask = out["index"].eq(index) & out["requested_date"].eq(date)
            if int(group_mask.sum()) != 499:
                raise ValueError(
                    f"official repair expected a 499-row {index} snapshot at {date.date()}"
                )
            code = predecessor if date < effective else successor
            if (group_mask & out["code"].eq(code)).any():
                raise ValueError(f"repair code {code} already exists at {date.date()}")
            templates = out.loc[out["index"].eq(index) & out["code"].eq(code)]
            if templates.empty:
                raise ValueError(f"no template row available for official repair code {code}")
            distances = (templates["requested_date"] - date).abs()
            row = templates.loc[distances.idxmin()].copy()
            row["requested_date"] = date
            row["updateDate"] = available
            row["membership_source"] = "official_index_announcement"
            row["availableDate"] = available
            row["effectiveDate"] = effective
            row["source_url"] = str(event["source_url"])
            additions.append(row)
            repairs.append(
                {
                    "index": index,
                    "requested_date": date.date().isoformat(),
                    "code": code,
                    "reason": "provider removed predecessor before official replacement and exposed successor after its effective date",
                    "source_url": str(event["source_url"]),
                }
            )
    if additions:
        out = pd.concat([out, pd.DataFrame(additions)], ignore_index=True)
    return out.sort_values(["requested_date", "index", "code"]).reset_index(drop=True), repairs


def classify_missing_membership_market_rows(
    merged: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Separate documented retained merger predecessors from unexplained gaps."""

    required = {"date", "membership_source", "effectiveDate", "_merge"}
    missing = required - set(merged.columns)
    if missing:
        raise KeyError(f"missing market-coverage fields: {sorted(missing)}")
    dates = pd.to_datetime(merged["date"]).dt.normalize()
    effective = pd.to_datetime(merged["effectiveDate"]).dt.normalize()
    missing_market = merged["_merge"].ne("both")
    documented = (
        missing_market
        & merged["membership_source"].eq("official_index_announcement")
        & dates.lt(effective)
    )
    return documented, missing_market & ~documented
