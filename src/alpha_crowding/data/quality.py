"""Quality metrics for real BaoStock daily-history acceptance."""

from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "turn",
    "pctChg",
    "pbMRQ",
)


def summarize_daily_history(frame: pd.DataFrame) -> dict[str, object]:
    """Return explicit acceptance statistics for one raw daily history."""

    required = {"date", "code", "tradestatus", "isST", *NUMERIC_FIELDS}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing daily-history fields: {sorted(missing)}")
    if frame.empty:
        return {"rows": 0, "status": "NO_DATA"}
    dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    codes = frame["code"].dropna().astype(str).unique()
    if len(codes) != 1:
        raise ValueError("daily history must contain exactly one security")
    numeric = frame[list(NUMERIC_FIELDS)].apply(pd.to_numeric, errors="coerce")
    traded = pd.to_numeric(frame["tradestatus"], errors="coerce").eq(1)
    st = pd.to_numeric(frame["isST"], errors="coerce").eq(1)
    positive_turn = numeric["turn"] > 0
    implied_float_shares = numeric["volume"] / (numeric["turn"] / 100.0)
    valid_float = positive_turn & np.isfinite(implied_float_shares) & (implied_float_shares > 0)
    price_order_bad = (
        (numeric["low"] > numeric["high"])
        | (numeric["open"] < numeric["low"])
        | (numeric["open"] > numeric["high"])
        | (numeric["close"] < numeric["low"])
        | (numeric["close"] > numeric["high"])
    )
    zero_volume_traded = traded & numeric["volume"].eq(0)
    return {
        "status": "DATA",
        "code": codes[0],
        "rows": int(len(frame)),
        "first_date": dates.min().date().isoformat(),
        "last_date": dates.max().date().isoformat(),
        "duplicate_dates": int(dates.duplicated().sum()),
        "date_order_violations": int((dates.diff().dropna() <= pd.Timedelta(0)).sum()),
        "traded_rows": int(traded.sum()),
        "suspended_rows": int((~traded).sum()),
        "st_rows": int(st.sum()),
        "zero_volume_traded_rows": int(zero_volume_traded.sum()),
        "bad_ohlc_order_rows": int(price_order_bad.fillna(False).sum()),
        "missing_pct_change_rows": int(numeric["pctChg"].isna().sum()),
        "missing_pb_rows": int(numeric["pbMRQ"].isna().sum()),
        "nonpositive_pb_rows": int((numeric["pbMRQ"] <= 0).fillna(False).sum()),
        "implied_float_share_rows": int(valid_float.sum()),
        "implied_float_share_coverage": float(valid_float.mean()),
    }


def daily_history_acceptance_failures(quality: dict[str, object]) -> list[str]:
    """Apply hard integrity gates to one daily-history quality summary."""

    failures = []
    if quality.get("status") != "DATA" or int(quality.get("rows", 0)) == 0:
        failures.append("no rows returned for a security present in the PIT universe")
    for field in (
        "duplicate_dates",
        "date_order_violations",
        "zero_volume_traded_rows",
        "bad_ohlc_order_rows",
    ):
        if int(quality.get(field, 0)) != 0:
            failures.append(f"{field}={quality[field]}")
    return failures


def compare_adjustments(raw: pd.DataFrame, adjusted: pd.DataFrame) -> dict[str, object]:
    """Check adjustment behavior without treating adjusted prices as execution prices."""

    if raw.empty or adjusted.empty:
        return {"overlap_rows": 0, "status": "NO_OVERLAP"}
    columns = ["date", "code", "close", "volume", "amount", "turn", "pctChg", "pbMRQ"]
    left = raw[columns].copy()
    right = adjusted[columns].copy()
    merged = left.merge(right, on=["date", "code"], suffixes=("_raw", "_adjusted"), validate="one_to_one")
    for column in columns[2:]:
        merged[f"{column}_raw"] = pd.to_numeric(merged[f"{column}_raw"], errors="coerce")
        merged[f"{column}_adjusted"] = pd.to_numeric(
            merged[f"{column}_adjusted"], errors="coerce"
        )
    ratio = merged["close_adjusted"] / merged["close_raw"]
    finite_ratio = ratio.where(np.isfinite(ratio) & (ratio > 0))
    ratio_changes = finite_ratio.pct_change(fill_method=None).abs() > 1e-8
    invariant = {}
    for column in ("volume", "amount", "turn", "pctChg", "pbMRQ"):
        left_values = merged[f"{column}_raw"]
        right_values = merged[f"{column}_adjusted"]
        invariant[column] = bool(
            np.allclose(left_values, right_values, rtol=1e-10, atol=1e-12, equal_nan=True)
        )
    return {
        "status": "DATA",
        "overlap_rows": int(len(merged)),
        "adjustment_ratio_change_rows": int(ratio_changes.sum()),
        "adjustment_ratio_min": float(finite_ratio.min()),
        "adjustment_ratio_max": float(finite_ratio.max()),
        "invariant_fields": invariant,
    }
