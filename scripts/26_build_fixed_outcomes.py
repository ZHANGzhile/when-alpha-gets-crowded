"""Build 20-session fixed-membership mechanism outcomes with drift."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.experiments import require_protocol_freeze
from alpha_crowding.outcomes import (
    MatureTailSpec,
    build_mature_tail_labels,
    fixed_membership_leg_path,
    forward_window_outcome,
)


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
CONFIG = ROOT / "config" / "outcomes.yaml"
CHECKPOINTS = ROOT / "data" / "processed" / "fixed_outcomes_by_date"
OUTPUT = ROOT / "data" / "processed" / "fixed_outcomes.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "fixed_outcomes.json"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _calendar() -> pd.DatetimeIndex:
    paths = list((ROOT / "data" / "raw" / "membership").glob("trade_calendar_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one production trade calendar; found {len(paths)}")
    frame = pd.read_csv(paths[0])
    return pd.DatetimeIndex(
        pd.to_datetime(frame.loc[pd.to_numeric(frame["is_trading_day"]).eq(1), "calendar_date"])
    ).normalize().sort_values()


def _read_forward_years(year: int) -> pd.DataFrame:
    paths = []
    for wanted in (year, year + 1):
        paths.extend(sorted((DAILY / f"year={wanted}").glob("*.parquet")))
    return pd.concat(
        [pd.read_parquet(path, columns=["date", "code", "daily_return"]) for path in paths],
        ignore_index=True,
    )


def _process_date(
    decision_at: pd.Timestamp,
    date_memberships: pd.DataFrame,
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    checkpoint = CHECKPOINTS / f"{decision_at.date()}.parquet"
    if checkpoint.exists():
        return pd.read_parquet(checkpoint)
    rows = []
    for factor, factor_rows in date_memberships.groupby("factor", sort=True):
        paths = {}
        for leg in ("LONG", "SHORT"):
            members = factor_rows[factor_rows["leg"].eq(leg)][["code", "weight"]]
            codes = members["code"].astype(str)
            security_returns = daily[daily["code"].isin(codes)]
            paths[leg] = fixed_membership_leg_path(
                members,
                security_returns,
                calendar,
                decision_at=decision_at,
                horizon_sessions=horizon,
            )
        series = {
            leg: path.set_index("date")["daily_return"]
            for leg, path in paths.items()
        }
        if len(paths["LONG"]) and len(paths["SHORT"]):
            ls = series["LONG"] - series["SHORT"]
        else:
            ls = pd.Series(dtype=float)
        for family, returns in (
            ("research_ls", ls),
            ("factor_long", series["LONG"]),
            ("factor_short", series["SHORT"]),
        ):
            measured = forward_window_outcome(
                returns,
                calendar,
                decision_at=decision_at,
                horizon_sessions=horizon,
            )
            rows.append(
                {
                    "factor": factor,
                    "target_family": family,
                    "membership_mode": "fixed",
                    **measured,
                }
            )
    output = pd.DataFrame(rows)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".parquet.tmp")
    output.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, checkpoint)
    return output


def main() -> int:
    require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    horizon = int(config["primary"]["horizon_trading_days"])
    calendar = _calendar()
    memberships = pd.read_parquet(MEMBERSHIPS)
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(memberships["date"].unique()))
    parts = []
    loaded_year = None
    daily = pd.DataFrame()
    for position, decision_at in enumerate(dates, start=1):
        if loaded_year != decision_at.year:
            daily = _read_forward_years(decision_at.year)
            daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
            loaded_year = decision_at.year
        part = _process_date(
            decision_at,
            memberships[memberships["date"].eq(decision_at)],
            daily,
            calendar,
            horizon,
        )
        parts.append(part)
        print(f"fixed outcomes {position}/{len(dates)} date={decision_at.date()}", flush=True)
    output = pd.concat(parts, ignore_index=True)
    mature_mask = output["outcome_mature"] & output["label_end_at"].notna()
    mature = output.loc[mature_mask].copy()
    quantiles = {0.10, *[float(value) for value in config["secondary"]["crash_quantiles"]]}
    for quantile in sorted(quantiles):
        suffix = f"q{int(round(quantile * 100)):02d}"
        labelled = build_mature_tail_labels(
            mature,
            sessions=calendar,
            group_cols=("factor", "target_family", "membership_mode", "horizon_sessions"),
            spec=MatureTailSpec(
                quantile=quantile,
                lookback_sessions=int(config["primary"]["threshold_history_trading_days"]),
                min_history=int(config["primary"]["minimum_mature_weekly_observations"]),
            ),
            threshold_name=f"historical_tail_threshold_{suffix}",
            history_count_name=f"mature_history_count_{suffix}",
            label_name=f"crash_{suffix}",
        )
        output[f"historical_tail_threshold_{suffix}"] = np.nan
        output[f"mature_history_count_{suffix}"] = 0
        output[f"crash_{suffix}"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
        for column in (
            f"historical_tail_threshold_{suffix}",
            f"mature_history_count_{suffix}",
            f"crash_{suffix}",
        ):
            output.loc[mature_mask, column] = labelled[column].to_numpy()
        output[f"mature_history_count_{suffix}"] = output[f"mature_history_count_{suffix}"].astype("int64")
    output = output.sort_values(["decision_at", "factor", "target_family"]).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_fixed_membership_mechanism_outcomes",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(output),
        "mature_outcome_rows": int(output["outcome_mature"].sum()),
        "main_label_rows": int(output["crash_q10"].notna().sum()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "checkpoint_directory": str(CHECKPOINTS.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
