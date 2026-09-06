"""Build exact-session dynamic LS, leg, and Active Long outcomes and tail labels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.experiments import require_protocol_freeze
from alpha_crowding.outcomes import (
    MatureTailSpec,
    build_mature_tail_labels,
    forward_window_outcome,
)


ROOT = Path(__file__).resolve().parents[1]
FACTOR_RETURNS = ROOT / "data" / "processed" / "factor_returns.parquet"
LEG_RETURNS = ROOT / "data" / "processed" / "factor_leg_returns.parquet"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
BENCHMARK = ROOT / "data" / "raw" / "benchmark" / "csi800.parquet"
CONFIG = ROOT / "config" / "outcomes.yaml"
OUTPUT = ROOT / "data" / "processed" / "dynamic_outcomes.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "dynamic_outcomes.json"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"


def _calendar() -> pd.DatetimeIndex:
    paths = list((ROOT / "data" / "raw" / "membership").glob("trade_calendar_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one production trade calendar; found {len(paths)}")
    frame = pd.read_csv(paths[0])
    return pd.DatetimeIndex(
        pd.to_datetime(
            frame.loc[
                pd.to_numeric(frame["is_trading_day"], errors="raise").eq(1),
                "calendar_date",
            ]
        )
    ).normalize().sort_values()


def main() -> int:
    require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    calendar = _calendar()
    factors = pd.read_parquet(FACTOR_RETURNS)
    legs = pd.read_parquet(LEG_RETURNS)
    benchmark = pd.read_parquet(BENCHMARK)
    for frame in (factors, legs, benchmark):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    benchmark_series = benchmark.set_index("date")["daily_return"]
    memberships = pd.read_parquet(MEMBERSHIPS, columns=["date", "factor"])
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    decisions = memberships.drop_duplicates().sort_values(["date", "factor"])
    research_horizons = sorted(
        {
            int(config["primary"]["horizon_trading_days"]),
            *[int(value) for value in config["secondary"]["horizons_trading_days"]],
        }
    )
    active_horizon = int(config["application"]["horizon_trading_days"])
    rows = []
    for factor, group in decisions.groupby("factor", sort=True):
        factor_frame = factors[factors["factor"].eq(factor)]
        ls_series = factor_frame.set_index("date")["long_short_return"]
        leg_series = {
            leg: legs[
                legs["factor"].eq(factor) & legs["leg"].eq(leg)
            ].set_index("date")["daily_return"]
            for leg in ("LONG", "SHORT")
        }
        for decision_at in group["date"]:
            for horizon in research_horizons:
                for family, series in (
                    ("research_ls", ls_series),
                    ("factor_long", leg_series["LONG"]),
                    ("factor_short", leg_series["SHORT"]),
                ):
                    measured = forward_window_outcome(
                        series,
                        calendar,
                        decision_at=decision_at,
                        horizon_sessions=horizon,
                    )
                    rows.append(
                        {
                            "factor": factor,
                            "target_family": family,
                            "membership_mode": "dynamic",
                            **measured,
                        }
                    )
            active = forward_window_outcome(
                leg_series["LONG"],
                calendar,
                decision_at=decision_at,
                horizon_sessions=active_horizon,
                benchmark_returns=benchmark_series,
            )
            rows.append(
                {
                    "factor": factor,
                    "target_family": "active_long",
                    "membership_mode": "dynamic",
                    **active,
                }
            )
    output = pd.DataFrame(rows)
    mature_mask = output["outcome_mature"] & output["label_end_at"].notna()
    mature = output.loc[mature_mask].copy()
    group_cols = ("factor", "target_family", "membership_mode", "horizon_sessions")
    quantiles = {
        0.10,
        *[float(value) for value in config["secondary"]["crash_quantiles"]],
    }
    for quantile in sorted(quantiles):
        suffix = f"q{int(round(quantile * 100)):02d}"
        labelled = build_mature_tail_labels(
            mature,
            sessions=calendar,
            group_cols=group_cols,
            spec=MatureTailSpec(
                quantile=quantile,
                lookback_sessions=int(config["primary"]["threshold_history_trading_days"]),
                min_history=int(config["primary"]["minimum_mature_weekly_observations"]),
            ),
            threshold_name=f"historical_tail_threshold_{suffix}",
            history_count_name=f"mature_history_count_{suffix}",
            label_name=f"crash_{suffix}",
        )
        for column in (
            f"historical_tail_threshold_{suffix}",
            f"mature_history_count_{suffix}",
            f"crash_{suffix}",
        ):
            output[column] = np.nan if "crash" not in column else pd.Series(pd.NA, index=output.index, dtype="boolean")
            output.loc[mature_mask, column] = labelled[column].to_numpy()
        output[f"mature_history_count_{suffix}"] = output[
            f"mature_history_count_{suffix}"
        ].fillna(0).astype("int64")
    output = output.sort_values(
        ["decision_at", "factor", "target_family", "horizon_sessions"]
    ).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_dynamic_factor_and_active_outcomes",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(output),
        "mature_outcome_rows": int(output["outcome_mature"].sum()),
        "main_label_rows": int(output["crash_q10"].notna().sum()),
        "main_crash_rows": int(output["crash_q10"].fillna(False).sum()),
        "first_main_label_date": str(output.loc[output["crash_q10"].notna(), "decision_at"].min().date()) if output["crash_q10"].notna().any() else None,
        "last_main_label_date": str(output.loc[output["crash_q10"].notna(), "decision_at"].max().date()) if output["crash_q10"].notna().any() else None,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
