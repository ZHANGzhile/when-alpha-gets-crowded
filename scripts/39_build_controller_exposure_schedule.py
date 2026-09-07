"""Build strictly OOS next-session controller exposure schedules."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.backtest import (
    build_oos_exposure_schedule,
    build_volatility_control_weights,
    probability_to_active_weight,
)
from alpha_crowding.experiments import require_protocol_freeze


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "models" / "primary_walk_forward_predictions.parquet"
FACTOR_LEG_RETURNS = ROOT / "data" / "processed" / "factor_leg_returns.parquet"
BENCHMARK = ROOT / "data" / "raw" / "benchmark" / "csi800.parquet"
CONFIG = ROOT / "config" / "controller.yaml"
OUTPUT = ROOT / "data" / "processed" / "controller_exposure_schedule.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "controller_exposure_schedule.json"
PREDICTION_MANIFEST = ROOT / "data" / "models" / "primary_walk_forward.json"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for path in (PREDICTIONS, FACTOR_LEG_RETURNS, BENCHMARK, CONFIG):
        if not path.exists():
            raise FileNotFoundError(f"required controller input is missing: {path}")
    prediction_manifest = json.loads(PREDICTION_MANIFEST.read_text(encoding="utf-8"))
    if prediction_manifest.get("status") != "COMPLETE":
        raise ValueError("controller requires a COMPLETE primary prediction manifest")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = list(config["information_models"])
    probability_columns = [f"probability_{model}" for model in models]
    predictions = pd.read_parquet(PREDICTIONS)
    active = predictions.loc[
        predictions["target_family"].eq("active_long"),
        ["decision_at", "factor", *probability_columns],
    ]
    tiers = [
        (float(row["upper_percentile"]), float(row["active_weight"]))
        for row in config["primary_policy"]["tiers"]
    ]
    schedule = build_oos_exposure_schedule(
        active,
        _calendar(),
        probability_columns=probability_columns,
        minimum_history_weeks=int(config["primary_policy"]["minimum_history_weeks"]),
        bands=tiers,
    )
    schedule["active_weight_full"] = 1.0
    schedule["active_weight_benchmark"] = 0.0
    schedule["active_weight_fixed_low"] = float(
        config["fixed_low_exposure"]["active_weight"]
    )
    leg_returns = pd.read_parquet(FACTOR_LEG_RETURNS)
    factor_long = leg_returns.loc[
        leg_returns["leg"].eq("LONG"), ["date", "factor", "daily_return"]
    ].copy()
    benchmark = pd.read_parquet(BENCHMARK, columns=["date", "daily_return"])
    for frame in (factor_long, benchmark):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    active_returns = factor_long.merge(
        benchmark.rename(columns={"daily_return": "benchmark_return"}),
        on="date",
        how="left",
        validate="many_to_one",
    )
    if active_returns["benchmark_return"].isna().any():
        raise ValueError("factor-long history lacks aligned CSI800 returns")
    active_returns["active_return"] = (
        active_returns["daily_return"] - active_returns["benchmark_return"]
    )
    volatility_config = config["volatility_control"]
    volatility = build_volatility_control_weights(
        schedule[["decision_at", "factor"]],
        active_returns[["date", "factor", "active_return"]],
        lookback_sessions=int(volatility_config["lookback_sessions"]),
        minimum_observations=int(volatility_config["minimum_observations"]),
        annualized_target_volatility=float(
            volatility_config["annualized_target_volatility"]
        ),
        minimum_active_weight=float(volatility_config["minimum_active_weight"]),
        maximum_active_weight=float(volatility_config["maximum_active_weight"]),
        periods_per_year=int(volatility_config["periods_per_year"]),
    )
    schedule = schedule.merge(
        volatility,
        on=["decision_at", "factor"],
        how="left",
        validate="one_to_one",
    )
    minimum = float(config["sensitivity_policy"]["minimum_active_weight"])
    for model in models:
        schedule[f"active_weight_{model}_probability_sensitivity"] = schedule[
            f"probability_{model}"
        ].map(lambda value: probability_to_active_weight(value, minimum_weight=minimum))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(OUTPUT, index=False, compression="zstd")
    primary_weight_columns = [f"active_weight_{model}" for model in models]
    payload = {
        "schema_version": 1,
        "purpose": "strictly_historical_oos_controller_exposure_schedule",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": _sha256(FREEZE_MANIFEST),
        "prediction_sha256": _sha256(PREDICTIONS),
        "factor_leg_returns_sha256": _sha256(FACTOR_LEG_RETURNS),
        "benchmark_sha256": _sha256(BENCHMARK),
        "rows": len(schedule),
        "models": models,
        "minimum_history_weeks": int(config["primary_policy"]["minimum_history_weeks"]),
        "fixed_low_exposure": config["fixed_low_exposure"],
        "volatility_control": config["volatility_control"],
        "warmup_rows": int(schedule[primary_weight_columns].isna().any(axis=1).sum()),
        "first_executable_date": (
            str(
                schedule.loc[
                    schedule[primary_weight_columns].notna().all(axis=1),
                    "effective_at",
                ]
                .min()
                .date()
            )
            if schedule[primary_weight_columns].notna().all(axis=1).any()
            else None
        ),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(OUTPUT),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
