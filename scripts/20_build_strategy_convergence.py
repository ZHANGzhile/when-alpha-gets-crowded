"""Build joint cross-factor Strategy Convergence placebo features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.measurement import (
    CONVERGENCE_ALGORITHM_VERSION,
    historical_zscore_by_group_trading_window,
    measure_strategy_convergence_placebos,
)


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
CONFIG = ROOT / "config" / "measurement.yaml"
CHECKPOINTS = ROOT / "data" / "processed" / "strategy_convergence_by_date"
OUTPUT = ROOT / "data" / "processed" / "strategy_convergence.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "strategy_convergence.json"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _calendar() -> pd.DatetimeIndex:
    dates = [
        pd.read_parquet(path, columns=["date"])["date"]
        for path in sorted(DAILY.glob("year=*/*.parquet"))
    ]
    return pd.DatetimeIndex(pd.concat(dates, ignore_index=True).unique()).normalize().sort_values()


def _process_date(
    decision_at: pd.Timestamp,
    memberships: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    directory = CHECKPOINTS / str(decision_at.date())
    marker = directory / "status.json"
    names = ["features", "edges", "placebo_features", "placebo_diagnostics", "placebo_memberships"]
    if marker.exists() and all((directory / f"{name}.parquet").exists() for name in names):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["source"] = "checkpoint"
        return payload
    date_memberships = memberships[memberships["date"].eq(decision_at)]
    placebo = config["placebo"]
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
    failures = []
    for leg in ("LONG", "SHORT"):
        try:
            result = measure_strategy_convergence_placebos(
                date_memberships,
                decision_at=decision_at,
                leg=leg,
                draws=int(placebo["draws"]),
                minimum_valid_draws=int(placebo["minimum_valid_draws"]),
                base_seed=int(placebo["seed"]),
                temperature=float(placebo["liquidity_match_temperature"]),
                maximum_mean_absolute_z_difference=float(
                    placebo["maximum_mean_absolute_liquidity_z_difference"]
                ),
                maximum_attempts_per_draw=int(placebo["maximum_attempts_per_draw"]),
            )
            parts["features"].append(result.features)
            parts["edges"].append(result.edges)
            parts["placebo_features"].append(result.placebo_features)
            parts["placebo_diagnostics"].append(result.placebo_diagnostics)
            parts["placebo_memberships"].append(result.placebo_memberships)
        except ValueError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append({"leg": leg, "reason": reason})
            factors = sorted(date_memberships["factor"].astype(str).unique())
            parts["features"].append(
                pd.DataFrame(
                    [
                        {
                            "decision_at": decision_at,
                            "leg": leg,
                            "factor": factor,
                            "feature": "strategy_convergence",
                            "actual": np.nan,
                            "placebo_mean": np.nan,
                            "placebo_std": np.nan,
                            "excess": np.nan,
                            "placebo_z": np.nan,
                            "valid_draws": 0,
                            "total_draws": int(placebo["draws"]),
                            "placebo_valid": False,
                            "invalid_reason": reason,
                            "algorithm_version": CONVERGENCE_ALGORITHM_VERSION,
                        }
                        for factor in factors
                    ]
                )
            )
    schemas = {
        "edges": ["decision_at", "leg", "factor_left", "factor_right", "jaccard", "intersection_count", "union_count"],
        "placebo_features": ["decision_at", "leg", "factor", "draw_id", "value", "joint_valid", "invalid_reason"],
        "placebo_diagnostics": ["decision_at", "leg", "factor", "draw_id", "valid", "invalid_reason"],
        "placebo_memberships": ["decision_at", "leg", "factor", "draw_id", "code", "industry"],
    }
    outputs = {}
    for name in names:
        if parts[name]:
            outputs[name] = pd.concat(parts[name], ignore_index=True)
        else:
            outputs[name] = pd.DataFrame(columns=schemas.get(name, []))
        _atomic_parquet(outputs[name], directory / f"{name}.parquet")
    payload = {
        "decision_at": str(decision_at.date()),
        "source": "computed",
        "feature_rows": len(outputs["features"]),
        "placebo_membership_rows": len(outputs["placebo_memberships"]),
        "failures": failures,
    }
    _atomic_json(payload, marker)
    return payload


def main(*, start_date: str | None, end_date: str | None, maximum_dates: int | None) -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    memberships = pd.read_parquet(MEMBERSHIPS)
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(memberships["date"].unique()))
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if maximum_dates is not None:
        if maximum_dates < 1:
            raise ValueError("maximum_dates must be positive")
        dates = dates[:maximum_dates]
    summaries = []
    for position, decision_at in enumerate(dates, start=1):
        summary = _process_date(decision_at, memberships, config)
        summaries.append(summary)
        print(
            f"strategy convergence {position}/{len(dates)} date={decision_at.date()} "
            f"source={summary['source']} failures={len(summary['failures'])}",
            flush=True,
        )
    features = pd.concat(
        [pd.read_parquet(CHECKPOINTS / str(date.date()) / "features.parquet") for date in dates],
        ignore_index=True,
    )
    standardized = historical_zscore_by_group_trading_window(
        features,
        _calendar(),
        value_col="excess",
        group_cols=("factor", "leg", "feature"),
        order_col="decision_at",
        lookback_sessions=int(config["historical_window_trading_days"]),
        min_periods=int(config["historical_min_weekly_observations"]),
    ).add_prefix("excess_")
    features = pd.concat([features, standardized.drop(columns="excess_value")], axis=1)
    _atomic_parquet(features, OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "production_joint_strategy_convergence",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_version": CONVERGENCE_ALGORITHM_VERSION,
        "decision_dates": len(dates),
        "feature_rows": len(features),
        "valid_feature_rows": int(features["placebo_valid"].sum()),
        "leg_failures": sum(len(item["failures"]) for item in summaries),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "checkpoint_directory": str(CHECKPOINTS.relative_to(ROOT)),
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--maximum-dates", type=int)
    args = parser.parse_args()
    raise SystemExit(main(start_date=args.start_date, end_date=args.end_date, maximum_dates=args.maximum_dates))
