"""Run frozen M0-M4 walk-forward comparisons for research and active targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.experiments import (
    binary_log_loss,
    moving_block_bootstrap_mean,
    require_protocol_freeze,
    run_annual_walk_forward,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "model_feature_panel.parquet"
OUTCOMES = ROOT / "data" / "processed" / "dynamic_outcomes.parquet"
CONFIG = ROOT / "config" / "experiments.yaml"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"
OUTPUT = ROOT / "data" / "models" / "primary_walk_forward_predictions.parquet"
MANIFEST = ROOT / "data" / "models" / "primary_walk_forward.json"

MARKET = [
    "market_return_20", "market_volatility_20", "market_volatility_60",
    "market_drawdown_252", "market_turnover_median", "market_illiquidity_median",
    "market_breadth_20", "cross_sectional_return_dispersion",
]
FACTOR = [
    "factor_return_20", "factor_volatility_20", "factor_volatility_60",
    "factor_drawdown_252", "signal_dispersion", "rank_ic_mean", "rank_ic_volatility",
]


def _feature_sets(*, active: bool) -> dict[str, list[str]]:
    legs = ["long"] if active else ["long", "short"]
    generic = [f"generic_risk_{leg}" for leg in legs]
    stress = [f"stress_trigger_{leg}" for leg in legs]
    crowding = [f"crowding_state_{leg}" for leg in legs]
    interactions = [f"crowding_x_stress_{leg}" for leg in legs]
    m0 = list(MARKET)
    m1 = [*m0, *FACTOR]
    m2 = [*m1, *generic, *stress]
    m3 = [*m2, *crowding]
    return {"M0": m0, "M1": m1, "M2": m2, "M3": m3, "M4": [*m3, *interactions]}


def main() -> int:
    freeze = require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    features = pd.read_parquet(FEATURES)
    outcomes = pd.read_parquet(OUTCOMES)
    results = []
    family_manifests = {}
    comparison_rows = []
    for family in ("research_ls", "active_long"):
        labels = outcomes[
            outcomes["target_family"].eq(family)
            & outcomes["membership_mode"].eq("dynamic")
            & outcomes["horizon_sessions"].eq(20)
            & outcomes["crash_q10"].notna()
        ][["decision_at", "factor", "label_start_at", "label_end_at", "crash_q10"]].rename(
            columns={"crash_q10": "target"}
        )
        panel = features.merge(labels, on=["decision_at", "factor"], validate="one_to_one")
        feature_sets = _feature_sets(active=family == "active_long")
        run = run_annual_walk_forward(
            panel,
            feature_sets,
            evaluation_start=config["development_period"][0],
            evaluation_end=config["confirmatory_period"][1],
            raw_data_cutoff=freeze["raw_data_cutoff"],
            target_col="target",
            c_grid=config["estimator"]["c_grid"],
            inner_validation_weeks=int(config["estimator"]["inner_time_validation"]["validation_weeks_per_split"]),
            inner_splits=int(config["estimator"]["inner_time_validation"]["splits"]),
            minimum_inner_training_weeks=int(config["estimator"]["inner_time_validation"]["minimum_training_weeks"]),
        )
        predictions = run.predictions.copy()
        predictions["target_family"] = family
        results.append(predictions)
        family_manifests[family] = run.manifest
        weekly_losses = {}
        for model in feature_sets:
            frame = predictions.copy()
            frame["loss"] = binary_log_loss(frame["target"], frame[f"probability_{model}"])
            weekly_losses[model] = frame.groupby("decision_at")["loss"].mean()
        for larger, smaller in (("M3", "M2"), ("M4", "M3")):
            improvement = weekly_losses[smaller] - weekly_losses[larger]
            bootstrap = moving_block_bootstrap_mean(
                improvement,
                block_length=int(config["bootstrap"]["primary_block_weeks"]),
                repetitions=int(config["bootstrap"]["repetitions"]),
                seed=42,
            )
            comparison_rows.append(
                {"target_family": family, "comparison": f"{larger}_vs_{smaller}", **asdict(bootstrap)}
            )
    predictions = pd.concat(results, ignore_index=True).sort_values(
        ["target_family", "decision_at", "factor"]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "frozen_primary_m0_m4_walk_forward",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": hashlib.sha256(FREEZE_MANIFEST.read_bytes()).hexdigest(),
        "prediction_rows": len(predictions),
        "comparisons": comparison_rows,
        "families": family_manifests,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "families"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
