"""Run M2/M3 OOS models for all continuous pseudo strategies and rank the real increment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.experiments import (
    binary_log_loss,
    monte_carlo_incremental_rank,
    require_protocol_freeze,
    run_annual_walk_forward,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "pseudo_model_feature_panel.parquet"
OUTCOMES = ROOT / "data" / "processed" / "pseudo_dynamic_outcomes.parquet"
REAL_PREDICTIONS = ROOT / "data" / "models" / "primary_walk_forward_predictions.parquet"
CONFIG = ROOT / "config" / "experiments.yaml"
CHECKPOINTS = ROOT / "data" / "models" / "pseudo_walk_forward_by_strategy"
PREDICTION_OUTPUT = ROOT / "data" / "models" / "pseudo_walk_forward_predictions.parquet"
SUMMARY_OUTPUT = ROOT / "data" / "models" / "pseudo_incremental_values.parquet"
MANIFEST = ROOT / "data" / "models" / "continuous_pseudo_walk_forward.json"
FEATURE_MANIFEST = (
    ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_model_features.json"
)
OUTCOME_MANIFEST = ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_outcomes.json"
REAL_MANIFEST = ROOT / "data" / "models" / "primary_walk_forward.json"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"

MARKET = [
    "market_return_20",
    "market_volatility_20",
    "market_volatility_60",
    "market_drawdown_252",
    "market_turnover_median",
    "market_illiquidity_median",
    "market_breadth_20",
    "cross_sectional_return_dispersion",
]
FACTOR = [
    "factor_return_20",
    "factor_volatility_20",
    "factor_volatility_60",
    "factor_drawdown_252",
    "signal_dispersion",
    "rank_ic_mean",
    "rank_ic_volatility",
]


def _feature_sets() -> dict[str, list[str]]:
    generic = ["generic_risk_long", "generic_risk_short"]
    stress = ["stress_trigger_long", "stress_trigger_short"]
    crowding = ["crowding_state_long", "crowding_state_short"]
    m2 = [*MARKET, *FACTOR, *generic, *stress]
    return {"M2": m2, "M3": [*m2, *crowding]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _incremental_value(predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    frame = predictions.copy()
    frame["loss_m2"] = binary_log_loss(frame["target"], frame["probability_M2"])
    frame["loss_m3"] = binary_log_loss(frame["target"], frame["probability_M3"])
    weekly = frame.groupby("decision_at")[["loss_m2", "loss_m3"]].mean()
    weekly["incremental_value_m3_vs_m2"] = weekly["loss_m2"] - weekly["loss_m3"]
    return float(weekly["incremental_value_m3_vs_m2"].mean()), weekly.reset_index()


def _strategy_run(
    pseudo_id: str,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, object],
    freeze: dict[str, object],
    *,
    feature_sha256: str,
    outcome_sha256: str,
    freeze_sha256: str,
) -> dict[str, object]:
    directory = CHECKPOINTS / pseudo_id
    prediction_path = directory / "predictions.parquet"
    weekly_path = directory / "weekly_increment.parquet"
    marker = directory / "status.json"
    signature = {
        "pseudo_strategy_id": pseudo_id,
        "feature_sha256": feature_sha256,
        "outcome_sha256": outcome_sha256,
        "experiment_config_sha256": _sha256(CONFIG),
        "protocol_freeze_sha256": freeze_sha256,
    }
    if marker.exists() and prediction_path.exists() and weekly_path.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    strategy_features = features.loc[
        features["pseudo_strategy_id"].eq(pseudo_id)
    ].rename(columns={"original_factor": "factor"})
    labels = outcomes.loc[
        outcomes["pseudo_strategy_id"].eq(pseudo_id)
        & outcomes["target_family"].eq("research_ls")
        & outcomes["membership_mode"].eq("continuous_pseudo_dynamic")
        & outcomes["horizon_sessions"].eq(20)
        & outcomes["crash_q10"].notna(),
        ["decision_at", "original_factor", "label_start_at", "label_end_at", "crash_q10"],
    ].rename(columns={"original_factor": "factor", "crash_q10": "target"})
    panel = strategy_features.merge(
        labels, on=["decision_at", "factor"], validate="one_to_one"
    )
    try:
        run = run_annual_walk_forward(
            panel,
            _feature_sets(),
            evaluation_start=config["development_period"][0],
            evaluation_end=config["confirmatory_period"][1],
            raw_data_cutoff=freeze["raw_data_cutoff"],
            target_col="target",
            c_grid=config["estimator"]["c_grid"],
            inner_validation_weeks=int(
                config["estimator"]["inner_time_validation"]["validation_weeks_per_split"]
            ),
            inner_splits=int(config["estimator"]["inner_time_validation"]["splits"]),
            minimum_inner_training_weeks=int(
                config["estimator"]["inner_time_validation"]["minimum_training_weeks"]
            ),
        )
        predictions = run.predictions.copy()
        predictions["pseudo_strategy_id"] = pseudo_id
        incremental, weekly = _incremental_value(predictions)
        weekly["pseudo_strategy_id"] = pseudo_id
        valid = True
        reason = None
        run_manifest = run.manifest
    except ValueError as exc:
        predictions = pd.DataFrame(
            columns=[
                "decision_at",
                "factor",
                "target",
                "probability_M2",
                "probability_M3",
                "pseudo_strategy_id",
            ]
        )
        weekly = pd.DataFrame(
            columns=[
                "decision_at",
                "loss_m2",
                "loss_m3",
                "incremental_value_m3_vs_m2",
                "pseudo_strategy_id",
            ]
        )
        incremental = None
        valid = False
        reason = f"{type(exc).__name__}: {exc}"
        run_manifest = None
    _atomic_parquet(predictions, prediction_path)
    _atomic_parquet(weekly, weekly_path)
    payload = {
        **signature,
        "source": "computed",
        "valid": valid,
        "invalid_reason": reason,
        "prediction_rows": len(predictions),
        "evaluation_weeks": int(weekly["decision_at"].nunique()),
        "incremental_value_m3_vs_m2": incremental,
        "walk_forward": run_manifest,
        "prediction_sha256": _sha256(prediction_path),
        "weekly_sha256": _sha256(weekly_path),
    }
    _atomic_json(payload, marker)
    return payload


def main(*, maximum_strategies: int | None) -> int:
    freeze = require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    for name, path in (
        ("pseudo model features", FEATURE_MANIFEST),
        ("pseudo outcomes", OUTCOME_MANIFEST),
        ("real primary predictions", REAL_MANIFEST),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise ValueError(f"{name} manifest must be COMPLETE")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    features = pd.read_parquet(FEATURES)
    outcomes = pd.read_parquet(OUTCOMES)
    for frame in (features, outcomes):
        frame["decision_at"] = pd.to_datetime(frame["decision_at"]).dt.normalize()
    pseudo_ids = sorted(features["pseudo_strategy_id"].unique())
    expected = int(config["placebos"]["continuous_strategy_count"])
    if len(pseudo_ids) != expected:
        raise ValueError(f"pseudo prediction requires {expected} strategies")
    if maximum_strategies is not None:
        if maximum_strategies < 1 or maximum_strategies > expected:
            raise ValueError(f"maximum_strategies must be between 1 and {expected}")
        pseudo_ids = pseudo_ids[:maximum_strategies]
    feature_sha = _sha256(FEATURES)
    outcome_sha = _sha256(OUTCOMES)
    freeze_sha = _sha256(FREEZE_MANIFEST)
    summaries = []
    for number, pseudo_id in enumerate(pseudo_ids, start=1):
        summary = _strategy_run(
            pseudo_id,
            features,
            outcomes,
            config,
            freeze,
            feature_sha256=feature_sha,
            outcome_sha256=outcome_sha,
            freeze_sha256=freeze_sha,
        )
        summaries.append(summary)
        print(
            f"pseudo OOS {number}/{len(pseudo_ids)} strategy={pseudo_id} "
            f"source={summary['source']} valid={summary['valid']}",
            flush=True,
        )
    summary_frame = pd.DataFrame(
        [
            {
                "pseudo_strategy_id": item["pseudo_strategy_id"],
                "valid": item["valid"],
                "invalid_reason": item["invalid_reason"],
                "prediction_rows": item["prediction_rows"],
                "evaluation_weeks": item["evaluation_weeks"],
                "incremental_value_m3_vs_m2": item["incremental_value_m3_vs_m2"],
            }
            for item in summaries
        ]
    )
    valid_ids = summary_frame.loc[summary_frame["valid"], "pseudo_strategy_id"].tolist()
    predictions = (
        pd.concat(
            [
                pd.read_parquet(CHECKPOINTS / pseudo_id / "predictions.parquet")
                for pseudo_id in valid_ids
            ],
            ignore_index=True,
        )
        if valid_ids
        else pd.DataFrame()
    )
    _atomic_parquet(predictions, PREDICTION_OUTPUT)
    _atomic_parquet(summary_frame, SUMMARY_OUTPUT)

    comparison = None
    status = "PARTIAL" if maximum_strategies is not None else "COMPLETE"
    if maximum_strategies is None:
        real = pd.read_parquet(REAL_PREDICTIONS)
        real = real.loc[real["target_family"].eq("research_ls")]
        observed, _ = _incremental_value(real)
        rank = monte_carlo_incremental_rank(
            observed,
            summary_frame["incremental_value_m3_vs_m2"],
            minimum_placebos=int(config["placebos"]["continuous_minimum_comparators"]),
        )
        comparison = rank.__dict__
    payload = {
        "schema_version": 1,
        "purpose": "continuous_full_pipeline_placebo_m3_vs_m2_oos_comparison",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": freeze_sha,
        "strategy_count": len(pseudo_ids),
        "configured_strategy_count": expected,
        "valid_strategy_count": len(valid_ids),
        "prediction_rows": len(predictions),
        "comparison": comparison,
        "outputs": {
            str(PREDICTION_OUTPUT.relative_to(ROOT)): _sha256(PREDICTION_OUTPUT),
            str(SUMMARY_OUTPUT.relative_to(ROOT)): _sha256(SUMMARY_OUTPUT),
        },
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-strategies", type=int)
    args = parser.parse_args()
    raise SystemExit(main(maximum_strategies=args.maximum_strategies))
