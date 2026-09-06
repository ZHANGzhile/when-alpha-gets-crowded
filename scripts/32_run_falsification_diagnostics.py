"""Run protocol-gated context, time-misalignment, and discriminant-validity tests."""

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
    discriminant_validity_table,
    moving_block_bootstrap_mean,
    multivariate_residual_diagnostic,
    require_protocol_freeze,
    run_annual_walk_forward,
    time_misalign_features,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "model_feature_panel.parquet"
OUTCOMES = ROOT / "data" / "processed" / "dynamic_outcomes.parquet"
CONFIG = ROOT / "config" / "experiments.yaml"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"
PREDICTIONS = ROOT / "data" / "models" / "falsification_predictions.parquet"
PAIRWISE = ROOT / "data" / "results" / "discriminant_pairwise.parquet"
RESIDUAL = ROOT / "data" / "results" / "discriminant_residual.parquet"
MANIFEST = ROOT / "data" / "results" / "falsification_diagnostics.json"

MARKET = [
    "market_return_20", "market_volatility_20", "market_volatility_60",
    "market_drawdown_252", "market_turnover_median", "market_illiquidity_median",
    "market_breadth_20", "cross_sectional_return_dispersion",
]
FACTOR = [
    "factor_return_20", "factor_volatility_20", "factor_volatility_60",
    "factor_drawdown_252", "signal_dispersion", "rank_ic_mean", "rank_ic_volatility",
]


def _base_components(*, active: bool) -> tuple[list[str], list[str], list[str]]:
    legs = ["long"] if active else ["long", "short"]
    generic = [f"generic_risk_{leg}" for leg in legs]
    stress = [f"stress_trigger_{leg}" for leg in legs]
    m2 = [*MARKET, *FACTOR, *generic, *stress]
    crowding = [f"crowding_state_{leg}" for leg in legs]
    interactions = [f"crowding_x_stress_{leg}" for leg in legs]
    return m2, crowding, interactions


def _context_feature_sets(*, active: bool) -> dict[str, list[str]]:
    legs = ["long"] if active else ["long", "short"]
    m2, crowding, _ = _base_components(active=active)
    context = [
        f"placebo_mean_{feature}_{leg}"
        for leg in legs
        for feature in (
            "residual_sync",
            "eigen_concentration",
            "strategy_convergence",
        )
    ]
    m2_context = [*m2, *context]
    return {
        "M2_context": m2_context,
        "M3_context": [*m2_context, *crowding],
    }


def _misaligned_feature_sets(*, active: bool) -> dict[str, list[str]]:
    m2, crowding, interactions = _base_components(active=active)
    m3 = [*m2, *crowding]
    return {"M2": m2, "M3": m3, "M4": [*m3, *interactions]}


def _labels(outcomes: pd.DataFrame, family: str) -> pd.DataFrame:
    return outcomes.loc[
        outcomes["target_family"].eq(family)
        & outcomes["membership_mode"].eq("dynamic")
        & outcomes["horizon_sessions"].eq(20)
        & outcomes["crash_q10"].notna(),
        ["decision_at", "factor", "label_start_at", "label_end_at", "crash_q10"],
    ].rename(columns={"crash_q10": "target"})


def _run(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: dict[str, object],
    raw_data_cutoff: object,
):
    return run_annual_walk_forward(
        panel,
        feature_sets,
        evaluation_start=config["development_period"][0],
        evaluation_end=config["confirmatory_period"][1],
        raw_data_cutoff=raw_data_cutoff,
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


def _comparison(
    predictions: pd.DataFrame,
    *,
    larger: str,
    smaller: str,
    config: dict[str, object],
    seed: int,
) -> dict[str, object]:
    losses: dict[str, pd.Series] = {}
    for model in (larger, smaller):
        scored = predictions.copy()
        scored["loss"] = binary_log_loss(
            scored["target"], scored[f"probability_{model}"]
        )
        losses[model] = scored.groupby("decision_at")["loss"].mean()
    improvement = losses[smaller] - losses[larger]
    result = moving_block_bootstrap_mean(
        improvement,
        block_length=int(config["bootstrap"]["primary_block_weeks"]),
        repetitions=int(config["bootstrap"]["repetitions"]),
        seed=seed,
    )
    return {"comparison": f"{larger}_vs_{smaller}", **asdict(result)}


def _discriminant_tables(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_parts = []
    specifications: dict[str, list[str]] = {}
    for leg in ("long", "short"):
        crowding = f"crowding_state_{leg}"
        comparators = [
            "market_volatility_20",
            "factor_volatility_20",
            "factor_drawdown_252",
            f"liquidity_stress_z_{leg}",
            f"generic_risk_{leg}",
        ]
        part = discriminant_validity_table(features, [crowding], comparators)
        part.insert(0, "leg", leg.upper())
        pairwise_parts.append(part)
        specifications[crowding] = comparators
    pairwise = pd.concat(pairwise_parts, ignore_index=True)
    residual = multivariate_residual_diagnostic(features, specifications)
    return pairwise, residual


def main() -> int:
    freeze = require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    features = pd.read_parquet(FEATURES)
    outcomes = pd.read_parquet(OUTCOMES)
    pairwise, residual = _discriminant_tables(features)

    prediction_parts: list[pd.DataFrame] = []
    comparisons: list[dict[str, object]] = []
    run_manifests: dict[str, object] = {}
    for family in ("research_ls", "active_long"):
        panel = features.merge(_labels(outcomes, family), on=["decision_at", "factor"])
        active = family == "active_long"

        context_sets = _context_feature_sets(active=active)
        context_run = _run(panel, context_sets, config, freeze["raw_data_cutoff"])
        context_predictions = context_run.predictions.copy()
        context_predictions["target_family"] = family
        context_predictions["falsification"] = "placebo_context"
        context_predictions["misalignment_weeks"] = 0
        prediction_parts.append(context_predictions)
        comparisons.append(
            {
                "target_family": family,
                "falsification": "placebo_context",
                "misalignment_weeks": 0,
                **_comparison(
                    context_predictions,
                    larger="M3_context",
                    smaller="M2_context",
                    config=config,
                    seed=142,
                ),
            }
        )
        run_manifests[f"{family}|placebo_context"] = context_run.manifest

        _, crowding, _ = _base_components(active=active)
        for lag in [int(value) for value in config["placebos"]["time_misalignment_weeks"]]:
            shifted = time_misalign_features(panel, crowding, lag_weeks=lag)
            for leg in (["long"] if active else ["long", "short"]):
                shifted[f"crowding_x_stress_{leg}"] = (
                    shifted[f"crowding_state_{leg}"] * shifted[f"stress_trigger_{leg}"]
                )
            feature_sets = _misaligned_feature_sets(active=active)
            run = _run(shifted, feature_sets, config, freeze["raw_data_cutoff"])
            predictions = run.predictions.merge(
                shifted[["decision_at", "factor", "misaligned_source_at"]],
                on=["decision_at", "factor"],
                validate="one_to_one",
            )
            predictions["target_family"] = family
            predictions["falsification"] = "time_misalignment"
            predictions["misalignment_weeks"] = lag
            prediction_parts.append(predictions)
            comparisons.append(
                {
                    "target_family": family,
                    "falsification": "time_misalignment",
                    "misalignment_weeks": lag,
                    **_comparison(
                        predictions,
                        larger="M3",
                        smaller="M2",
                        config=config,
                        seed=242 + lag,
                    ),
                }
            )
            run_manifests[f"{family}|misaligned={lag}"] = run.manifest

    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False).sort_values(
        ["target_family", "falsification", "misalignment_weeks", "decision_at", "factor"]
    )
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    PAIRWISE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS, index=False, compression="zstd")
    pairwise.to_parquet(PAIRWISE, index=False, compression="zstd")
    residual.to_parquet(RESIDUAL, index=False, compression="zstd")
    outputs = [PREDICTIONS, PAIRWISE, RESIDUAL]
    payload = {
        "schema_version": 1,
        "purpose": "frozen_context_time_misalignment_and_discriminant_validity",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": hashlib.sha256(FREEZE_MANIFEST.read_bytes()).hexdigest(),
        "comparisons": comparisons,
        "pairwise_rows": len(pairwise),
        "residual_rows": len(residual),
        "runs": run_manifests,
        "outputs": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in outputs
        },
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key != "runs"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
