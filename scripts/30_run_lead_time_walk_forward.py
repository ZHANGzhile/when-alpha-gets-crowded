"""Run frozen 0/5/10-session lead experiments with alert-time thresholds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.experiments import (
    attach_lead_features,
    binary_log_loss,
    build_lead_labels,
    moving_block_bootstrap_mean,
    require_protocol_freeze,
    run_annual_walk_forward,
)
from alpha_crowding.outcomes import MatureTailSpec


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "model_feature_panel.parquet"
OUTCOMES = ROOT / "data" / "processed" / "dynamic_outcomes.parquet"
CONFIG = ROOT / "config" / "experiments.yaml"
OUTCOME_CONFIG = ROOT / "config" / "outcomes.yaml"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"
OUTPUT = ROOT / "data" / "models" / "lead_time_predictions.parquet"
PANELS = ROOT / "data" / "models" / "lead_time_model_rows.parquet"
MANIFEST = ROOT / "data" / "models" / "lead_time_walk_forward.json"

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
    freeze = require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    outcome_config = yaml.safe_load(OUTCOME_CONFIG.read_text(encoding="utf-8"))
    calendar = _calendar()
    features = pd.read_parquet(FEATURES)
    outcomes = pd.read_parquet(OUTCOMES)
    outcomes = outcomes.loc[
        outcomes["membership_mode"].eq("dynamic")
        & outcomes["horizon_sessions"].eq(
            int(outcome_config["primary"]["horizon_trading_days"])
        )
        & outcomes["outcome_mature"].fillna(False)
    ].copy()
    spec = MatureTailSpec(
        quantile=float(outcome_config["primary"]["crash_quantile"]),
        lookback_sessions=int(
            outcome_config["primary"]["threshold_history_trading_days"]
        ),
        min_history=int(
            outcome_config["primary"]["minimum_mature_weekly_observations"]
        ),
    )

    prediction_parts: list[pd.DataFrame] = []
    panel_parts: list[pd.DataFrame] = []
    run_manifests: dict[str, object] = {}
    comparison_rows: list[dict[str, object]] = []
    for family in ("research_ls", "active_long"):
        family_outcomes = outcomes.loc[outcomes["target_family"].eq(family)].copy()
        feature_sets = _feature_sets(active=family == "active_long")
        for lead in [int(value) for value in config["lead_time_trading_days"]]:
            labels = build_lead_labels(
                family_outcomes,
                sessions=calendar,
                lead_sessions=lead,
                spec=spec,
            )
            panel = attach_lead_features(labels, features, sessions=calendar)
            run = run_annual_walk_forward(
                panel,
                feature_sets,
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
            mapping = panel[
                ["decision_at", "factor", "target_anchor_at", "feature_source_at"]
            ]
            predictions = run.predictions.merge(
                mapping, on=["decision_at", "factor"], validate="one_to_one"
            )
            predictions = predictions.rename(columns={"decision_at": "issue_at"})
            predictions["target_family"] = family
            predictions["lead_sessions"] = lead
            prediction_parts.append(predictions)
            audit_panel = panel[
                [
                    "decision_at", "target_anchor_at", "factor", "lead_sessions",
                    "feature_source_at", "feature_staleness_sessions",
                    "historical_tail_threshold", "mature_history_count", "target",
                    "label_start_at", "label_end_at",
                ]
            ].copy()
            audit_panel["target_family"] = family
            panel_parts.append(audit_panel.rename(columns={"decision_at": "issue_at"}))
            key = f"{family}|lead={lead}"
            run_manifests[key] = {
                **run.manifest,
                "lead_sessions": lead,
                "alert_time_thresholds_rebuilt": True,
                "maximum_feature_staleness_sessions": int(
                    panel["feature_staleness_sessions"].max()
                ),
            }

            weekly_losses: dict[str, pd.Series] = {}
            for model in feature_sets:
                scored = predictions.copy()
                scored["loss"] = binary_log_loss(
                    scored["target"], scored[f"probability_{model}"]
                )
                weekly_losses[model] = scored.groupby("target_anchor_at")["loss"].mean()
            for larger, smaller in (("M3", "M2"), ("M4", "M3")):
                improvement = weekly_losses[smaller] - weekly_losses[larger]
                bootstrap = moving_block_bootstrap_mean(
                    improvement,
                    block_length=int(config["bootstrap"]["primary_block_weeks"]),
                    repetitions=int(config["bootstrap"]["repetitions"]),
                    seed=42 + lead,
                )
                comparison_rows.append(
                    {
                        "target_family": family,
                        "lead_sessions": lead,
                        "comparison": f"{larger}_vs_{smaller}",
                        **asdict(bootstrap),
                    }
                )

    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["target_family", "lead_sessions", "issue_at", "factor"]
    )
    audit_rows = pd.concat(panel_parts, ignore_index=True).sort_values(
        ["target_family", "lead_sessions", "issue_at", "factor"]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT, index=False, compression="zstd")
    audit_rows.to_parquet(PANELS, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "frozen_zero_five_ten_session_lead_walk_forward",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": hashlib.sha256(FREEZE_MANIFEST.read_bytes()).hexdigest(),
        "lead_sessions": [int(value) for value in config["lead_time_trading_days"]],
        "prediction_rows": len(predictions),
        "audit_rows": len(audit_rows),
        "comparisons": comparison_rows,
        "runs": run_manifests,
        "outputs": {
            str(OUTPUT.relative_to(ROOT)): hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            str(PANELS.relative_to(ROOT)): hashlib.sha256(PANELS.read_bytes()).hexdigest(),
        },
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key != "runs"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
