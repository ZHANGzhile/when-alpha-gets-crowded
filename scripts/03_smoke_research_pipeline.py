"""Run the M0-M4 comparison on synthetic data to verify engineering only."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_crowding.experiments import (
    moving_block_bootstrap_mean,
    run_annual_walk_forward,
    weekly_average_loss,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "smoke" / "m0_m4_summary.json"

FEATURE_SETS = {
    "M0": ["market_state"],
    "M1": ["market_state", "factor_state"],
    "M2": ["market_state", "factor_state", "generic_risk", "stress"],
    "M3": ["market_state", "factor_state", "generic_risk", "stress", "crowding"],
    "M4": [
        "market_state",
        "factor_state",
        "generic_risk",
        "stress",
        "crowding",
        "crowding_x_stress",
    ],
}


def synthetic_panel(seed: int = 42, weeks: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-08", periods=weeks, freq="W-FRI")
    factor_offsets = {"MOM": 0.20, "REV": -0.10, "LOWVOL": -0.25, "VALUE": 0.05}
    rows = []
    for date in dates:
        market = rng.normal()
        for factor, offset in factor_offsets.items():
            factor_state = 0.4 * market + rng.normal(scale=0.9)
            generic = 0.3 * market + rng.normal(scale=0.9)
            stress = rng.normal()
            crowding = rng.normal()
            interaction = crowding * stress
            logit = (
                -1.9
                + offset
                + 0.25 * market
                + 0.20 * factor_state
                + 0.25 * generic
                + 0.35 * stress
                + 1.10 * crowding
                + 0.70 * interaction
            )
            probability = 1.0 / (1.0 + np.exp(-logit))
            rows.append(
                {
                    "decision_at": date,
                    "label_start_at": date + pd.offsets.BDay(1),
                    "label_end_at": date + pd.offsets.BDay(20),
                    "factor": factor,
                    "market_state": market,
                    "factor_state": factor_state,
                    "generic_risk": generic,
                    "stress": stress,
                    "crowding": crowding,
                    "crowding_x_stress": interaction,
                    "target": int(rng.random() < probability),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    panel = synthetic_panel()
    result = run_annual_walk_forward(
        panel,
        FEATURE_SETS,
        evaluation_start="2020-01-01",
        evaluation_end="2023-12-31",
        raw_data_cutoff="2024-01-31",
        c_grid=[0.1, 1.0, 10.0],
        inner_validation_weeks=13,
        inner_splits=2,
        minimum_inner_training_weeks=104,
    )

    weekly_losses = {
        model: weekly_average_loss(
            result.predictions, probability_col=f"probability_{model}"
        )
        for model in FEATURE_SETS
    }
    delta_32 = weekly_losses["M2"] - weekly_losses["M3"]
    delta_43 = weekly_losses["M3"] - weekly_losses["M4"]
    summary = {
        "synthetic_engineering_smoke_only": True,
        "seed": 42,
        "outer_scheme": "annual_expanding",
        "training_rows_by_fold": {
            fold["fold"]: fold["train_rows"] for fold in result.manifest["folds"]
        },
        "evaluation_rows": len(result.predictions),
        "evaluation_weeks": int(result.predictions["decision_at"].nunique()),
        "weekly_log_loss": {
            model: float(loss.mean()) for model, loss in weekly_losses.items()
        },
        "delta_log_loss_M2_minus_M3": asdict(
            moving_block_bootstrap_mean(
                delta_32, block_length=13, repetitions=1_000, seed=42
            )
        ),
        "delta_log_loss_M3_minus_M4": asdict(
            moving_block_bootstrap_mean(
                delta_43, block_length=13, repetitions=1_000, seed=43
            )
        ),
        "manifest": result.manifest,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
