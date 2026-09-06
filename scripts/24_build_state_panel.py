"""Assemble the model-ready C/G/S state panel without changing component weights."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.measurement import compute_generic_risk, compute_stress_trigger


ROOT = Path(__file__).resolve().parents[1]
CROWDING = ROOT / "data" / "processed" / "crowding_state.parquet"
STRUCTURAL = ROOT / "data" / "processed" / "structural_features_core.parquet"
PORTFOLIO_RISK = ROOT / "data" / "processed" / "portfolio_risk_features.parquet"
RETURN_STRESS = ROOT / "data" / "processed" / "factor_return_stress.parquet"
MARKET_STATE = ROOT / "data" / "processed" / "market_state.parquet"
FACTOR_STATE = ROOT / "data" / "processed" / "factor_state.parquet"
OUTPUT = ROOT / "data" / "processed" / "state_panel.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "state_panel.json"


def main() -> int:
    keys = ["decision_at", "factor", "leg"]
    crowding = pd.read_parquet(CROWDING)
    structural = pd.read_parquet(STRUCTURAL)
    structural_wide = structural.pivot(
        index=keys, columns="feature", values="actual_historical_z"
    ).rename(
        columns={
            "residual_sync": "raw_sync_z",
            "eigen_concentration": "raw_eigen_z",
        }
    ).reset_index()
    risk = pd.read_parquet(PORTFOLIO_RISK)
    returns = pd.read_parquet(RETURN_STRESS)
    market = pd.read_parquet(MARKET_STATE)
    factor_state = pd.read_parquet(FACTOR_STATE)
    panel = crowding.merge(structural_wide, on=keys, how="outer", validate="one_to_one")
    panel = panel.merge(risk, on=keys, how="outer", validate="one_to_one", suffixes=("", "_risk"))
    panel = panel.merge(
        returns[keys + ["factor_return_shock", "factor_return_recent", "factor_return_history_n"]],
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    market_columns = [
        "decision_at",
        "market_return_20",
        "market_volatility_20",
        "market_volatility_60",
        "market_drawdown_252",
        "market_turnover_median",
        "market_illiquidity_median",
        "market_breadth_20",
        "cross_sectional_return_dispersion",
        "cross_sectional_assets",
    ]
    factor_columns = [
        "decision_at",
        "factor",
        "factor_return_20",
        "factor_volatility_20",
        "factor_volatility_60",
        "factor_drawdown_252",
        "signal_dispersion",
        "rank_ic_mean",
        "rank_ic_volatility",
        "rank_ic_history_n",
        "rank_ic_latest_realized_at",
    ]
    panel = panel.merge(
        market[market_columns], on="decision_at", how="left", validate="many_to_one"
    )
    panel = panel.merge(
        factor_state[factor_columns],
        on=["decision_at", "factor"],
        how="left",
        validate="many_to_one",
    )
    panel["turnover_level_z"] = panel["turnover_level_historical_z"]
    panel["illiquidity_level_z"] = panel["illiquidity_level_historical_z"]
    generic = compute_generic_risk(panel)
    panel["turnover_shock_z"] = panel["turnover_shock_historical_z"]
    panel["turnover_sync_z"] = panel["turnover_sync_historical_z"]
    panel["liquidity_stress_z"] = panel["liquidity_shock_historical_z"]
    panel["factor_return_shock_z"] = panel["factor_return_shock"]
    stress = compute_stress_trigger(panel)
    panel = pd.concat([panel, generic, stress], axis=1)
    panel = panel.sort_values(keys).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_model_ready_market_factor_c_g_s_state_panel",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(panel),
        "crowding_valid_rows": int(panel["crowding_state_valid"].sum()),
        "generic_risk_valid_rows": int(panel["generic_risk_valid"].sum()),
        "stress_trigger_valid_rows": int(panel["stress_trigger_valid"].sum()),
        "all_states_valid_rows": int((panel["crowding_state_valid"] & panel["generic_risk_valid"] & panel["stress_trigger_valid"]).sum()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
