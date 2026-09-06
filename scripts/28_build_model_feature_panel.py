"""Pivot the leg state table into one auditable row per factor decision."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "processed" / "state_panel.parquet"
OUTPUT = ROOT / "data" / "processed" / "model_feature_panel.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "model_feature_panel.json"

COMMON = [
    "market_return_20",
    "market_volatility_20",
    "market_volatility_60",
    "market_drawdown_252",
    "market_turnover_median",
    "market_illiquidity_median",
    "market_breadth_20",
    "cross_sectional_return_dispersion",
    "factor_return_20",
    "factor_volatility_20",
    "factor_volatility_60",
    "factor_drawdown_252",
    "signal_dispersion",
    "rank_ic_mean",
    "rank_ic_volatility",
]
LEG_FIELDS = [
    "crowding_state",
    "crowding_state_core",
    "generic_risk",
    "stress_trigger",
    "excess_sync_historical_z",
    "excess_eigen_historical_z",
    "excess_overlap_historical_z",
    "turnover_shock_z",
    "turnover_sync_z",
    "liquidity_stress_z",
    "factor_return_shock_z",
]


def main() -> int:
    state = pd.read_parquet(STATE)
    keys = ["decision_at", "factor"]
    if state.duplicated([*keys, "leg"]).any():
        raise ValueError("state panel must be unique by decision/factor/leg")
    for column in COMMON:
        if (state.groupby(keys, dropna=False)[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"common state differs across legs: {column}")
    common = state[keys + COMMON].drop_duplicates(keys)
    selected = state[state["leg"].isin(["LONG", "SHORT"])][keys + ["leg", *LEG_FIELDS]]
    wide = selected.pivot(index=keys, columns="leg", values=LEG_FIELDS)
    wide.columns = [f"{field}_{str(leg).lower()}" for field, leg in wide.columns]
    output = common.merge(wide.reset_index(), on=keys, validate="one_to_one")
    for leg in ("long", "short"):
        output[f"crowding_x_stress_{leg}"] = (
            output[f"crowding_state_{leg}"] * output[f"stress_trigger_{leg}"]
        )
    output = output.sort_values(keys).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_model_feature_panel_long_short_separated",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(output),
        "columns": list(output.columns),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
