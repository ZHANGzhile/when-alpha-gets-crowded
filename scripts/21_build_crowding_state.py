"""Assemble the full and core structural Crowding State from audited features."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.measurement import compute_crowding_state


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "data" / "processed" / "structural_features_core.parquet"
CONVERGENCE = ROOT / "data" / "processed" / "strategy_convergence.parquet"
OUTPUT = ROOT / "data" / "processed" / "crowding_state.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "crowding_state.json"

FEATURE_COLUMNS = {
    "residual_sync": "excess_sync_historical_z",
    "eigen_concentration": "excess_eigen_historical_z",
    "strategy_convergence": "excess_overlap_historical_z",
}


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def main() -> int:
    long_features = pd.concat(
        [pd.read_parquet(CORE), pd.read_parquet(CONVERGENCE)], ignore_index=True
    )
    required = {"decision_at", "factor", "leg", "feature", "excess_historical_z"}
    missing = required - set(long_features.columns)
    if missing:
        raise KeyError(f"missing structural feature fields: {sorted(missing)}")
    duplicates = long_features.duplicated(["decision_at", "factor", "leg", "feature"])
    if duplicates.any():
        raise ValueError("structural feature keys are not unique")
    wide = long_features.pivot(
        index=["decision_at", "factor", "leg"],
        columns="feature",
        values="excess_historical_z",
    ).rename(columns=FEATURE_COLUMNS)
    required_components = set(FEATURE_COLUMNS.values())
    absent_components = required_components - set(wide.columns)
    if absent_components:
        raise ValueError(f"missing Crowding State components: {sorted(absent_components)}")
    full = compute_crowding_state(wide)
    core = compute_crowding_state(
        wide,
        components=("excess_sync_historical_z", "excess_eigen_historical_z"),
    ).rename(
        columns={
            "crowding_state": "crowding_state_core",
            "crowding_state_valid": "crowding_state_core_valid",
            "crowding_state_component_count": "crowding_state_core_component_count",
        }
    )
    output = pd.concat([wide, full, core], axis=1).reset_index()
    _atomic_parquet(output, OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "production_structural_crowding_state",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(output),
        "valid_full_rows": int(output["crowding_state_valid"].sum()),
        "valid_core_rows": int(output["crowding_state_core_valid"].sum()),
        "first_valid_full_date": (
            str(output.loc[output["crowding_state_valid"], "decision_at"].min().date())
            if output["crowding_state_valid"].any()
            else None
        ),
        "first_valid_core_date": (
            str(output.loc[output["crowding_state_core_valid"], "decision_at"].min().date())
            if output["crowding_state_core_valid"].any()
            else None
        ),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
