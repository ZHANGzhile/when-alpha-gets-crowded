"""Validate the historical tradable CSI800 replication weights required by P7."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.backtest import validate_benchmark_replication_weights


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "benchmark" / "csi800_replication_weights.parquet"
SCHEDULE = ROOT / "data" / "processed" / "controller_exposure_schedule.parquet"
OUTPUT = ROOT / "data" / "processed" / "benchmark_replication_weights.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "benchmark_replication_weights.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(
            "P7 requires point-in-time tradable CSI800 replication weights at "
            f"{SOURCE}; constituent membership or equal weights are not substitutes"
        )
    weights = validate_benchmark_replication_weights(pd.read_parquet(SOURCE))
    schedule = pd.read_parquet(SCHEDULE)
    executable = schedule.loc[
        schedule["active_weight_M2"].notna(), "decision_at"
    ].drop_duplicates()
    missing_dates = pd.DatetimeIndex(executable).difference(weights["decision_at"].unique())
    if len(missing_dates):
        raise ValueError(
            f"benchmark replication lacks {len(missing_dates)} controller decisions; "
            f"examples={list(missing_dates[:5])}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "point_in_time_tradable_csi800_replication_weights",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _sha256(SOURCE),
        "decision_dates": int(weights["decision_at"].nunique()),
        "rows": len(weights),
        "first_decision": str(weights["decision_at"].min().date()),
        "last_decision": str(weights["decision_at"].max().date()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(OUTPUT),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
