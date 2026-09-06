"""Build strictly OOS next-session controller exposure schedules."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.backtest import (
    build_oos_exposure_schedule,
    probability_to_active_weight,
)
from alpha_crowding.experiments import require_protocol_freeze


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data" / "models" / "primary_walk_forward_predictions.parquet"
CONFIG = ROOT / "config" / "controller.yaml"
OUTPUT = ROOT / "data" / "processed" / "controller_exposure_schedule.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "controller_exposure_schedule.json"
PREDICTION_MANIFEST = ROOT / "data" / "models" / "primary_walk_forward.json"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    prediction_manifest = json.loads(PREDICTION_MANIFEST.read_text(encoding="utf-8"))
    if prediction_manifest.get("status") != "COMPLETE":
        raise ValueError("controller requires a COMPLETE primary prediction manifest")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = list(config["information_models"])
    probability_columns = [f"probability_{model}" for model in models]
    predictions = pd.read_parquet(PREDICTIONS)
    active = predictions.loc[
        predictions["target_family"].eq("active_long"),
        ["decision_at", "factor", *probability_columns],
    ]
    tiers = [
        (float(row["upper_percentile"]), float(row["active_weight"]))
        for row in config["primary_policy"]["tiers"]
    ]
    schedule = build_oos_exposure_schedule(
        active,
        _calendar(),
        probability_columns=probability_columns,
        minimum_history_weeks=int(config["primary_policy"]["minimum_history_weeks"]),
        bands=tiers,
    )
    schedule["active_weight_full"] = 1.0
    minimum = float(config["sensitivity_policy"]["minimum_active_weight"])
    for model in models:
        schedule[f"active_weight_{model}_probability_sensitivity"] = schedule[
            f"probability_{model}"
        ].map(lambda value: probability_to_active_weight(value, minimum_weight=minimum))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(OUTPUT, index=False, compression="zstd")
    primary_weight_columns = [f"active_weight_{model}" for model in models]
    payload = {
        "schema_version": 1,
        "purpose": "strictly_historical_oos_controller_exposure_schedule",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": _sha256(FREEZE_MANIFEST),
        "prediction_sha256": _sha256(PREDICTIONS),
        "rows": len(schedule),
        "models": models,
        "minimum_history_weeks": int(config["primary_policy"]["minimum_history_weeks"]),
        "warmup_rows": int(schedule[primary_weight_columns].isna().any(axis=1).sum()),
        "first_executable_date": (
            str(
                schedule.loc[
                    schedule[primary_weight_columns].notna().all(axis=1),
                    "effective_at",
                ]
                .min()
                .date()
            )
            if schedule[primary_weight_columns].notna().all(axis=1).any()
            else None
        ),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(OUTPUT),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
