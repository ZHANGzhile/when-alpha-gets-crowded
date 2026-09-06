"""Build past-only factor-return shock features at weekly decision sessions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.measurement import factor_return_shock


ROOT = Path(__file__).resolve().parents[1]
RETURNS = ROOT / "data" / "processed" / "factor_leg_returns.parquet"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
CONFIG = ROOT / "config" / "measurement.yaml"
OUTPUT = ROOT / "data" / "processed" / "factor_return_stress.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "factor_return_stress.json"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    stress = config["stress"]
    returns = pd.read_parquet(RETURNS)
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    memberships = pd.read_parquet(MEMBERSHIPS, columns=["date", "factor", "leg"])
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    decisions = memberships[memberships["leg"].isin(["LONG", "SHORT"])].drop_duplicates()
    calendar_paths = list((ROOT / "data" / "raw" / "membership").glob("trade_calendar_*.csv"))
    if len(calendar_paths) != 1:
        raise ValueError(f"expected one production trade calendar; found {len(calendar_paths)}")
    calendar_frame = pd.read_csv(calendar_paths[0])
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            calendar_frame.loc[
                pd.to_numeric(calendar_frame["is_trading_day"], errors="raise").eq(1),
                "calendar_date",
            ]
        )
    ).normalize().sort_values()
    rows = []
    for (factor, leg), group in decisions.groupby(["factor", "leg"], sort=True):
        series_frame = returns[
            returns["factor"].eq(factor) & returns["leg"].eq(leg)
        ].sort_values("date")
        series = series_frame.set_index("date")["daily_return"]
        for decision_at in sorted(group["date"].unique()):
            try:
                measured = factor_return_shock(
                    series,
                    calendar,
                    decision_at=decision_at,
                    recent_sessions=int(stress["factor_return_shock_days"]),
                    baseline_sessions=int(stress["factor_return_baseline_days"]),
                    baseline_minimum=int(stress["factor_return_baseline_minimum_days"]),
                )
                rows.append({"factor": factor, "leg": leg, **measured, "valid": bool(np.isfinite(measured["factor_return_shock"])), "invalid_reason": None if np.isfinite(measured["factor_return_shock"]) else "missing_return_or_zero_historical_mad"})
            except ValueError as exc:
                rows.append(
                    {
                        "decision_at": decision_at,
                        "factor": factor,
                        "leg": leg,
                        "factor_return_recent": np.nan,
                        "factor_return_shock": np.nan,
                        "factor_return_history_n": 0,
                        "valid": False,
                        "invalid_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    output = pd.DataFrame(rows).sort_values(["decision_at", "factor", "leg"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_factor_return_stress",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(output),
        "valid_rows": int(output["valid"].sum()),
        "first_valid_date": str(output.loc[output["valid"], "decision_at"].min().date()) if output["valid"].any() else None,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
