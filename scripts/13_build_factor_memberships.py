"""Build production weekly factor candidates and exact long/short legs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import join_industry_asof
from alpha_crowding.factors import build_factor_memberships


ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ROOT / "data" / "interim" / "weekly_security_signals.parquet"
INDUSTRY = ROOT / "data" / "interim" / "monthly_industry.parquet"
OUTPUT = ROOT / "data" / "processed" / "factor_memberships.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "factor_memberships.json"


def main(*, include_value: bool) -> int:
    weekly = pd.read_parquet(SIGNALS)
    industry = pd.read_parquet(INDUSTRY)
    joined = join_industry_asof(weekly, industry, maximum_age_days=45)
    enabled = ["MOM", "REV", "LOWVOL"]
    if include_value:
        enabled.append("VALUE")
    memberships = build_factor_memberships(joined, enabled_factors=enabled)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    memberships.to_parquet(OUTPUT, index=False, compression="zstd")
    leg_rows = memberships[memberships["leg"].isin(["LONG", "SHORT"])]
    payload = {
        "schema_version": 1,
        "purpose": "production_point_in_time_factor_memberships",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "enabled_factors": enabled,
        "value_confirmation_eligible": False,
        "rows": len(memberships),
        "leg_rows": len(leg_rows),
        "decision_dates": int(memberships["date"].nunique()),
        "candidate_rows_by_factor": memberships.groupby("factor").size().to_dict(),
        "leg_rows_by_factor_and_leg": {
            "|".join(key): int(value)
            for key, value in leg_rows.groupby(["factor", "leg"]).size().items()
        },
        "stale_or_missing_industry_rows": int((~joined["industry_is_fresh"]).sum()),
        "path": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-value", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(include_value=args.include_value))
