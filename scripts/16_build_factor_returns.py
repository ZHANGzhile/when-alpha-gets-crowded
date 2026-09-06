"""Build production daily factor-leg and long-short return histories."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import normalize_baostock_daily
from alpha_crowding.factors import combine_long_short_returns, simulate_factor_leg_returns


ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
DAILY = ROOT / "data" / "raw" / "daily"
LEG_OUTPUT = ROOT / "data" / "processed" / "factor_leg_returns.parquet"
FACTOR_OUTPUT = ROOT / "data" / "processed" / "factor_returns.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "factor_returns.json"


def main() -> int:
    memberships = pd.read_parquet(MEMBERSHIPS)
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    calendar_paths = list((ROOT / "data" / "raw" / "membership").glob("trade_calendar_*.csv"))
    if len(calendar_paths) != 1:
        raise ValueError(f"expected one production trade calendar; found {len(calendar_paths)}")
    calendar = pd.read_csv(calendar_paths[0])
    trading = pd.to_numeric(calendar["is_trading_day"], errors="raise").eq(1)
    sessions = pd.DatetimeIndex(
        pd.to_datetime(calendar.loc[trading, "calendar_date"])
    ).normalize().sort_values()
    unexecutable = memberships["date"].ge(sessions.max())
    omitted_decisions = int(memberships.loc[unexecutable, "date"].nunique())
    executable = memberships.loc[~unexecutable].copy()
    leg_codes = sorted(
        executable.loc[executable["leg"].isin(["LONG", "SHORT"]), "code"]
        .dropna()
        .astype(str)
        .unique()
    )
    return_parts = []
    for position, code in enumerate(leg_codes, start=1):
        path = DAILY / f"{code.replace('.', '_')}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"daily history missing for factor holding {code}")
        normalized = normalize_baostock_daily(pd.read_parquet(path))
        return_parts.append(normalized[["date", "code", "daily_return"]])
        if position % 100 == 0:
            print(f"factor return inputs {position}/{len(leg_codes)} securities", flush=True)
    security_returns = pd.concat(return_parts, ignore_index=True)
    leg_returns = simulate_factor_leg_returns(executable, security_returns, sessions)
    factor_returns = combine_long_short_returns(leg_returns)
    LEG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    leg_returns.to_parquet(LEG_OUTPUT, index=False, compression="zstd")
    factor_returns.to_parquet(FACTOR_OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_dynamic_factor_return_histories",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_rule": "target decided at weekly close, active for next session close-to-close return",
        "nominal_long_short_scale": "+1 long / -1 short",
        "omitted_decisions_without_next_session": omitted_decisions,
        "holding_securities": len(leg_codes),
        "leg_rows": len(leg_returns),
        "factor_rows": len(factor_returns),
        "first_return_date": factor_returns["date"].min().date().isoformat(),
        "last_return_date": factor_returns["date"].max().date().isoformat(),
        "rebalance_events": int(leg_returns["rebalance"].sum()),
        "gross_traded_fraction": float(leg_returns["gross_traded_fraction"].sum()),
        "leg_path": str(LEG_OUTPUT.relative_to(ROOT)),
        "leg_sha256": hashlib.sha256(LEG_OUTPUT.read_bytes()).hexdigest(),
        "factor_path": str(FACTOR_OUTPUT.relative_to(ROOT)),
        "factor_sha256": hashlib.sha256(FACTOR_OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
