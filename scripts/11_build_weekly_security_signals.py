"""Build real weekly security-level factor signals for the complete PIT universe."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import (
    add_lagged_matching_characteristics,
    classify_missing_membership_market_rows,
    normalize_baostock_daily,
)
from alpha_crowding.factors import compute_raw_signals


ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "data" / "interim" / "weekly_membership.parquet"
DAILY = ROOT / "data" / "raw" / "daily"
OUTPUT = ROOT / "data" / "interim" / "weekly_security_signals.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "weekly_security_signals.json"


def main() -> int:
    membership = pd.read_parquet(MEMBERSHIP)
    membership["requested_date"] = pd.to_datetime(membership["requested_date"]).dt.normalize()
    decision_dates = pd.DatetimeIndex(membership["requested_date"].unique())
    codes = sorted(membership["code"].dropna().astype(str).unique())
    missing_files = [
        code for code in codes if not (DAILY / f"{code.replace('.', '_')}.parquet").exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"daily histories missing for {len(missing_files)} universe securities; "
            f"examples={missing_files[:10]}"
        )

    parts = []
    for position, code in enumerate(codes, start=1):
        raw = pd.read_parquet(DAILY / f"{code.replace('.', '_')}.parquet")
        normalized = add_lagged_matching_characteristics(normalize_baostock_daily(raw))
        signal_input = normalized[["date", "code", "return_index", "pb_mrq"]].rename(
            columns={"return_index": "close"}
        )
        signals = compute_raw_signals(signal_input)
        audit_fields = normalized[
            [
                "date", "code", "daily_return", "return_index", "amount", "turn",
                "float_shares_estimate", "float_market_cap", "tradestatus", "isST",
                "eligible_for_new_position", "daily_illiquidity", "lagged_liquidity",
                "lagged_float_market_cap",
            ]
        ].copy()
        signals = signals.merge(audit_fields, on=["date", "code"], validate="one_to_one")
        signals["listing_observation_days"] = signals.groupby("code").cumcount() + 1
        weekly = signals[signals["date"].isin(decision_dates)].copy()
        parts.append(weekly)
        if position % 100 == 0:
            print(f"weekly signals {position}/{len(codes)} securities", flush=True)

    security_signals = pd.concat(parts, ignore_index=True)
    panel = membership.rename(columns={"requested_date": "date"}).merge(
        security_signals,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    official_retained_predecessor, unexplained_missing = (
        classify_missing_membership_market_rows(panel)
    )
    if unexplained_missing.any():
        examples = panel.loc[unexplained_missing, ["date", "code"]].head().to_dict("records")
        raise ValueError(f"membership rows lack same-date market records: {examples}")
    panel["market_record_status"] = "OBSERVED"
    panel.loc[official_retained_predecessor, "market_record_status"] = (
        "OFFICIAL_RETAINED_NONTRADING_PREDECESSOR"
    )
    panel.loc[official_retained_predecessor, "eligible_for_new_position"] = False
    panel.loc[official_retained_predecessor, "listing_observation_days"] = 0
    panel = panel.drop(columns="_merge")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("weekly security panel contains duplicate date/code keys")
    panel["base_eligible"] = (
        panel["eligible_for_new_position"].fillna(False).astype(bool)
        & panel["listing_observation_days"].ge(120)
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_weekly_security_factor_inputs",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(panel),
        "decision_dates": int(panel["date"].nunique()),
        "securities": int(panel["code"].nunique()),
        "factor_nonmissing": {
            factor: int(panel[factor].notna().sum())
            for factor in ("momentum", "reversal", "low_volatility", "value")
        },
        "base_eligible_rows": int(panel["base_eligible"].sum()),
        "path": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
