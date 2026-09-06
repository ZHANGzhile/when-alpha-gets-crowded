"""Build portfolio-level G inputs and non-return S inputs on real members."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.measurement import (
    historical_zscore_by_group_trading_window,
    portfolio_risk_characteristics,
)


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
CONFIG = ROOT / "config" / "measurement.yaml"
CHECKPOINTS = ROOT / "data" / "processed" / "portfolio_risk_by_date"
OUTPUT = ROOT / "data" / "processed" / "portfolio_risk_features.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "portfolio_risk_features.json"

FEATURES = [
    "turnover_level",
    "illiquidity_level",
    "turnover_shock",
    "turnover_sync",
    "liquidity_shock",
]


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _calendar() -> pd.DatetimeIndex:
    pieces = [
        pd.read_parquet(path, columns=["date"])["date"]
        for path in sorted(DAILY.glob("year=*/*.parquet"))
    ]
    return pd.DatetimeIndex(pd.concat(pieces, ignore_index=True).unique()).normalize().sort_values()


def _read_window(year: int) -> pd.DataFrame:
    paths = []
    for wanted in (year - 2, year - 1, year):
        paths.extend(sorted((DAILY / f"year={wanted}").glob("*.parquet")))
    return pd.concat(
        [pd.read_parquet(path, columns=["date", "code", "turn", "daily_illiquidity"]) for path in paths],
        ignore_index=True,
    )


def _process_date(
    decision_at: pd.Timestamp,
    memberships: pd.DataFrame,
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict[str, object],
) -> dict[str, object]:
    directory = CHECKPOINTS / str(decision_at.date())
    marker = directory / "status.json"
    feature_path = directory / "features.parquet"
    stock_path = directory / "stock_diagnostics.parquet"
    if marker.exists() and feature_path.exists() and stock_path.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["source"] = "checkpoint"
        return payload
    date_rows = memberships[memberships["date"].eq(decision_at)]
    stress = config["stress"]
    portfolio_rows = []
    stock_parts = []
    failures = []
    for (factor, leg), group in date_rows[date_rows["leg"].isin(["LONG", "SHORT"])].groupby(
        ["factor", "leg"], sort=True
    ):
        codes = group["code"].astype(str).tolist()
        try:
            result = portfolio_risk_characteristics(
                daily,
                codes,
                calendar,
                decision_at=decision_at,
                turnover_recent_sessions=int(stress["turnover_shock_recent_days"]),
                turnover_baseline_sessions=int(stress["turnover_baseline_days"]),
                turnover_recent_minimum=int(stress["turnover_recent_minimum_days"]),
                turnover_baseline_minimum=int(stress["turnover_baseline_minimum_days"]),
                turnover_sync_threshold=float(stress["turnover_sync_threshold"]),
                liquidity_recent_sessions=int(stress["illiquidity_recent_days"]),
                liquidity_baseline_sessions=int(stress["illiquidity_baseline_days"]),
                liquidity_recent_minimum=int(stress["illiquidity_recent_minimum_days"]),
                liquidity_baseline_minimum=int(stress["illiquidity_baseline_minimum_days"]),
            )
            portfolio_rows.append({"factor": factor, "leg": leg, **result["portfolio"], "valid": True, "invalid_reason": None})
            stocks = result["stocks"].copy()
            stocks.insert(0, "decision_at", decision_at)
            stocks.insert(1, "factor", factor)
            stocks.insert(2, "leg", leg)
            stock_parts.append(stocks)
        except ValueError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failures.append({"factor": str(factor), "leg": str(leg), "reason": reason})
            portfolio_rows.append(
                {
                    "decision_at": decision_at,
                    "factor": factor,
                    "leg": leg,
                    **{feature: np.nan for feature in FEATURES},
                    "valid": False,
                    "invalid_reason": reason,
                }
            )
    features = pd.DataFrame(portfolio_rows)
    stocks = pd.concat(stock_parts, ignore_index=True) if stock_parts else pd.DataFrame(
        columns=["decision_at", "factor", "leg", "code"]
    )
    _atomic_parquet(features, feature_path)
    _atomic_parquet(stocks, stock_path)
    payload = {
        "decision_at": str(decision_at.date()),
        "source": "computed",
        "feature_rows": len(features),
        "stock_diagnostic_rows": len(stocks),
        "failures": failures,
    }
    _atomic_json(payload, marker)
    return payload


def main(*, start_date: str | None, end_date: str | None, maximum_dates: int | None) -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    memberships = pd.read_parquet(MEMBERSHIPS)
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    calendar = _calendar()
    dates = pd.DatetimeIndex(sorted(memberships["date"].unique()))
    if start_date:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if maximum_dates is not None:
        if maximum_dates < 1:
            raise ValueError("maximum_dates must be positive")
        dates = dates[:maximum_dates]
    summaries = []
    loaded_year = None
    daily = pd.DataFrame()
    for position, decision_at in enumerate(dates, start=1):
        if loaded_year != decision_at.year:
            daily = _read_window(decision_at.year)
            daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
            loaded_year = decision_at.year
        summary = _process_date(decision_at, memberships, daily, calendar, config)
        summaries.append(summary)
        print(
            f"portfolio risk {position}/{len(dates)} date={decision_at.date()} "
            f"source={summary['source']} failures={len(summary['failures'])}",
            flush=True,
        )
    features = pd.concat(
        [pd.read_parquet(CHECKPOINTS / str(date.date()) / "features.parquet") for date in dates],
        ignore_index=True,
    )
    for feature in FEATURES:
        standardized = historical_zscore_by_group_trading_window(
            features,
            calendar,
            value_col=feature,
            group_cols=("factor", "leg"),
            order_col="decision_at",
            lookback_sessions=int(config["historical_window_trading_days"]),
            min_periods=int(config["historical_min_weekly_observations"]),
        )
        features[f"{feature}_historical_z"] = standardized["historical_z"]
        features[f"{feature}_history_n"] = standardized["history_n"]
    _atomic_parquet(features, OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "production_portfolio_generic_risk_and_nonreturn_stress_inputs",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_dates": len(dates),
        "rows": len(features),
        "valid_rows": int(features["valid"].sum()),
        "group_failures": sum(len(summary["failures"]) for summary in summaries),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "checkpoint_directory": str(CHECKPOINTS.relative_to(ROOT)),
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--maximum-dates", type=int)
    args = parser.parse_args()
    raise SystemExit(main(start_date=args.start_date, end_date=args.end_date, maximum_dates=args.maximum_dates))
