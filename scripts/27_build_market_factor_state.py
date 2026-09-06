"""Build causal Market State and Factor State tables for M0-M4."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.measurement import causal_rank_ic_state, trailing_return_state


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
SIGNALS = ROOT / "data" / "interim" / "weekly_security_signals.parquet"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
FACTOR_RETURNS = ROOT / "data" / "processed" / "factor_returns.parquet"
BENCHMARK = ROOT / "data" / "raw" / "benchmark" / "csi800.parquet"
CONFIG = ROOT / "config" / "measurement.yaml"
MARKET_OUTPUT = ROOT / "data" / "processed" / "market_state.parquet"
FACTOR_OUTPUT = ROOT / "data" / "processed" / "factor_state.parquet"
RANK_IC_OUTPUT = ROOT / "data" / "processed" / "rank_ic_history.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "market_factor_state.json"


def _calendar() -> pd.DatetimeIndex:
    paths = list((ROOT / "data" / "raw" / "membership").glob("trade_calendar_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one production trade calendar; found {len(paths)}")
    frame = pd.read_csv(paths[0])
    return pd.DatetimeIndex(
        pd.to_datetime(frame.loc[pd.to_numeric(frame["is_trading_day"]).eq(1), "calendar_date"])
    ).normalize().sort_values()


def _rank_ic_history(
    memberships: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    rows = []
    for year, year_memberships in memberships.groupby(memberships["date"].dt.year, sort=True):
        paths = [
            path
            for wanted in (int(year), int(year) + 1)
            for path in sorted((DAILY / f"year={wanted}").glob("*.parquet"))
        ]
        market = pd.concat(
            [pd.read_parquet(path, columns=["date", "code", "return_index"]) for path in paths],
            ignore_index=True,
        )
        market["date"] = pd.to_datetime(market["date"]).dt.normalize()
        for decision_at, date_rows in year_memberships.groupby("date", sort=True):
            position = int(calendar.get_indexer([decision_at])[0])
            if position < 0 or position + horizon >= len(calendar):
                continue
            realized_at = calendar[position + horizon]
            current = market[market["date"].eq(decision_at)][["code", "return_index"]].rename(
                columns={"return_index": "return_index_start"}
            )
            future = market[market["date"].eq(realized_at)][["code", "return_index"]].rename(
                columns={"return_index": "return_index_end"}
            )
            realized = current.merge(future, on="code", how="inner", validate="one_to_one")
            realized["future_security_return"] = (
                realized["return_index_end"] / realized["return_index_start"] - 1.0
            )
            for factor, candidates in date_rows.groupby("factor", sort=True):
                merged = candidates[["code", "raw_signal"]].merge(
                    realized[["code", "future_security_return"]], on="code", how="left", validate="one_to_one"
                ).dropna()
                value = (
                    float(merged["raw_signal"].corr(merged["future_security_return"], method="spearman"))
                    if len(merged) >= 20
                    else np.nan
                )
                rows.append(
                    {
                        "signal_at": decision_at,
                        "realized_at": realized_at,
                        "factor": factor,
                        "rank_ic": value,
                        "valid_assets": len(merged),
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["market_factor_state"]
    calendar = _calendar()
    memberships = pd.read_parquet(MEMBERSHIPS)
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    decisions = pd.DatetimeIndex(sorted(memberships["date"].unique()))
    benchmark = pd.read_parquet(BENCHMARK)
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark_series = benchmark.set_index("date")["daily_return"]
    signals = pd.read_parquet(SIGNALS)
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    factor_returns = pd.read_parquet(FACTOR_RETURNS)
    factor_returns["date"] = pd.to_datetime(factor_returns["date"]).dt.normalize()

    market_rows = []
    for decision_at in decisions:
        state = trailing_return_state(
            benchmark_series,
            calendar,
            decision_at=decision_at,
            return_sessions=int(config["trailing_return_days"]),
            volatility_short_sessions=int(config["volatility_short_days"]),
            volatility_long_sessions=int(config["volatility_long_days"]),
            drawdown_sessions=int(config["drawdown_days"]),
        )
        cross = signals[signals["date"].eq(decision_at) & signals["base_eligible"]]
        state.update(
            {
                "market_turnover_median": float(cross["turn"].median()),
                "market_illiquidity_median": float(cross["daily_illiquidity"].median()),
                "market_breadth_20": float(cross["reversal"].lt(0).mean()),
                "cross_sectional_return_dispersion": float(cross["daily_return"].std(ddof=1)),
                "cross_sectional_assets": len(cross),
            }
        )
        market_rows.append(state)
    market_state = pd.DataFrame(market_rows).rename(
        columns={
            "trailing_return": "market_return_20",
            "volatility_short": "market_volatility_20",
            "volatility_long": "market_volatility_60",
            "current_drawdown": "market_drawdown_252",
        }
    )

    rank_ic = _rank_ic_history(
        memberships, calendar, int(config["rank_ic_horizon_days"])
    )
    factor_rows = []
    for factor, factor_memberships in memberships.groupby("factor", sort=True):
        series = factor_returns[factor_returns["factor"].eq(factor)].set_index("date")[
            "long_short_return"
        ]
        ic_state = causal_rank_ic_state(
            rank_ic[rank_ic["factor"].eq(factor)],
            decisions,
            lookback_observations=int(config["rank_ic_lookback_weekly_observations"]),
            minimum_observations=int(config["rank_ic_minimum_weekly_observations"]),
        ).set_index("decision_at")
        for decision_at, date_rows in factor_memberships.groupby("date", sort=True):
            state = trailing_return_state(
                series,
                calendar,
                decision_at=decision_at,
                return_sessions=int(config["trailing_return_days"]),
                volatility_short_sessions=int(config["volatility_short_days"]),
                volatility_long_sessions=int(config["volatility_long_days"]),
                drawdown_sessions=int(config["drawdown_days"]),
            )
            state.update(
                {
                    "factor": factor,
                    "signal_dispersion": float(date_rows["raw_signal"].std(ddof=1)),
                    **ic_state.loc[decision_at].to_dict(),
                }
            )
            factor_rows.append(state)
    factor_state = pd.DataFrame(factor_rows).rename(
        columns={
            "trailing_return": "factor_return_20",
            "volatility_short": "factor_volatility_20",
            "volatility_long": "factor_volatility_60",
            "current_drawdown": "factor_drawdown_252",
        }
    )
    MARKET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    market_state.to_parquet(MARKET_OUTPUT, index=False, compression="zstd")
    factor_state.to_parquet(FACTOR_OUTPUT, index=False, compression="zstd")
    rank_ic.to_parquet(RANK_IC_OUTPUT, index=False, compression="zstd")
    payload = {
        "schema_version": 1,
        "purpose": "production_causal_market_and_factor_states",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_rows": len(market_state),
        "factor_rows": len(factor_state),
        "rank_ic_rows": len(rank_ic),
        "rank_ic_nonmissing": int(rank_ic["rank_ic"].notna().sum()),
        "market_output": str(MARKET_OUTPUT.relative_to(ROOT)),
        "market_sha256": hashlib.sha256(MARKET_OUTPUT.read_bytes()).hexdigest(),
        "factor_output": str(FACTOR_OUTPUT.relative_to(ROOT)),
        "factor_sha256": hashlib.sha256(FACTOR_OUTPUT.read_bytes()).hexdigest(),
        "rank_ic_output": str(RANK_IC_OUTPUT.relative_to(ROOT)),
        "rank_ic_sha256": hashlib.sha256(RANK_IC_OUTPUT.read_bytes()).hexdigest(),
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
