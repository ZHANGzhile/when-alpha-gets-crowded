"""Build causal factor states and model features for continuous pseudo strategies."""

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

from alpha_crowding.experiments import (
    PSEUDO_ALGORITHM_VERSION,
    permute_signal_within_industry,
)
from alpha_crowding.measurement import causal_rank_ic_state, trailing_return_state


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
REAL_MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
PSEUDO_RETURNS = ROOT / "data" / "processed" / "pseudo_factor_returns.parquet"
PSEUDO_STATE = ROOT / "data" / "processed" / "pseudo_state_panel.parquet"
MARKET_STATE = ROOT / "data" / "processed" / "market_state.parquet"
CONFIG = ROOT / "config" / "measurement.yaml"
EXPERIMENTS = ROOT / "config" / "experiments.yaml"
RANK_IC_CHECKPOINTS = ROOT / "data" / "processed" / "pseudo_rank_ic_by_date"
RANK_IC_OUTPUT = ROOT / "data" / "processed" / "pseudo_rank_ic_history.parquet"
FACTOR_STATE_OUTPUT = ROOT / "data" / "processed" / "pseudo_factor_state.parquet"
FEATURE_OUTPUT = ROOT / "data" / "processed" / "pseudo_model_feature_panel.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_model_features.json"
STATE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_state.json"
DAILY_MANIFEST = ROOT / "data" / "raw" / "manifests" / "daily_market.json"

MARKET_COLUMNS = [
    "market_return_20",
    "market_volatility_20",
    "market_volatility_60",
    "market_drawdown_252",
    "market_turnover_median",
    "market_illiquidity_median",
    "market_breadth_20",
    "cross_sectional_return_dispersion",
]
FACTOR_COLUMNS = [
    "factor_return_20",
    "factor_volatility_20",
    "factor_volatility_60",
    "factor_drawdown_252",
    "signal_dispersion",
    "rank_ic_mean",
    "rank_ic_volatility",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _read_rank_ic_market(year: int) -> pd.DataFrame:
    paths = [
        path
        for wanted in (year, year + 1)
        for path in sorted((DAILY / f"year={wanted}").glob("*.parquet"))
    ]
    if not paths:
        raise FileNotFoundError(f"daily partitions are missing for {year}/{year + 1}")
    return pd.concat(
        [pd.read_parquet(path, columns=["date", "code", "return_index"]) for path in paths],
        ignore_index=True,
    )


def _process_rank_ic_date(
    decision_at: pd.Timestamp,
    candidates: pd.DataFrame,
    market: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    pseudo_ids: list[str],
    base_seed: int,
    horizon: int,
    source_sha256: str,
    daily_manifest_sha256: str,
) -> dict[str, object]:
    directory = RANK_IC_CHECKPOINTS / str(decision_at.date())
    output_path = directory / "rank_ic.parquet"
    marker = directory / "status.json"
    signature = {
        "source_sha256": source_sha256,
        "strategy_count": len(pseudo_ids),
        "base_seed": base_seed,
        "horizon": horizon,
        "algorithm_version": PSEUDO_ALGORITHM_VERSION,
        "daily_manifest_sha256": daily_manifest_sha256,
    }
    if marker.exists() and output_path.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    position = int(calendar.get_loc(decision_at))
    rows = []
    if position + horizon < len(calendar):
        realized_at = calendar[position + horizon]
        start = market.loc[market["date"].eq(decision_at), ["code", "return_index"]]
        end = market.loc[market["date"].eq(realized_at), ["code", "return_index"]]
        realized = start.merge(end, on="code", suffixes=("_start", "_end"), validate="one_to_one")
        realized["future_security_return"] = (
            realized["return_index_end"] / realized["return_index_start"] - 1.0
        )
        for factor, factor_candidates in candidates.groupby("factor", sort=True):
            base = factor_candidates[["code", "industry", "raw_signal"]].merge(
                realized[["code", "future_security_return"]],
                on="code",
                how="left",
                validate="one_to_one",
            )
            for pseudo_id in pseudo_ids:
                permuted = permute_signal_within_industry(
                    base,
                    base_seed=base_seed,
                    pseudo_strategy_id=pseudo_id,
                    decision_at=decision_at,
                    factor=str(factor),
                ).dropna(subset=["future_security_return"])
                value = (
                    float(
                        permuted["permuted_signal"].corr(
                            permuted["future_security_return"], method="spearman"
                        )
                    )
                    if len(permuted) >= 20
                    else np.nan
                )
                rows.append(
                    {
                        "signal_at": decision_at,
                        "realized_at": realized_at,
                        "original_factor": str(factor),
                        "pseudo_strategy_id": pseudo_id,
                        "rank_ic": value,
                        "valid_assets": len(permuted),
                    }
                )
    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(
            columns=[
                "signal_at",
                "realized_at",
                "original_factor",
                "pseudo_strategy_id",
                "rank_ic",
                "valid_assets",
            ]
        )
    _atomic_parquet(output, output_path)
    payload = {
        **signature,
        "decision_at": str(decision_at.date()),
        "source": "computed",
        "rows": len(output),
        "valid_rows": int(output["rank_ic"].notna().sum()),
        "output_sha256": _sha256(output_path),
    }
    _atomic_json(payload, marker)
    return payload


def _build_factor_state(
    decisions: pd.DataFrame,
    memberships: pd.DataFrame,
    returns: pd.DataFrame,
    rank_ic: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict[str, object],
) -> pd.DataFrame:
    signal_dispersion = memberships.groupby(["date", "factor"])["raw_signal"].std(ddof=1)
    rows = []
    for (pseudo_id, factor), group in decisions.groupby(
        ["pseudo_strategy_id", "original_factor"], sort=True
    ):
        dates = pd.DatetimeIndex(sorted(group["decision_at"].unique()))
        series = returns.loc[
            returns["pseudo_strategy_id"].eq(pseudo_id)
            & returns["original_factor"].eq(factor)
        ].set_index("date")["long_short_return"]
        ic_state = causal_rank_ic_state(
            rank_ic.loc[
                rank_ic["pseudo_strategy_id"].eq(pseudo_id)
                & rank_ic["original_factor"].eq(factor)
            ],
            dates,
            lookback_observations=int(config["rank_ic_lookback_weekly_observations"]),
            minimum_observations=int(config["rank_ic_minimum_weekly_observations"]),
        ).set_index("decision_at")
        for decision_at in dates:
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
                    "original_factor": factor,
                    "pseudo_strategy_id": pseudo_id,
                    "signal_dispersion": float(signal_dispersion.loc[(decision_at, factor)]),
                    **ic_state.loc[decision_at].to_dict(),
                }
            )
            rows.append(state)
    return pd.DataFrame(rows).rename(
        columns={
            "trailing_return": "factor_return_20",
            "volatility_short": "factor_volatility_20",
            "volatility_long": "factor_volatility_60",
            "current_drawdown": "factor_drawdown_252",
        }
    )


def _build_model_features(state: pd.DataFrame, factor_state: pd.DataFrame) -> pd.DataFrame:
    keys = ["decision_at", "original_factor", "pseudo_strategy_id"]
    leg_fields = ["crowding_state", "generic_risk", "stress_trigger"]
    if state.duplicated([*keys, "leg"]).any():
        raise ValueError("pseudo state must be unique by strategy/factor/date/leg")
    wide = state.pivot(index=keys, columns="leg", values=leg_fields)
    wide.columns = [f"{field}_{str(leg).lower()}" for field, leg in wide.columns]
    market = pd.read_parquet(MARKET_STATE)
    features = factor_state[
        keys + FACTOR_COLUMNS
    ].merge(wide.reset_index(), on=keys, validate="one_to_one")
    features = features.merge(
        market[["decision_at", *MARKET_COLUMNS]],
        on="decision_at",
        how="left",
        validate="many_to_one",
    )
    for leg in ("long", "short"):
        features[f"crowding_x_stress_{leg}"] = (
            features[f"crowding_state_{leg}"] * features[f"stress_trigger_{leg}"]
        )
    return features.sort_values(keys).reset_index(drop=True)


def main(*, maximum_dates: int | None, maximum_strategies: int | None) -> int:
    upstream = json.loads(STATE_MANIFEST.read_text(encoding="utf-8"))
    if upstream.get("status") != "COMPLETE":
        raise ValueError("pseudo model features require a COMPLETE pseudo state manifest")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["market_factor_state"]
    experiments = yaml.safe_load(EXPERIMENTS.read_text(encoding="utf-8"))["placebos"]
    state = pd.read_parquet(PSEUDO_STATE)
    state["decision_at"] = pd.to_datetime(state["decision_at"]).dt.normalize()
    decisions = state[
        ["decision_at", "original_factor", "pseudo_strategy_id"]
    ].drop_duplicates()
    dates = pd.DatetimeIndex(sorted(decisions["decision_at"].unique()))
    pseudo_ids = sorted(decisions["pseudo_strategy_id"].unique())
    expected = int(experiments["continuous_strategy_count"])
    if len(pseudo_ids) != expected:
        raise ValueError(f"pseudo model features require {expected} strategies")
    partial = maximum_dates is not None or maximum_strategies is not None
    if maximum_dates is not None:
        if maximum_dates < 1:
            raise ValueError("maximum_dates must be positive")
        dates = dates[:maximum_dates]
    if maximum_strategies is not None:
        if maximum_strategies < 1 or maximum_strategies > expected:
            raise ValueError(f"maximum_strategies must be between 1 and {expected}")
        pseudo_ids = pseudo_ids[:maximum_strategies]
    decisions = decisions.loc[
        decisions["decision_at"].isin(dates)
        & decisions["pseudo_strategy_id"].isin(pseudo_ids)
    ]
    state = state.loc[
        state["decision_at"].isin(dates)
        & state["pseudo_strategy_id"].isin(pseudo_ids)
    ]
    memberships = pd.read_parquet(REAL_MEMBERSHIPS)
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    calendar = _calendar()
    source_sha = _sha256(REAL_MEMBERSHIPS)
    daily_manifest_sha = _sha256(DAILY_MANIFEST)
    base_seed = int(experiments["continuous_strategy_seed"])
    horizon = int(config["rank_ic_horizon_days"])
    summaries = []
    loaded_year = None
    market = pd.DataFrame()
    for number, decision_at in enumerate(dates, start=1):
        if loaded_year != decision_at.year:
            market = _read_rank_ic_market(decision_at.year)
            market["date"] = pd.to_datetime(market["date"]).dt.normalize()
            loaded_year = decision_at.year
        summary = _process_rank_ic_date(
            decision_at,
            memberships.loc[memberships["date"].eq(decision_at)],
            market,
            calendar,
            pseudo_ids=pseudo_ids,
            base_seed=base_seed,
            horizon=horizon,
            source_sha256=source_sha,
            daily_manifest_sha256=daily_manifest_sha,
        )
        summaries.append(summary)
        print(
            f"pseudo RankIC {number}/{len(dates)} date={decision_at.date()} "
            f"source={summary['source']} valid={summary['valid_rows']}/{summary['rows']}",
            flush=True,
        )
    rank_ic_parts = [
        pd.read_parquet(RANK_IC_CHECKPOINTS / str(date.date()) / "rank_ic.parquet")
        for date in dates
    ]
    rank_ic = pd.concat(rank_ic_parts, ignore_index=True)
    _atomic_parquet(rank_ic, RANK_IC_OUTPUT)
    returns = pd.read_parquet(
        PSEUDO_RETURNS,
        columns=["date", "original_factor", "pseudo_strategy_id", "long_short_return"],
    )
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    factor_state = _build_factor_state(
        decisions, memberships, returns, rank_ic, calendar, config
    )
    _atomic_parquet(factor_state, FACTOR_STATE_OUTPUT)
    features = _build_model_features(state, factor_state)
    _atomic_parquet(features, FEATURE_OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "continuous_pseudo_causal_factor_state_and_model_features",
        "status": "PARTIAL" if partial else "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy_count": len(pseudo_ids),
        "configured_strategy_count": expected,
        "decision_dates": len(dates),
        "rank_ic_rows": len(rank_ic),
        "rank_ic_nonmissing": int(rank_ic["rank_ic"].notna().sum()),
        "factor_state_rows": len(factor_state),
        "feature_rows": len(features),
        "signal_dispersion_note": "unchanged by within-industry signal permutation",
        "outputs": {
            str(RANK_IC_OUTPUT.relative_to(ROOT)): _sha256(RANK_IC_OUTPUT),
            str(FACTOR_STATE_OUTPUT.relative_to(ROOT)): _sha256(FACTOR_STATE_OUTPUT),
            str(FEATURE_OUTPUT.relative_to(ROOT)): _sha256(FEATURE_OUTPUT),
        },
        "checkpoint_count": len(summaries),
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-dates", type=int)
    parser.add_argument("--maximum-strategies", type=int)
    args = parser.parse_args()
    raise SystemExit(
        main(
            maximum_dates=args.maximum_dates,
            maximum_strategies=args.maximum_strategies,
        )
    )
