"""Build each continuous pseudo strategy's comparable C/G/S state panel."""

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
    aggregate_stock_risk_groups,
    assemble_continuous_pseudo_state,
    factor_return_shock,
    historical_zscore_by_group_trading_window,
    stock_risk_characteristics,
)


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
PSEUDO_MEMBERSHIPS = ROOT / "data" / "processed" / "pseudo_memberships_by_date"
PSEUDO_LEG_RETURNS = ROOT / "data" / "processed" / "pseudo_leg_returns.parquet"
PSEUDO_STRUCTURAL = ROOT / "data" / "processed" / "pseudo_structural_features.parquet"
PSEUDO_CROWDING = ROOT / "data" / "processed" / "pseudo_crowding_state.parquet"
CONFIG = ROOT / "config" / "measurement.yaml"
EXPERIMENTS = ROOT / "config" / "experiments.yaml"
RISK_CHECKPOINTS = ROOT / "data" / "processed" / "pseudo_risk_by_date"
RETURN_CHECKPOINTS = ROOT / "data" / "processed" / "pseudo_return_stress_by_strategy"
RISK_OUTPUT = ROOT / "data" / "processed" / "pseudo_portfolio_risk_features.parquet"
RETURN_OUTPUT = ROOT / "data" / "processed" / "pseudo_factor_return_stress.parquet"
STATE_OUTPUT = ROOT / "data" / "processed" / "pseudo_state_panel.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "continuous_pseudo_state.json"
CROWDING_MANIFEST = ROOT / "data" / "raw" / "manifests" / "pseudo_crowding_state.json"

KEYS = ["decision_at", "original_factor", "pseudo_strategy_id", "leg"]
RISK_FEATURES = [
    "turnover_level",
    "illiquidity_level",
    "turnover_shock",
    "turnover_sync",
    "liquidity_shock",
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


def _read_risk_window(year: int) -> pd.DataFrame:
    paths = []
    for wanted in (year - 2, year - 1, year):
        paths.extend(sorted((DAILY / f"year={wanted}").glob("*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no daily partitions for {year - 2} through {year}")
    return pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["date", "code", "turn", "daily_illiquidity"],
            )
            for path in paths
        ],
        ignore_index=True,
    )


def _risk_arguments(config: dict[str, object]) -> dict[str, object]:
    stress = config["stress"]
    return {
        "turnover_recent_sessions": int(stress["turnover_shock_recent_days"]),
        "turnover_baseline_sessions": int(stress["turnover_baseline_days"]),
        "turnover_recent_minimum": int(stress["turnover_recent_minimum_days"]),
        "turnover_baseline_minimum": int(stress["turnover_baseline_minimum_days"]),
        "liquidity_recent_sessions": int(stress["illiquidity_recent_days"]),
        "liquidity_baseline_sessions": int(stress["illiquidity_baseline_days"]),
        "liquidity_recent_minimum": int(stress["illiquidity_recent_minimum_days"]),
        "liquidity_baseline_minimum": int(stress["illiquidity_baseline_minimum_days"]),
    }


def _process_risk_date(
    decision_at: pd.Timestamp,
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict[str, object],
) -> dict[str, object]:
    directory = RISK_CHECKPOINTS / str(decision_at.date())
    feature_path = directory / "features.parquet"
    stock_path = directory / "stock_features.parquet"
    marker = directory / "status.json"
    membership_path = PSEUDO_MEMBERSHIPS / str(decision_at.date()) / "memberships.parquet"
    if not membership_path.exists():
        raise FileNotFoundError(f"pseudo memberships missing for {decision_at.date()}")
    signature = {
        "membership_sha256": _sha256(membership_path),
        "measurement_config_sha256": _sha256(CONFIG),
    }
    if marker.exists() and feature_path.exists() and stock_path.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    memberships = pd.read_parquet(membership_path)
    stocks = stock_risk_characteristics(
        daily,
        memberships["code"].astype(str).unique().tolist(),
        calendar,
        decision_at=decision_at,
        **_risk_arguments(config),
    )
    group_cols = ("original_factor", "pseudo_strategy_id", "leg")
    features = aggregate_stock_risk_groups(
        stocks,
        memberships,
        group_cols=group_cols,
        decision_at=decision_at,
        turnover_sync_threshold=float(config["stress"]["turnover_sync_threshold"]),
    )
    features["valid"] = features[RISK_FEATURES].notna().all(axis=1)
    features["invalid_reason"] = np.where(
        features["valid"], None, "one_or_more_stock_aggregate_features_missing"
    )
    _atomic_parquet(stocks, stock_path)
    _atomic_parquet(features, feature_path)
    payload = {
        **signature,
        "decision_at": str(decision_at.date()),
        "source": "computed",
        "stock_rows": len(stocks),
        "portfolio_rows": len(features),
        "valid_rows": int(features["valid"].sum()),
        "stock_sha256": _sha256(stock_path),
        "feature_sha256": _sha256(feature_path),
    }
    _atomic_json(payload, marker)
    return payload


def _standardize_risk(
    features: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict[str, object],
) -> pd.DataFrame:
    output = features.copy()
    for feature in RISK_FEATURES:
        standardized = historical_zscore_by_group_trading_window(
            output,
            calendar,
            value_col=feature,
            group_cols=("original_factor", "pseudo_strategy_id", "leg"),
            order_col="decision_at",
            lookback_sessions=int(config["historical_window_trading_days"]),
            min_periods=int(config["historical_min_weekly_observations"]),
        )
        output[f"{feature}_historical_z"] = standardized["historical_z"]
        output[f"{feature}_history_n"] = standardized["history_n"]
    return output


def _return_stress_for_strategy(
    pseudo_id: str,
    returns: pd.DataFrame,
    decision_rows: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict[str, object],
    *,
    source_sha256: str,
    decision_digest: str,
) -> dict[str, object]:
    directory = RETURN_CHECKPOINTS / pseudo_id
    output_path = directory / "features.parquet"
    marker = directory / "status.json"
    signature = {
        "pseudo_strategy_id": pseudo_id,
        "return_source_sha256": source_sha256,
        "decision_digest": decision_digest,
        "measurement_config_sha256": _sha256(CONFIG),
    }
    if marker.exists() and output_path.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    stress = config["stress"]
    rows = []
    strategy_returns = returns.loc[returns["pseudo_strategy_id"].eq(pseudo_id)]
    strategy_decisions = decision_rows.loc[
        decision_rows["pseudo_strategy_id"].eq(pseudo_id)
    ]
    for (factor, leg), dates in strategy_decisions.groupby(
        ["original_factor", "leg"], sort=True
    ):
        source = strategy_returns.loc[
            strategy_returns["original_factor"].eq(factor)
            & strategy_returns["leg"].eq(leg)
        ].sort_values("date")
        series = source.set_index("date")["daily_return"]
        for decision_at in sorted(dates["decision_at"].unique()):
            try:
                measured = factor_return_shock(
                    series,
                    calendar,
                    decision_at=decision_at,
                    recent_sessions=int(stress["factor_return_shock_days"]),
                    baseline_sessions=int(stress["factor_return_baseline_days"]),
                    baseline_minimum=int(stress["factor_return_baseline_minimum_days"]),
                )
                valid = bool(np.isfinite(measured["factor_return_shock"]))
                reason = None if valid else "missing_return_or_zero_historical_mad"
            except ValueError as exc:
                measured = {
                    "decision_at": decision_at,
                    "factor_return_recent": np.nan,
                    "factor_return_shock": np.nan,
                    "factor_return_history_n": 0,
                }
                valid = False
                reason = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "original_factor": factor,
                    "pseudo_strategy_id": pseudo_id,
                    "leg": leg,
                    **measured,
                    "valid": valid,
                    "invalid_reason": reason,
                }
            )
    output = pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    _atomic_parquet(output, output_path)
    payload = {
        **signature,
        "source": "computed",
        "rows": len(output),
        "valid_rows": int(output["valid"].sum()),
        "output_sha256": _sha256(output_path),
    }
    _atomic_json(payload, marker)
    return payload


def _return_checkpoint_current(
    pseudo_id: str,
    source_sha256: str,
    decision_digest: str,
) -> bool:
    directory = RETURN_CHECKPOINTS / pseudo_id
    output_path = directory / "features.parquet"
    marker = directory / "status.json"
    if not marker.exists() or not output_path.exists():
        return False
    existing = json.loads(marker.read_text(encoding="utf-8"))
    return (
        existing.get("pseudo_strategy_id") == pseudo_id
        and existing.get("return_source_sha256") == source_sha256
        and existing.get("decision_digest") == decision_digest
        and existing.get("measurement_config_sha256") == _sha256(CONFIG)
    )


def _assemble_state(risk: pd.DataFrame, return_stress: pd.DataFrame) -> pd.DataFrame:
    crowding = pd.read_parquet(PSEUDO_CROWDING)
    structural = pd.read_parquet(PSEUDO_STRUCTURAL)
    return assemble_continuous_pseudo_state(crowding, structural, risk, return_stress)


def main(*, maximum_dates: int | None) -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiments = yaml.safe_load(EXPERIMENTS.read_text(encoding="utf-8"))
    upstream = json.loads(CROWDING_MANIFEST.read_text(encoding="utf-8"))
    if maximum_dates is None and upstream.get("status") != "COMPLETE":
        raise ValueError("full pseudo state requires a COMPLETE pseudo Crowding manifest")
    crowding = pd.read_parquet(PSEUDO_CROWDING, columns=KEYS)
    crowding["decision_at"] = pd.to_datetime(crowding["decision_at"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(crowding["decision_at"].unique()))
    if maximum_dates is not None:
        if maximum_dates < 1:
            raise ValueError("maximum_dates must be positive")
        dates = dates[:maximum_dates]
        crowding = crowding.loc[crowding["decision_at"].isin(dates)]
    calendar = _calendar()

    risk_summaries = []
    loaded_year = None
    daily = pd.DataFrame()
    for number, decision_at in enumerate(dates, start=1):
        if loaded_year != decision_at.year:
            daily = _read_risk_window(decision_at.year)
            daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
            loaded_year = decision_at.year
        summary = _process_risk_date(decision_at, daily, calendar, config)
        risk_summaries.append(summary)
        print(
            f"pseudo risk {number}/{len(dates)} date={decision_at.date()} "
            f"source={summary['source']} valid={summary['valid_rows']}/{summary['portfolio_rows']}",
            flush=True,
        )
    risk = pd.concat(
        [
            pd.read_parquet(RISK_CHECKPOINTS / str(date.date()) / "features.parquet")
            for date in dates
        ],
        ignore_index=True,
    )
    risk = _standardize_risk(risk, calendar, config)
    _atomic_parquet(risk, RISK_OUTPUT)

    return_sha = _sha256(PSEUDO_LEG_RETURNS)
    pseudo_ids = sorted(crowding["pseudo_strategy_id"].unique())
    expected_count = int(experiments["placebos"]["continuous_strategy_count"])
    if maximum_dates is None and len(pseudo_ids) != expected_count:
        raise ValueError(
            f"full pseudo state requires {expected_count} strategies; found {len(pseudo_ids)}"
        )
    decision_digests = {
        pseudo_id: hashlib.sha256(
            crowding.loc[crowding["pseudo_strategy_id"].eq(pseudo_id), KEYS]
            .sort_values(KEYS)
            .to_csv(index=False)
            .encode("utf-8")
        ).hexdigest()
        for pseudo_id in pseudo_ids
    }
    needs_returns = any(
        not _return_checkpoint_current(
            pseudo_id, return_sha, decision_digests[pseudo_id]
        )
        for pseudo_id in pseudo_ids
    )
    returns = pd.DataFrame()
    if needs_returns:
        returns = pd.read_parquet(
            PSEUDO_LEG_RETURNS,
            columns=[
                "date",
                "original_factor",
                "pseudo_strategy_id",
                "leg",
                "daily_return",
            ],
        )
        returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    return_summaries = []
    for number, pseudo_id in enumerate(pseudo_ids, start=1):
        summary = _return_stress_for_strategy(
            pseudo_id,
            returns,
            crowding,
            calendar,
            config,
            source_sha256=return_sha,
            decision_digest=decision_digests[pseudo_id],
        )
        return_summaries.append(summary)
        print(
            f"pseudo return stress {number}/{len(pseudo_ids)} strategy={pseudo_id} "
            f"source={summary['source']} valid={summary['valid_rows']}/{summary['rows']}",
            flush=True,
        )
    return_stress = pd.concat(
        [
            pd.read_parquet(RETURN_CHECKPOINTS / pseudo_id / "features.parquet")
            for pseudo_id in pseudo_ids
        ],
        ignore_index=True,
    )
    _atomic_parquet(return_stress, RETURN_OUTPUT)
    state = _assemble_state(risk, return_stress)
    if maximum_dates is not None:
        state = state.loc[state["decision_at"].isin(dates)].reset_index(drop=True)
    _atomic_parquet(state, STATE_OUTPUT)
    status = "PARTIAL" if maximum_dates is not None else "COMPLETE"
    payload = {
        "schema_version": 1,
        "purpose": "continuous_pseudo_comparable_c_g_s_state_panel",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_dates": len(dates),
        "strategy_count": len(pseudo_ids),
        "configured_strategy_count": expected_count,
        "upstream_crowding_manifest_sha256": _sha256(CROWDING_MANIFEST),
        "risk_rows": len(risk),
        "return_stress_rows": len(return_stress),
        "state_rows": len(state),
        "valid_c_rows": int(state["crowding_state_valid"].sum()),
        "valid_g_rows": int(state["generic_risk_valid"].sum()),
        "valid_s_rows": int(state["stress_trigger_valid"].sum()),
        "all_c_g_s_valid_rows": int(
            (
                state["crowding_state_valid"]
                & state["generic_risk_valid"]
                & state["stress_trigger_valid"]
            ).sum()
        ),
        "risk_checkpoint_count": len(risk_summaries),
        "return_checkpoint_count": len(return_summaries),
        "outputs": {
            str(RISK_OUTPUT.relative_to(ROOT)): _sha256(RISK_OUTPUT),
            str(RETURN_OUTPUT.relative_to(ROOT)): _sha256(RETURN_OUTPUT),
            str(STATE_OUTPUT.relative_to(ROOT)): _sha256(STATE_OUTPUT),
        },
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-dates", type=int)
    args = parser.parse_args()
    raise SystemExit(main(maximum_dates=args.maximum_dates))
