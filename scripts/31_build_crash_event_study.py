"""Build the frozen descriptive crash event study from independent episodes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.experiments import require_protocol_freeze
from alpha_crowding.outcomes import (
    build_event_study_panel,
    first_threshold_breach_at,
    merge_crash_episodes,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "model_feature_panel.parquet"
OUTCOMES = ROOT / "data" / "processed" / "dynamic_outcomes.parquet"
FACTOR_RETURNS = ROOT / "data" / "processed" / "factor_returns.parquet"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"
EPISODES = ROOT / "data" / "results" / "crash_episodes.parquet"
PANEL = ROOT / "data" / "results" / "crash_event_study_panel.parquet"
PATHS = ROOT / "data" / "results" / "crash_event_study_paths.parquet"
MANIFEST = ROOT / "data" / "results" / "crash_event_study.json"

EVENT_VARIABLES = [
    "crowding_state_long",
    "crowding_state_short",
    "crowding_state_ls_mean",
    "stress_trigger_long",
    "stress_trigger_short",
    "stress_trigger_ls_mean",
    "factor_return_20",
    "factor_volatility_20",
    "liquidity_stress_z_long",
    "liquidity_stress_z_short",
    "liquidity_stress_ls_mean",
    "generic_risk_long",
    "generic_risk_short",
    "market_return_20",
    "market_volatility_20",
    "market_volatility_60",
    "market_drawdown_252",
    "market_illiquidity_median",
    "market_breadth_20",
]


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
    calendar = _calendar()
    outcomes = pd.read_parquet(OUTCOMES)
    selected = outcomes.loc[
        outcomes["target_family"].eq("research_ls")
        & outcomes["membership_mode"].eq("dynamic")
        & outcomes["horizon_sessions"].eq(20)
        & outcomes["crash_q10"].fillna(False)
    ].copy()
    if selected.empty:
        raise ValueError("no mature research LS crash events are available")

    returns = pd.read_parquet(FACTOR_RETURNS)
    returns["date"] = pd.to_datetime(returns["date"], errors="raise").dt.normalize()
    breach_dates: list[pd.Timestamp] = []
    for row in selected.itertuples(index=False):
        series = returns.loc[returns["factor"].eq(row.factor)].set_index("date")[
            "long_short_return"
        ]
        breach_dates.append(
            first_threshold_breach_at(
                series,
                calendar,
                decision_at=row.decision_at,
                horizon_sessions=int(row.horizon_sessions),
                threshold=float(row.historical_tail_threshold_q10),
            )
        )
    selected["breach_at"] = breach_dates
    selected["tail_event"] = True
    episodes = merge_crash_episodes(selected, pre_weeks=8, post_weeks=4)

    features = pd.read_parquet(FEATURES)
    missing = set(EVENT_VARIABLES) - set(features.columns)
    if missing:
        raise KeyError(f"missing event-study feature columns: {sorted(missing)}")
    weekly = features[["decision_at", "factor", *EVENT_VARIABLES]].copy()
    weekly["target_family"] = "research_ls"
    weekly["membership_mode"] = "dynamic"
    event_panel = build_event_study_panel(weekly, episodes)
    if event_panel.empty:
        raise ValueError("crash episodes have no overlapping weekly feature observations")

    long = event_panel.melt(
        id_vars=["episode_id", "factor", "event_at", "relative_week"],
        value_vars=EVENT_VARIABLES,
        var_name="variable",
        value_name="value",
    )
    grouped = long.groupby(["relative_week", "variable"], sort=True, dropna=False)
    paths = grouped["value"].agg(["mean", "median", "std", "count"]).reset_index()
    episode_counts = grouped["episode_id"].nunique().rename("episode_count").reset_index()
    paths = paths.merge(
        episode_counts, on=["relative_week", "variable"], validate="one_to_one"
    ).sort_values(["variable", "relative_week"]).reset_index(drop=True)

    EPISODES.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(EPISODES, index=False, compression="zstd")
    event_panel.to_parquet(PANEL, index=False, compression="zstd")
    paths.to_parquet(PATHS, index=False, compression="zstd")
    outputs = [EPISODES, PANEL, PATHS]
    payload = {
        "schema_version": 1,
        "purpose": "frozen_descriptive_research_ls_crash_event_study",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": hashlib.sha256(FREEZE_MANIFEST.read_bytes()).hexdigest(),
        "event_window_weeks": [-8, 4],
        "event_definition": "first cumulative 20-session return breach of alert-time q10 threshold",
        "episode_count": len(episodes),
        "source_crash_rows": int(episodes["source_event_count"].sum()),
        "event_panel_rows": len(event_panel),
        "variables": EVENT_VARIABLES,
        "outputs": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in outputs
        },
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
