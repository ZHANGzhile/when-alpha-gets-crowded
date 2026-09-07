"""Run the stock-level controller with actual-trade cost accounting."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.backtest import (
    add_expost_mean_exposure_diagnostic,
    assess_benchmark_replication_quality,
    build_controller_stock_targets,
    simulate_stock_level_controller,
    summarize_benchmark_replication_quality,
    summarize_controller_performance,
    summarize_crash_episode_losses,
    validate_benchmark_replication_weights,
    validate_execution_constraints,
)
from alpha_crowding.experiments import require_protocol_freeze
from alpha_crowding.outcomes import first_threshold_breach_at, merge_crash_episodes


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "processed" / "controller_exposure_schedule.parquet"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
BENCHMARK = ROOT / "data" / "processed" / "benchmark_replication_weights.parquet"
BENCHMARK_INDEX = ROOT / "data" / "raw" / "benchmark" / "csi800.parquet"
CONSTRAINTS = ROOT / "data" / "raw" / "execution" / "open_tradeability.parquet"
DAILY_MARKET = ROOT / "data" / "interim" / "daily_market"
CONFIG = ROOT / "config" / "controller.yaml"
FREEZE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"
DAILY_MANIFEST = ROOT / "data" / "raw" / "manifests" / "daily_market.json"
DYNAMIC_OUTCOMES = ROOT / "data" / "processed" / "dynamic_outcomes.parquet"
FACTOR_LEG_RETURNS = ROOT / "data" / "processed" / "factor_leg_returns.parquet"
OUTPUT_DIRECTORY = ROOT / "data" / "results" / "controller"
TARGET_OUTPUT = OUTPUT_DIRECTORY / "stock_targets.parquet"
PATH_OUTPUT = OUTPUT_DIRECTORY / "portfolio_paths.parquet"
LEDGER_OUTPUT = OUTPUT_DIRECTORY / "execution_ledger.parquet"
METRICS_OUTPUT = OUTPUT_DIRECTORY / "performance_metrics.parquet"
REPLICATION_OUTPUT = OUTPUT_DIRECTORY / "benchmark_replication_quality.parquet"
EPISODES_OUTPUT = OUTPUT_DIRECTORY / "active_crash_episodes.parquet"
EPISODE_LOSSES_OUTPUT = OUTPUT_DIRECTORY / "crash_episode_losses.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "controller_backtest.json"
REPLICATION_MANIFEST = (
    ROOT / "data" / "raw" / "manifests" / "benchmark_replication_quality.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_daily_market(first_date: pd.Timestamp) -> pd.DataFrame:
    parts = sorted(DAILY_MARKET.glob("year=*/part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no daily market partitions under {DAILY_MARKET}")
    selected = [
        path
        for path in parts
        if int(path.parent.name.split("=", 1)[1]) >= first_date.year
    ]
    frame = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["date", "code", "overnight_return", "intraday_return"],
            )
            for path in selected
        ],
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.loc[frame["date"].ge(first_date)].reset_index(drop=True)


def main() -> int:
    require_protocol_freeze(FREEZE_MANIFEST, ROOT)
    for path in (
        SCHEDULE,
        MEMBERSHIPS,
        BENCHMARK,
        BENCHMARK_INDEX,
        CONSTRAINTS,
        DAILY_MANIFEST,
        DYNAMIC_OUTCOMES,
        FACTOR_LEG_RETURNS,
    ):
        if not path.exists():
            raise FileNotFoundError(f"required controller input is missing: {path}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    schedule = pd.read_parquet(SCHEDULE)
    primary_columns = [
        f"active_weight_{model}" for model in config["information_models"]
    ]
    schedule = schedule.loc[schedule[primary_columns].notna().all(axis=1)].copy()
    schedule = add_expost_mean_exposure_diagnostic(schedule)
    memberships = pd.read_parquet(MEMBERSHIPS)
    benchmark = validate_benchmark_replication_weights(pd.read_parquet(BENCHMARK))
    constraints = validate_execution_constraints(pd.read_parquet(CONSTRAINTS))
    policy_columns = [
        column
        for column in schedule.columns
        if column.startswith("active_weight_")
    ]
    schedule = schedule.loc[schedule[policy_columns].notna().all(axis=1)].copy()
    targets = build_controller_stock_targets(
        schedule,
        memberships,
        benchmark,
        policy_columns=policy_columns,
    )
    if targets.empty:
        raise ValueError("controller has no executable stock targets after warm-up")
    daily_market = _load_daily_market(targets["effective_at"].min())
    costs = sorted(
        {
            float(config["cost_bps"]["primary"]),
            *map(float, config["cost_bps"]["sensitivity"]),
        }
    )
    paths = []
    ledgers = []
    for cost_bps in costs:
        portfolio, ledger = simulate_stock_level_controller(
            targets,
            daily_market,
            constraints,
            one_way_cost_bps=cost_bps,
        )
        portfolio["cost_bps"] = cost_bps
        ledger["cost_bps"] = cost_bps
        paths.append(portfolio)
        ledgers.append(ledger)
    portfolio_paths = pd.concat(paths, ignore_index=True)
    execution_ledger = pd.concat(ledgers, ignore_index=True)
    benchmark_index = pd.read_parquet(
        BENCHMARK_INDEX, columns=["date", "daily_return"]
    )
    benchmark_index["date"] = pd.to_datetime(benchmark_index["date"]).dt.normalize()
    if benchmark_index["date"].duplicated().any():
        raise ValueError("CSI800 benchmark index contains duplicate dates")
    performance_metrics = summarize_controller_performance(
        portfolio_paths,
        benchmark_index,
        execution_ledger,
    )
    performance_metrics["policy_role"] = "ex_ante_strategy"
    diagnostic_policy = "active_weight_M3_expost_mean"
    diagnostic_mask = performance_metrics["policy"].eq(diagnostic_policy)
    performance_metrics.loc[diagnostic_mask, "policy_role"] = (
        "ex_post_mean_exposure_diagnostic"
    )
    performance_metrics["ex_ante_executable"] = ~diagnostic_mask
    replication_quality = summarize_benchmark_replication_quality(
        portfolio_paths,
        benchmark_index,
        execution_ledger,
    )
    audit = config["benchmark_replication_audit"]
    replication_quality, replication_accepted = assess_benchmark_replication_quality(
        replication_quality,
        **audit,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    replication_quality.to_parquet(
        REPLICATION_OUTPUT, index=False, compression="zstd"
    )
    replication_payload = {
        "schema_version": 1,
        "purpose": "executable_csi800_replication_quality_gate",
        "status": "ACCEPTED" if replication_accepted else "REJECTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": audit,
        "rows": len(replication_quality),
        "output": str(REPLICATION_OUTPUT.relative_to(ROOT)),
        "output_sha256": _sha256(REPLICATION_OUTPUT),
    }
    REPLICATION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REPLICATION_MANIFEST.write_text(
        json.dumps(replication_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not replication_accepted:
        raise ValueError(
            "executable CSI800 replication failed its pre-specified quality gate; "
            f"inspect {REPLICATION_OUTPUT}"
        )
    outcomes = pd.read_parquet(DYNAMIC_OUTCOMES)
    active_events = outcomes.loc[
        outcomes["target_family"].eq("active_long")
        & outcomes["membership_mode"].eq("dynamic")
        & outcomes["horizon_sessions"].eq(20)
        & outcomes["crash_q10"].fillna(False)
    ].copy()
    leg_returns = pd.read_parquet(FACTOR_LEG_RETURNS)
    leg_returns["date"] = pd.to_datetime(leg_returns["date"]).dt.normalize()
    benchmark_series = benchmark_index.set_index("date")["daily_return"]
    calendar = pd.DatetimeIndex(benchmark_index["date"]).sort_values()
    breach_dates = []
    for event in active_events.itertuples(index=False):
        factor_long = leg_returns.loc[
            leg_returns["factor"].eq(event.factor)
            & leg_returns["leg"].eq("LONG")
        ].set_index("date")["daily_return"]
        breach_dates.append(
            first_threshold_breach_at(
                factor_long,
                calendar,
                decision_at=event.decision_at,
                horizon_sessions=int(event.horizon_sessions),
                threshold=float(event.historical_tail_threshold_q10),
                benchmark_returns=benchmark_series,
            )
        )
    active_events["breach_at"] = breach_dates
    active_events["tail_event"] = True
    active_episodes = merge_crash_episodes(active_events)
    episode_losses, episode_summary = summarize_crash_episode_losses(
        portfolio_paths,
        benchmark_index,
        active_episodes,
    )
    performance_metrics = performance_metrics.merge(
        episode_summary,
        on=["factor", "policy", "cost_bps"],
        how="left",
        validate="one_to_one",
    )
    performance_metrics["crash_episode_count"] = performance_metrics[
        "crash_episode_count"
    ].fillna(0).astype(int)
    targets.to_parquet(TARGET_OUTPUT, index=False, compression="zstd")
    portfolio_paths.to_parquet(PATH_OUTPUT, index=False, compression="zstd")
    execution_ledger.to_parquet(LEDGER_OUTPUT, index=False, compression="zstd")
    performance_metrics.to_parquet(METRICS_OUTPUT, index=False, compression="zstd")
    active_episodes.to_parquet(EPISODES_OUTPUT, index=False, compression="zstd")
    episode_losses.to_parquet(
        EPISODE_LOSSES_OUTPUT, index=False, compression="zstd"
    )
    payload = {
        "schema_version": 1,
        "purpose": "stock_level_constrained_controller_backtest",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": _sha256(FREEZE_MANIFEST),
        "source_sha256": {
            "exposure_schedule": _sha256(SCHEDULE),
            "factor_memberships": _sha256(MEMBERSHIPS),
            "benchmark_replication": _sha256(BENCHMARK),
            "benchmark_index": _sha256(BENCHMARK_INDEX),
            "execution_constraints": _sha256(CONSTRAINTS),
            "daily_market_manifest": _sha256(DAILY_MANIFEST),
            "dynamic_outcomes": _sha256(DYNAMIC_OUTCOMES),
            "factor_leg_returns": _sha256(FACTOR_LEG_RETURNS),
            "controller_config": _sha256(CONFIG),
        },
        "policies": policy_columns,
        "policy_roles": {
            policy: (
                "ex_post_mean_exposure_diagnostic"
                if policy == diagnostic_policy
                else "ex_ante_strategy"
            )
            for policy in policy_columns
        },
        "cost_bps": costs,
        "stock_target_rows": len(targets),
        "portfolio_rows": len(portfolio_paths),
        "ledger_rows": len(execution_ledger),
        "metric_rows": len(performance_metrics),
        "benchmark_replication_quality_rows": len(replication_quality),
        "active_crash_episodes": len(active_episodes),
        "crash_episode_loss_rows": len(episode_losses),
        "rejected_ledger_rows": int(execution_ledger["reject_reason"].ne("").sum()),
        "outputs": {
            "stock_targets": str(TARGET_OUTPUT.relative_to(ROOT)),
            "portfolio_paths": str(PATH_OUTPUT.relative_to(ROOT)),
            "execution_ledger": str(LEDGER_OUTPUT.relative_to(ROOT)),
            "performance_metrics": str(METRICS_OUTPUT.relative_to(ROOT)),
            "benchmark_replication_quality": str(
                REPLICATION_OUTPUT.relative_to(ROOT)
            ),
            "active_crash_episodes": str(EPISODES_OUTPUT.relative_to(ROOT)),
            "crash_episode_losses": str(EPISODE_LOSSES_OUTPUT.relative_to(ROOT)),
        },
        "output_sha256": {
            "stock_targets": _sha256(TARGET_OUTPUT),
            "portfolio_paths": _sha256(PATH_OUTPUT),
            "execution_ledger": _sha256(LEDGER_OUTPUT),
            "performance_metrics": _sha256(METRICS_OUTPUT),
            "benchmark_replication_quality": _sha256(REPLICATION_OUTPUT),
            "active_crash_episodes": _sha256(EPISODES_OUTPUT),
            "crash_episode_losses": _sha256(EPISODE_LOSSES_OUTPUT),
        },
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
