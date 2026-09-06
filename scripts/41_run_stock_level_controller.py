"""Run the stock-level controller with actual-trade cost accounting."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.backtest import (
    build_controller_stock_targets,
    simulate_stock_level_controller,
    summarize_controller_performance,
    validate_benchmark_replication_weights,
    validate_execution_constraints,
)
from alpha_crowding.experiments import require_protocol_freeze


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
OUTPUT_DIRECTORY = ROOT / "data" / "results" / "controller"
TARGET_OUTPUT = OUTPUT_DIRECTORY / "stock_targets.parquet"
PATH_OUTPUT = OUTPUT_DIRECTORY / "portfolio_paths.parquet"
LEDGER_OUTPUT = OUTPUT_DIRECTORY / "execution_ledger.parquet"
METRICS_OUTPUT = OUTPUT_DIRECTORY / "performance_metrics.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "controller_backtest.json"


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
    ):
        if not path.exists():
            raise FileNotFoundError(f"required controller input is missing: {path}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    schedule = pd.read_parquet(SCHEDULE)
    primary_columns = [
        f"active_weight_{model}" for model in config["information_models"]
    ]
    schedule = schedule.loc[schedule[primary_columns].notna().all(axis=1)].copy()
    memberships = pd.read_parquet(MEMBERSHIPS)
    benchmark = validate_benchmark_replication_weights(pd.read_parquet(BENCHMARK))
    constraints = validate_execution_constraints(pd.read_parquet(CONSTRAINTS))
    policy_columns = [
        column
        for column in schedule.columns
        if column == "active_weight_full" or column.startswith("active_weight_M")
    ]
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
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(TARGET_OUTPUT, index=False, compression="zstd")
    portfolio_paths.to_parquet(PATH_OUTPUT, index=False, compression="zstd")
    execution_ledger.to_parquet(LEDGER_OUTPUT, index=False, compression="zstd")
    performance_metrics.to_parquet(METRICS_OUTPUT, index=False, compression="zstd")
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
            "controller_config": _sha256(CONFIG),
        },
        "policies": policy_columns,
        "cost_bps": costs,
        "stock_target_rows": len(targets),
        "portfolio_rows": len(portfolio_paths),
        "ledger_rows": len(execution_ledger),
        "metric_rows": len(performance_metrics),
        "rejected_ledger_rows": int(execution_ledger["reject_reason"].ne("").sum()),
        "outputs": {
            "stock_targets": str(TARGET_OUTPUT.relative_to(ROOT)),
            "portfolio_paths": str(PATH_OUTPUT.relative_to(ROOT)),
            "execution_ledger": str(LEDGER_OUTPUT.relative_to(ROOT)),
            "performance_metrics": str(METRICS_OUTPUT.relative_to(ROOT)),
        },
        "output_sha256": {
            "stock_targets": _sha256(TARGET_OUTPUT),
            "portfolio_paths": _sha256(PATH_OUTPUT),
            "execution_ledger": _sha256(LEDGER_OUTPUT),
            "performance_metrics": _sha256(METRICS_OUTPUT),
        },
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
