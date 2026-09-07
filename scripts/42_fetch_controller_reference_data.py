"""Fetch and build the Tushare reference inputs required by P7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.backtest import (
    build_controller_stock_targets,
    build_drifted_benchmark_weights,
    build_open_tradeability,
)
from alpha_crowding.data import (
    create_tushare_client,
    fetch_csi800_weights,
    fetch_daily_stock_limits,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "processed" / "controller_exposure_schedule.parquet"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
DAILY_MARKET = ROOT / "data" / "interim" / "daily_market"
CONFIG = ROOT / "config" / "controller.yaml"
WEIGHT_CHECKPOINTS = ROOT / "data" / "raw" / "tushare" / "index_weight"
LIMIT_CHECKPOINTS = ROOT / "data" / "raw" / "tushare" / "stock_limit"
BENCHMARK_OUTPUT = (
    ROOT / "data" / "raw" / "benchmark" / "csi800_replication_weights.parquet"
)
CONSTRAINT_OUTPUT = ROOT / "data" / "raw" / "execution" / "open_tradeability.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "controller_reference_data.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


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
                columns=["date", "code", "daily_return", "open", "tradestatus"],
            )
            for path in selected
        ],
        ignore_index=True,
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.loc[frame["date"].ge(first_date)].reset_index(drop=True)


def _weight_months(first_decision: pd.Timestamp, last_decision: pd.Timestamp):
    first_anchor_month = first_decision.to_period("M") - 2
    return pd.period_range(first_anchor_month, last_decision.to_period("M"), freq="M")


def _with_retries(call, *, max_attempts: int, retry_base_seconds: float):
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(retry_base_seconds * attempt)
    raise AssertionError("retry loop exited unexpectedly")


def _load_or_fetch_weights(
    client,
    month: pd.Period,
    *,
    max_attempts: int,
    retry_base_seconds: float,
    request_interval_seconds: float,
) -> pd.DataFrame:
    path = WEIGHT_CHECKPOINTS / f"{month.strftime('%Y%m')}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    start = month.start_time.strftime("%Y%m%d")
    end = month.end_time.strftime("%Y%m%d")
    frame = _with_retries(
        lambda: fetch_csi800_weights(client, start_date=start, end_date=end),
        max_attempts=max_attempts,
        retry_base_seconds=retry_base_seconds,
    )
    _atomic_parquet(frame, path)
    if request_interval_seconds > 0.0:
        time.sleep(request_interval_seconds)
    return frame


def _load_or_fetch_limits(
    client,
    date: pd.Timestamp,
    *,
    max_attempts: int,
    retry_base_seconds: float,
    request_interval_seconds: float,
) -> pd.DataFrame:
    path = LIMIT_CHECKPOINTS / f"{date.strftime('%Y%m%d')}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = _with_retries(
        lambda: fetch_daily_stock_limits(
            client, trade_date=date.strftime("%Y%m%d")
        ),
        max_attempts=max_attempts,
        retry_base_seconds=retry_base_seconds,
    )
    _atomic_parquet(frame, path)
    if request_interval_seconds > 0.0:
        time.sleep(request_interval_seconds)
    return frame


def main(
    *,
    max_attempts: int,
    retry_base_seconds: float,
    request_interval_seconds: float,
) -> int:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if retry_base_seconds < 0.0 or request_interval_seconds < 0.0:
        raise ValueError("retry and request intervals must be non-negative")
    for path in (SCHEDULE, MEMBERSHIPS, CONFIG):
        if not path.exists():
            raise FileNotFoundError(f"required controller input is missing: {path}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    schedule = pd.read_parquet(SCHEDULE)
    schedule["decision_at"] = pd.to_datetime(schedule["decision_at"]).dt.normalize()
    schedule["effective_at"] = pd.to_datetime(schedule["effective_at"]).dt.normalize()
    primary_columns = [
        f"active_weight_{model}" for model in config["information_models"]
    ]
    schedule = schedule.loc[schedule[primary_columns].notna().all(axis=1)].copy()
    policy_columns = [
        column
        for column in schedule.columns
        if column.startswith("active_weight_")
    ]
    schedule = schedule.loc[schedule[policy_columns].notna().all(axis=1)].copy()
    if schedule.empty:
        raise ValueError("controller schedule has no common executable OOS rows")
    client = create_tushare_client()
    first_decision = schedule["decision_at"].min()
    last_decision = schedule["decision_at"].max()
    weight_parts = []
    for month in _weight_months(first_decision, last_decision):
        weight_parts.append(
            _load_or_fetch_weights(
                client,
                month,
                max_attempts=max_attempts,
                retry_base_seconds=retry_base_seconds,
                request_interval_seconds=request_interval_seconds,
            )
        )
    anchors = pd.concat(weight_parts, ignore_index=True)
    if anchors.empty:
        raise ValueError("Tushare returned no CSI800 weight anchors")
    anchors["weight_date"] = pd.to_datetime(anchors["weight_date"]).dt.normalize()
    anchors["available_at"] = anchors["weight_date"]
    first_anchor = anchors.loc[
        anchors["weight_date"].le(first_decision), "weight_date"
    ].max()
    if pd.isna(first_anchor):
        raise ValueError("no CSI800 weight anchor exists before the OOS controller")
    daily_market = _load_daily_market(first_anchor)
    decisions = pd.DatetimeIndex(schedule["decision_at"].drop_duplicates())
    benchmark = build_drifted_benchmark_weights(
        anchors,
        daily_market[["date", "code", "daily_return"]],
        decisions,
    )
    _atomic_parquet(benchmark, BENCHMARK_OUTPUT)

    targets = build_controller_stock_targets(
        schedule,
        pd.read_parquet(MEMBERSHIPS),
        benchmark,
        policy_columns=policy_columns,
    )
    orders = targets[["effective_at", "code"]].drop_duplicates().rename(
        columns={"effective_at": "date"}
    )
    opens = orders.merge(
        daily_market[["date", "code", "open", "tradestatus"]],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    limit_parts = []
    for date in pd.DatetimeIndex(orders["date"].drop_duplicates()).sort_values():
        limit_parts.append(
            _load_or_fetch_limits(
                client,
                date,
                max_attempts=max_attempts,
                retry_base_seconds=retry_base_seconds,
                request_interval_seconds=request_interval_seconds,
            )
        )
    limits = pd.concat(limit_parts, ignore_index=True)
    limits = orders.merge(
        limits,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )
    constraints = build_open_tradeability(opens, limits)
    _atomic_parquet(constraints, CONSTRAINT_OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "tushare_controller_reference_data",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Tushare Pro",
            "index_weight_endpoint": "index_weight",
            "stock_limit_endpoint": "stk_limit",
            "index_code": "000906.SH",
            "weight_availability_assumption": "snapshot_trade_date_close",
            "limit_availability_assumption": "09:00 Asia/Shanghai",
            "execution_time": "09:30 Asia/Shanghai",
        },
        "weight_anchor_dates": int(anchors["weight_date"].nunique()),
        "benchmark_decision_dates": int(benchmark["decision_at"].nunique()),
        "benchmark_rows": len(benchmark),
        "execution_dates": int(constraints["date"].nunique()),
        "execution_rows": len(constraints),
        "blocked_buys": int((~constraints["can_buy"]).sum()),
        "blocked_sells": int((~constraints["can_sell"]).sum()),
        "outputs": {
            str(BENCHMARK_OUTPUT.relative_to(ROOT)): _sha256(BENCHMARK_OUTPUT),
            str(CONSTRAINT_OUTPUT.relative_to(ROOT)): _sha256(CONSTRAINT_OUTPUT),
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--request-interval-seconds", type=float, default=0.15)
    args = parser.parse_args()
    raise SystemExit(
        main(
            max_attempts=args.max_attempts,
            retry_base_seconds=args.retry_base_seconds,
            request_interval_seconds=args.request_interval_seconds,
        )
    )
