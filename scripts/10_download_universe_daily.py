"""Download raw daily histories for every security ever seen in the PIT universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from alpha_crowding.data import (
    daily_history_acceptance_failures,
    fetch_daily_bars,
    session,
    summarize_daily_history,
)


ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "data" / "interim" / "weekly_membership.parquet"
RAW = ROOT / "data" / "raw" / "daily"
WORKERS = ROOT / "data" / "raw" / "manifests" / "daily"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "daily_download.json"
START_DATE = "2012-01-01"
END_DATE = "2026-08-31"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _download_one(code: str) -> dict[str, object]:
    safe = code.replace(".", "_")
    path = RAW / f"{safe}.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        source = "checkpoint"
    else:
        with session() as bs:
            frame = fetch_daily_bars(
                bs,
                code=code,
                start_date=START_DATE,
                end_date=END_DATE,
                adjustflag="3",
            )
        RAW.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
        source = "network"
    entry = {
        "code": code,
        "source": source,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "quality": summarize_daily_history(frame),
    }
    quality_failures = daily_history_acceptance_failures(entry["quality"])
    if quality_failures:
        raise ValueError(f"{code} failed daily-history acceptance: {quality_failures}")
    _atomic_json(entry, WORKERS / f"{safe}.json")
    return entry


def _persist_run(payload: dict[str, object]) -> None:
    _atomic_json(payload, MANIFEST)


def _record_failure(code: str, exc: Exception) -> dict[str, str]:
    return {"code": code, "error_type": type(exc).__name__, "error": str(exc)}


def _parallel_download(
    codes: list[str],
    *,
    max_workers: int,
    on_progress: Callable[[dict[str, dict[str, object]], list[dict[str, str]]], None]
    | None = None,
) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    completed: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    if max_workers == 1:
        for code in codes:
            try:
                completed[code] = _download_one(code)
            except Exception as exc:
                failures.append(_record_failure(code, exc))
            if on_progress is not None:
                on_progress(completed, failures)
        return completed, failures
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, code): code for code in codes}
        for future in as_completed(futures):
            code = futures[future]
            try:
                completed[code] = future.result()
            except Exception as exc:
                failures.append(_record_failure(code, exc))
            if on_progress is not None:
                on_progress(completed, failures)
    return completed, failures


def main(*, max_workers: int, limit: int | None = None) -> int:
    if max_workers < 1:
        raise ValueError("workers must be positive")
    if not MEMBERSHIP.exists():
        raise FileNotFoundError("weekly membership dataset is not complete")
    membership = pd.read_parquet(MEMBERSHIP, columns=["code"])
    codes = sorted(membership["code"].dropna().astype(str).unique())
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        codes = codes[:limit]
    completed: dict[str, dict[str, object]] = {}
    for code in codes:
        worker = WORKERS / f"{code.replace('.', '_')}.json"
        if worker.exists():
            entry = json.loads(worker.read_text(encoding="utf-8"))
            if not daily_history_acceptance_failures(entry["quality"]):
                completed[code] = entry
    pending = [code for code in codes if code not in completed]
    run: dict[str, object] = {
        "schema_version": 1,
        "purpose": "production_universe_daily_history",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "range": [START_DATE, END_DATE],
        "universe_codes": len(codes),
        "universe_codes_sha256": hashlib.sha256(
            "\n".join(codes).encode("utf-8")
        ).hexdigest(),
        "workers": max_workers,
        "status": "RUNNING",
        "completed": len(completed),
        "failures": [],
    }
    print(
        f"daily histories checkpoints={len(completed)}, pending={len(pending)}, workers={max_workers}",
        flush=True,
    )
    _persist_run(run)
    failures: list[dict[str, str]] = []
    try:
        first_pass_seen = 0

        def record_first_pass(
            current: dict[str, dict[str, object]], current_failures: list[dict[str, str]]
        ) -> None:
            nonlocal first_pass_seen
            processed = len(current) + len(current_failures)
            if processed == len(pending) or processed - first_pass_seen >= 10:
                first_pass_seen = processed
                run["completed"] = len(completed) + len(current)
                run["failures"] = current_failures
                _persist_run(run)
                print(
                    f"daily histories completed={run['completed']}/{len(codes)} "
                    f"first_pass_failures={len(current_failures)}",
                    flush=True,
                )

        new_completed, failures = _parallel_download(
            pending, max_workers=max_workers, on_progress=record_first_pass
        )
        completed.update(new_completed)
        run["completed"] = len(completed)
        run["failures"] = failures
        _persist_run(run)
        print(
            f"daily histories first pass={len(completed)}/{len(codes)} failures={len(failures)}",
            flush=True,
        )

        if failures:
            retry_codes = [failure["code"] for failure in failures]
            print(f"serial retry for {len(retry_codes)} failed securities", flush=True)
            retry_seen = 0

            def record_retry(
                current: dict[str, dict[str, object]], current_failures: list[dict[str, str]]
            ) -> None:
                nonlocal retry_seen
                processed = len(current) + len(current_failures)
                if processed == len(retry_codes) or processed - retry_seen >= 10:
                    retry_seen = processed
                    run["completed"] = len(completed) + len(current)
                    run["failures"] = current_failures
                    _persist_run(run)
                    print(
                        f"daily histories serial_retry={processed}/{len(retry_codes)} "
                        f"remaining_failures={len(current_failures)}",
                        flush=True,
                    )

            retry_completed, failures = _parallel_download(
                retry_codes, max_workers=1, on_progress=record_retry
            )
            completed.update(retry_completed)
            run["completed"] = len(completed)
            run["failures"] = failures
            _persist_run(run)

        run["status"] = "COMPLETE" if not failures else "FAILED_AFTER_SERIAL_RETRY"
        run["total_rows"] = sum(item["quality"].get("rows", 0) for item in completed.values())
        run["zero_row_securities"] = sorted(
            code for code, item in completed.items() if item["quality"].get("rows", 0) == 0
        )
    finally:
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _persist_run(run)
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    raise SystemExit(main(max_workers=args.workers, limit=args.limit))
