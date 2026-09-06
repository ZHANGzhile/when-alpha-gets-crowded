"""Download full-market monthly PIT industry snapshots for production joins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import (
    classify_industry_snapshot,
    fetch_industry,
    industry_snapshot_summary,
    session,
)


ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "data" / "interim" / "weekly_membership.parquet"
RAW = ROOT / "data" / "raw" / "industry_monthly"
OUTPUT = ROOT / "data" / "interim" / "monthly_industry.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "monthly_industry.json"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def _one_snapshot(iso_date: str) -> dict[str, object]:
    path = RAW / f"industry_all_{iso_date}.csv"
    if path.exists():
        frame = pd.read_csv(path, dtype=str)
        source = "checkpoint"
    else:
        with session() as bs:
            frame = fetch_industry(bs, date=iso_date)
        RAW.mkdir(parents=True, exist_ok=True)
        _atomic_csv(frame, path)
        source = "network"
    summary = industry_snapshot_summary(frame)
    summary["source"] = source
    summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return summary


def _persist(payload: dict[str, object]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, MANIFEST)


def main(*, max_workers: int) -> int:
    membership = pd.read_parquet(MEMBERSHIP, columns=["requested_date"])
    dates = pd.to_datetime(membership["requested_date"]).drop_duplicates().sort_values()
    month_ends = dates.groupby(dates.dt.to_period("M")).max()
    requested = [date.date().isoformat() for date in month_ends]
    completed = {}
    pending = []
    for iso_date in requested:
        path = RAW / f"industry_all_{iso_date}.csv"
        if path.exists():
            completed[iso_date] = _one_snapshot(iso_date)
        else:
            pending.append(iso_date)
    run: dict[str, object] = {
        "schema_version": 1,
        "purpose": "production_monthly_point_in_time_industry",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_snapshots": len(requested),
        "workers": max_workers,
        "status": "RUNNING",
        "snapshots": list(completed.values()),
        "failures": [],
    }
    failures = []
    print(
        f"industry checkpoints={len(completed)}/{len(requested)}, pending={len(pending)}",
        flush=True,
    )
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one_snapshot, date): date for date in pending}
            for future in as_completed(futures):
                date = futures[future]
                try:
                    completed[date] = future.result()
                except Exception as exc:
                    failures.append((date, type(exc).__name__, str(exc)))
                if (len(completed) + len(failures)) % 5 == 0:
                    print(
                        f"industry completed={len(completed)}/{len(requested)} failures={len(failures)}",
                        flush=True,
                    )
                    run["snapshots"] = [completed[d] for d in requested if d in completed]
                    run["failures"] = failures
                    _persist(run)
        if failures:
            print(f"serial retry for {len(failures)} industry snapshots", flush=True)
            retry_failures = []
            for date, _, _ in failures:
                try:
                    completed[date] = _one_snapshot(date)
                except Exception as exc:
                    retry_failures.append((date, type(exc).__name__, str(exc)))
            failures = retry_failures
        if failures:
            raise RuntimeError(f"{len(failures)} industry snapshots failed after retry")

        frames = []
        for date in requested:
            frame = pd.read_csv(RAW / f"industry_all_{date}.csv", dtype=str)
            classified = classify_industry_snapshot(frame)
            frames.append(classified)
        combined = pd.concat(frames, ignore_index=True)
        combined["requested_date"] = pd.to_datetime(combined["requested_date"]).dt.normalize()
        if combined.duplicated(["requested_date", "code"]).any():
            raise ValueError("monthly industry contains duplicate date/code keys")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(OUTPUT, index=False, compression="zstd")
        run["status"] = "COMPLETE"
        run["snapshots"] = [completed[d] for d in requested]
        run["failures"] = []
        run["rows"] = len(combined)
        run["unique_securities"] = int(combined["code"].nunique())
        run["stable_mapping_coverage"] = float(combined["stable_sector"].notna().mean())
        run["path"] = str(OUTPUT.relative_to(ROOT))
        run["sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    finally:
        run["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _persist(run)
    print(json.dumps(run, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(main(max_workers=args.workers))
