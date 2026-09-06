"""Download the production weekly PIT CSI300/CSI500 membership history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.data import (
    fetch_constituents,
    fetch_trade_calendar,
    repair_merger_membership_gaps,
    session,
    validate_constituent_snapshot,
    weekly_last_sessions,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "membership"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "membership_download.json"
CONSOLIDATED = ROOT / "data" / "interim" / "weekly_membership.parquet"
EVENTS_CONFIG = ROOT / "config" / "membership_events.yaml"


def _events() -> list[dict[str, object]]:
    payload = yaml.safe_load(EVENTS_CONFIG.read_text(encoding="utf-8"))
    return list(payload["merger_replacements"])


def _raw_expected_rows(index: str, iso_date: str) -> int:
    expected = {"CSI300": 300, "CSI500": 500}[index]
    date = pd.Timestamp(iso_date).normalize()
    for event in _events():
        if (
            event["index"] == index
            and pd.Timestamp(event["provider_gap_start"]) <= date
            < pd.Timestamp(event["provider_successor_visible"])
        ):
            return expected - 1
    return expected


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def _persist(payload: dict[str, object]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, MANIFEST)


def _snapshot_from_path(path: Path, *, index: str, iso_date: str) -> dict[str, object]:
    frame = pd.read_csv(path, dtype=str)
    summary = validate_constituent_snapshot(
        frame,
        index=index,
        requested_date=iso_date,
        expected_rows=_raw_expected_rows(index, iso_date),
    )
    summary["source"] = "checkpoint"
    summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return summary


def _fetch_week(iso_date: str) -> list[dict[str, object]]:
    """Fetch both indices in one isolated BaoStock login session."""

    summaries = []
    with session() as bs:
        for index in ("CSI300", "CSI500"):
            path = RAW / f"{iso_date}_{index}.csv"
            if path.exists():
                summaries.append(_snapshot_from_path(path, index=index, iso_date=iso_date))
                continue
            frame = fetch_constituents(bs, index=index, date=iso_date)
            _atomic_csv(frame, path)
            summary = validate_constituent_snapshot(
                frame,
                index=index,
                requested_date=iso_date,
                expected_rows=_raw_expected_rows(index, iso_date),
            )
            summary["source"] = "network"
            summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            summaries.append(summary)
    return summaries


def main(start_date: str, end_date: str, *, max_workers: int = 1) -> int:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end precedes start")
    if max_workers < 1:
        raise ValueError("workers must be positive")
    RAW.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "schema_version": 1,
        "purpose": "production_point_in_time_weekly_universe",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "range": [start.date().isoformat(), end.date().isoformat()],
        "status": "RUNNING",
        "completed_snapshots": 0,
        "snapshots": [],
    }
    try:
        calendar_path = RAW / f"trade_calendar_{start.date()}_{end.date()}.csv"
        if calendar_path.exists():
            calendar = pd.read_csv(calendar_path, dtype=str)
        else:
            with session() as bs:
                calendar = fetch_trade_calendar(
                    bs, start_date=start.date().isoformat(), end_date=end.date().isoformat()
                )
            _atomic_csv(calendar, calendar_path)
        week_ends = weekly_last_sessions(calendar)
        dates = [date.date().isoformat() for date in week_ends]
        expected_snapshots = len(dates) * 2
        audit["weekly_decisions"] = len(dates)
        audit["expected_snapshots"] = expected_snapshots
        audit["workers"] = max_workers

        completed_by_date = {}
        pending = []
        for iso_date in dates:
            paths = [RAW / f"{iso_date}_{index}.csv" for index in ("CSI300", "CSI500")]
            if all(path.exists() for path in paths):
                completed_by_date[iso_date] = [
                    _snapshot_from_path(path, index=index, iso_date=iso_date)
                    for path, index in zip(paths, ("CSI300", "CSI500"))
                ]
            else:
                pending.append(iso_date)
        print(
            f"membership checkpoints={2 * len(completed_by_date)}/{expected_snapshots}, "
            f"pending_weeks={len(pending)}, workers={max_workers}",
            flush=True,
        )
        failures = []
        if pending:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_fetch_week, iso_date): iso_date for iso_date in pending}
                for future in as_completed(futures):
                    iso_date = futures[future]
                    try:
                        completed_by_date[iso_date] = future.result()
                    except Exception as exc:
                        failures.append((iso_date, type(exc).__name__, str(exc)))
                    processed = len(completed_by_date) + len(failures)
                    if processed % 10 == 0:
                        audit["snapshots"] = [
                            summary
                            for date in dates if date in completed_by_date
                            for summary in completed_by_date[date]
                        ]
                        audit["completed_snapshots"] = len(audit["snapshots"])
                        audit["failures"] = [
                            {"requested_date": d, "error_type": t, "error": m}
                            for d, t, m in failures
                        ]
                        print(
                            f"membership {audit['completed_snapshots']}/{expected_snapshots} "
                            f"failed_weeks={len(failures)}",
                            flush=True,
                        )
                        _persist(audit)

        if failures:
            print(f"serial retry for {len(failures)} failed weeks", flush=True)
            still_failed = []
            for iso_date, _, _ in failures:
                try:
                    completed_by_date[iso_date] = _fetch_week(iso_date)
                except Exception as exc:
                    still_failed.append((iso_date, type(exc).__name__, str(exc)))
            failures = still_failed
        audit["snapshots"] = [
            summary
            for date in dates if date in completed_by_date
            for summary in completed_by_date[date]
        ]
        audit["completed_snapshots"] = len(audit["snapshots"])
        audit["failures"] = [
            {"requested_date": d, "error_type": t, "error": m}
            for d, t, m in failures
        ]
        if failures:
            raise RuntimeError(f"{len(failures)} membership weeks failed after serial retry")

        paths = sorted(RAW.glob("????-??-??_CSI*.csv"))
        requested_paths = [
            path for path in paths
            if start <= pd.Timestamp(path.name[:10]) <= end
        ]
        combined_raw = pd.concat(
            [pd.read_csv(path, dtype=str) for path in requested_paths], ignore_index=True
        )
        combined, repairs = repair_merger_membership_gaps(combined_raw, _events())
        if combined.duplicated(["requested_date", "index", "code"]).any():
            raise ValueError("consolidated membership contains duplicate keys")
        overlap = combined.groupby(["requested_date", "code"])["index"].nunique()
        if (overlap > 1).any():
            raise ValueError("CSI300 and CSI500 overlap in a weekly snapshot")
        sizes = combined.groupby(["requested_date", "index"]).size()
        for index, expected in (("CSI300", 300), ("CSI500", 500)):
            if not sizes.xs(index, level="index").eq(expected).all():
                raise ValueError(f"repaired {index} snapshots do not all contain {expected} rows")
        CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(CONSOLIDATED, index=False)
        audit["consolidated_path"] = str(CONSOLIDATED.relative_to(ROOT))
        audit["consolidated_rows"] = len(combined)
        audit["unique_securities"] = int(combined["code"].nunique())
        audit["official_repair_rows"] = len(repairs)
        audit["official_repairs"] = repairs
        audit["consolidated_sha256"] = hashlib.sha256(CONSOLIDATED.read_bytes()).hexdigest()
        audit["status"] = "COMPLETE"
    except Exception as exc:
        audit["status"] = "FAILED_RESTART_FROM_CHECKPOINT"
        audit["error_type"] = type(exc).__name__
        audit["error"] = str(exc)
        raise
    finally:
        audit["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _persist(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(main(args.start, args.end, max_workers=args.workers))
