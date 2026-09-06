"""Download and audit real full histories for a fixed 20-security acceptance set."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import (
    compare_adjustments,
    fetch_daily_bars,
    session,
    summarize_daily_history,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "audit" / "stock20"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "stock20_audit.json"
WORKER_MANIFESTS = ROOT / "data" / "raw" / "manifests" / "stock20"
START_DATE = "2012-01-01"
END_DATE = "2026-08-31"
SECURITIES = (
    "sh.600000", "sh.600519", "sh.601398", "sh.600036", "sh.600030",
    "sh.601318", "sh.601857", "sh.600276", "sh.600111", "sh.600005",
    "sz.000001", "sz.000858", "sz.000002", "sz.000333", "sz.002594",
    "sz.300750", "sz.000024", "sh.600485", "sz.000820", "sh.688981",
)
ADJUSTMENT_SECURITIES = {
    "sh.600000", "sh.600519", "sh.601398", "sh.600036", "sh.600005",
    "sz.000001", "sz.000024",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def _persist(payload: dict[str, object]) -> None:
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, MANIFEST)


def _worker_manifest_path(code: str) -> Path:
    return WORKER_MANIFESTS / f"{code.replace('.', '_')}.json"


def _audit_one(code: str) -> dict[str, object]:
    """Run one independent, restartable security audit in a child process."""

    safe_code = code.replace(".", "_")
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    hashes: dict[str, str] = {}
    requested = [("3", "raw")]
    if code in ADJUSTMENT_SECURITIES:
        requested.append(("2", "forward_adjusted"))
    with session() as bs:
        for adjustflag, label in requested:
            path = RAW / f"{safe_code}_{label}_{START_DATE}_{END_DATE}.csv"
            if path.exists():
                frame = pd.read_csv(path, dtype=str)
                source = "checkpoint"
            else:
                frame = fetch_daily_bars(
                    bs,
                    code=code,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjustflag=adjustflag,
                )
                _atomic_csv(frame, path)
                source = "network"
            frames[label] = frame
            sources[label] = source
            hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {
        "code": code,
        "source": sources,
        "sha256": hashes,
        "raw_quality": summarize_daily_history(frames["raw"]),
        "adjustment_check": (
            compare_adjustments(frames["raw"], frames["forward_adjusted"])
            if "forward_adjusted" in frames
            else {"status": "NOT_REQUESTED", "reason": "fixed_seven_security_adjustment_sample"}
        ),
    }
    worker_path = _worker_manifest_path(code)
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = worker_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, worker_path)
    return entry


def main(*, max_workers: int = 4) -> int:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    RAW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    WORKER_MANIFESTS.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {
        "audit_version": 1,
        "purpose": "production_data_acceptance_not_model_selection",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_range": [START_DATE, END_DATE],
        "status": "RUNNING",
        "securities": [],
    }
    failures = []
    try:
        completed = {}
        for code in SECURITIES:
            worker_path = _worker_manifest_path(code)
            if worker_path.exists():
                completed[code] = json.loads(worker_path.read_text(encoding="utf-8"))
        pending = [code for code in SECURITIES if code not in completed]
        print(
            f"checkpoints={len(completed)}, pending={len(pending)}, workers={max_workers}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_audit_one, code): code for code in pending}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    completed[code] = future.result()
                    print(f"completed {len(completed):02d}/{len(SECURITIES)} {code}", flush=True)
                except Exception as exc:
                    failures.append(
                        {"code": code, "error_type": type(exc).__name__, "error": str(exc)}
                    )
                    print(f"failed {code}: {type(exc).__name__}: {exc}", flush=True)
                audit["securities"] = [completed[code] for code in SECURITIES if code in completed]
                audit["failures"] = failures
                _persist(audit)
        audit["status"] = (
            "QUERY_COMPLETE_REVIEW_REQUIRED" if not failures else "PARTIAL_FAILED_REVIEW_REQUIRED"
        )
    finally:
        audit["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _persist(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    raise SystemExit(main(max_workers=arguments.workers))
