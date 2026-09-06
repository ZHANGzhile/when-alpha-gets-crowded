"""Audit full-market historical industry snapshots and taxonomy discontinuities."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import fetch_industry, session
from alpha_crowding.data.industry_taxonomy import industry_snapshot_summary


ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIT = ROOT / "data" / "raw" / "audit" / "industry_taxonomy"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "industry_taxonomy_audit.json"
SNAPSHOT_DATES = (
    "2012-06-29",
    "2012-12-31",
    "2013-06-28",
    "2013-12-31",
    "2014-06-30",
    "2014-12-31",
    "2015-06-30",
    "2015-12-31",
    "2016-06-30",
    "2020-06-30",
    "2024-06-28",
)


def _persist(audit: dict[str, object]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def main(snapshot_dates: tuple[str, ...] = SNAPSHOT_DATES) -> int:
    audit: dict[str, object] = {
        "audit_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "requested_dates": list(snapshot_dates),
        "snapshots": [],
    }
    try:
        RAW_AUDIT.mkdir(parents=True, exist_ok=True)
        with session() as bs:
            audit["baostock_version"] = getattr(bs, "__version__", "unknown")
            for date in snapshot_dates:
                path = RAW_AUDIT / f"industry_all_{date}.csv"
                if path.exists():
                    frame = pd.read_csv(path)
                    source = "checkpoint"
                else:
                    print(f"querying full-market industry snapshot {date}", flush=True)
                    frame = fetch_industry(bs, date=date)
                    frame.to_csv(path, index=False, encoding="utf-8")
                    source = "network"
                summary = industry_snapshot_summary(frame)
                summary["path"] = str(path.relative_to(ROOT))
                summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                summary["source"] = source
                audit["snapshots"].append(summary)
                _persist(audit)
        audit["status"] = "QUERY_COMPLETE_REVIEW_REQUIRED"
    except Exception as exc:
        audit["status"] = "FAILED"
        audit["error_type"] = type(exc).__name__
        audit["error"] = str(exc)
        raise
    finally:
        audit["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _persist(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    requested = tuple(sys.argv[1:]) or SNAPSHOT_DATES
    for value in requested:
        pd.Timestamp(value)
    raise SystemExit(main(requested))
