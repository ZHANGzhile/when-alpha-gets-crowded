"""Run a bounded BaoStock capability probe and persist auditable raw results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data.baostock_source import (
    fetch_constituents,
    fetch_daily_bars,
    fetch_industry,
    session,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_AUDIT = ROOT / "data" / "raw" / "audit"
MANIFESTS = ROOT / "data" / "raw" / "manifests"
CONSTITUENT_DATES = ["2012-06-29", "2016-06-30", "2020-06-30", "2024-06-28"]
INDUSTRY_PROBES = [
    ("sh.600000", "2012-06-29"),
    ("sh.600000", "2020-06-30"),
    ("sz.000001", "2016-06-30"),
    ("sz.000001", "2024-06-28"),
]


def _write_csv(frame: pd.DataFrame, filename: str) -> dict[str, object]:
    RAW_AUDIT.mkdir(parents=True, exist_ok=True)
    path = RAW_AUDIT / filename
    frame.to_csv(path, index=False, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.relative_to(ROOT)), "rows": len(frame), "sha256": digest}


def main() -> int:
    results: dict[str, object] = {
        "audit_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "queries": [],
        "warnings": [],
    }
    try:
        with session() as bs:
            results["baostock_version"] = getattr(bs, "__version__", "unknown")
            for date in CONSTITUENT_DATES:
                for index in ("CSI300", "CSI500"):
                    frame = fetch_constituents(bs, index=index, date=date)
                    item = _write_csv(frame, f"constituents_{index}_{date}.csv")
                    item.update({"type": "constituents", "index": index, "requested_date": date})
                    results["queries"].append(item)

            for code, date in INDUSTRY_PROBES:
                frame = fetch_industry(bs, code=code, date=date)
                item = _write_csv(frame, f"industry_{code.replace('.', '_')}_{date}.csv")
                item.update({"type": "industry", "code": code, "requested_date": date})
                results["queries"].append(item)

            for adjustflag in ("2", "3"):
                frame = fetch_daily_bars(
                    bs,
                    code="sh.600000",
                    start_date="2017-05-22",
                    end_date="2017-05-31",
                    adjustflag=adjustflag,
                )
                item = _write_csv(frame, f"bars_sh_600000_2017-05_adjust{adjustflag}.csv")
                item.update({"type": "daily_bars", "code": "sh.600000", "adjustflag": adjustflag})
                results["queries"].append(item)

        results["status"] = "QUERY_COMPLETE_REVIEW_REQUIRED"
    except Exception as exc:
        results["status"] = "FAILED"
        results["error_type"] = type(exc).__name__
        results["error"] = str(exc)
        raise
    finally:
        results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        manifest = MANIFESTS / "baostock_capability_audit.json"
        manifest.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

