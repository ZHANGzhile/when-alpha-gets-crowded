"""Use a small stable stock set to bracket BaoStock's taxonomy transition."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import fetch_industry, session


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "manifests" / "industry_transition_probe.json"
CODES = ("sh.600000", "sz.000001", "sh.600519")
DATES = tuple(
    stamp.date().isoformat()
    for stamp in pd.date_range("2012-06-30", "2016-06-30", freq="ME")
)


def main() -> int:
    result: dict[str, object] = {
        "audit_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "probes": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session() as bs:
            for date in DATES:
                labels = []
                for code in CODES:
                    frame = fetch_industry(bs, code=code, date=date)
                    if not frame.empty:
                        labels.extend(frame[["code", "updateDate", "industry"]].to_dict("records"))
                result["probes"].append({"requested_date": date, "records": labels})
                OUTPUT.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        result["status"] = "QUERY_COMPLETE"
    except Exception as exc:
        result["status"] = "FAILED"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        raise
    finally:
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
