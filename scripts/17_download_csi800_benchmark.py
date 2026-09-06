"""Download and validate the CSI800 price-index series used by Active targets."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import fetch_index_bars, session


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "benchmark" / "csi800.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "csi800_benchmark.json"
CODE = "sh.000906"
START_DATE = "2012-01-01"
END_DATE = "2026-08-31"


def main() -> int:
    with session() as bs:
        frame = fetch_index_bars(
            bs, code=CODE, start_date=START_DATE, end_date=END_DATE
        )
    if frame.empty:
        raise ValueError("CSI800 benchmark query returned no data")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("CSI800 benchmark dates are duplicated or unordered")
    if not frame["code"].eq(CODE).all():
        raise ValueError("CSI800 benchmark response contains another code")
    numeric = frame[["open", "high", "low", "close", "preclose", "pctChg"]].apply(
        pd.to_numeric, errors="coerce"
    )
    bad_ohlc = (
        (numeric["low"] > numeric["high"])
        | (numeric["open"] < numeric["low"])
        | (numeric["open"] > numeric["high"])
        | (numeric["close"] < numeric["low"])
        | (numeric["close"] > numeric["high"])
    )
    if bad_ohlc.fillna(False).any():
        raise ValueError("CSI800 benchmark contains invalid OHLC ordering")
    frame["daily_return"] = pd.to_numeric(frame["pctChg"], errors="coerce") / 100.0
    missing_returns = int(frame["daily_return"].isna().sum())
    if missing_returns > 1:
        raise ValueError(f"CSI800 benchmark has {missing_returns} missing returns")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "production_csi800_benchmark_for_active_targets",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": CODE,
        "range": [START_DATE, END_DATE],
        "rows": len(frame),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "missing_returns": missing_returns,
        "path": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "status": "COMPLETE",
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
