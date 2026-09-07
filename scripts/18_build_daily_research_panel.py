"""Build a partitioned, normalized daily panel for production measurements.

The raw archive remains immutable.  This stage converts the per-security BaoStock
files into year partitions so 60/252-session feature windows can be scanned
without reopening every security file for every factor-date.  Batch markers make
the expensive conversion resumable after interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from alpha_crowding.data import (
    add_lagged_matching_characteristics,
    normalize_baostock_daily,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "daily"
BUILDING = ROOT / "data" / "interim" / "daily_market.building"
OUTPUT = ROOT / "data" / "interim" / "daily_market"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "daily_market.json"
SOURCE_MANIFEST = ROOT / "data" / "raw" / "manifests" / "daily_download.json"

OUTPUT_COLUMNS = [
    "date",
    "code",
    "open",
    "close",
    "preclose",
    "daily_return",
    "overnight_return",
    "intraday_return",
    "return_index",
    "volume",
    "amount",
    "turn",
    "tradestatus",
    "isST",
    "float_market_cap",
    "eligible_for_new_position",
    "daily_illiquidity",
    "lagged_liquidity",
    "lagged_float_market_cap",
]


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch_marker(batch_number: int) -> Path:
    return BUILDING / "_batches" / f"batch_{batch_number:04d}.json"


def _process_batch(paths: list[Path], batch_number: int) -> dict[str, object]:
    marker = _batch_marker(batch_number)
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        expected = [BUILDING / relative for relative in payload["parts"]]
        if all(path.exists() for path in expected):
            payload["source"] = "checkpoint"
            return payload

    normalized_parts: list[pd.DataFrame] = []
    for path in paths:
        raw = pd.read_parquet(path)
        normalized = add_lagged_matching_characteristics(
            normalize_baostock_daily(raw)
        )
        normalized_parts.append(normalized.loc[:, OUTPUT_COLUMNS])
    batch = pd.concat(normalized_parts, ignore_index=True)
    batch["year"] = batch["date"].dt.year.astype("int16")
    written: list[str] = []
    rows_by_year: dict[str, int] = {}
    for year, year_frame in batch.groupby("year", sort=True):
        directory = BUILDING / f"year={int(year)}"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"part_{batch_number:04d}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        year_frame.drop(columns="year").sort_values(
            ["date", "code"], kind="mergesort"
        ).to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, destination)
        written.append(str(destination.relative_to(BUILDING)))
        rows_by_year[str(int(year))] = int(len(year_frame))
    payload = {
        "batch": batch_number,
        "source": "raw",
        "securities": len(paths),
        "rows": int(len(batch)),
        "minimum_date": str(batch["date"].min().date()),
        "maximum_date": str(batch["date"].max().date()),
        "rows_by_year": rows_by_year,
        "parts": written,
    }
    _atomic_json(payload, marker)
    return payload


def main(*, batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("status") != "COMPLETE":
        raise RuntimeError("complete daily download is required")
    paths = sorted(RAW.glob("*.parquet"))
    if len(paths) != int(source["universe_codes"]):
        raise RuntimeError(
            f"raw file count {len(paths)} does not match universe "
            f"{source['universe_codes']}"
        )
    if OUTPUT.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("source_daily_manifest_sha256") == _sha256(SOURCE_MANIFEST)
        ):
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return 0
        raise FileExistsError(
            "daily_market exists but is not validated against the current source; "
            "move it aside before rebuilding"
        )
    BUILDING.mkdir(parents=True, exist_ok=True)
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
    summaries = []
    for batch_number, batch_paths in enumerate(batches):
        summary = _process_batch(batch_paths, batch_number)
        summaries.append(summary)
        print(
            f"daily research panel batch={batch_number + 1}/{len(batches)} "
            f"rows={summary['rows']} source={summary['source']}",
            flush=True,
        )
    total_rows = sum(int(item["rows"]) for item in summaries)
    total_securities = sum(int(item["securities"]) for item in summaries)
    if total_securities != len(paths):
        raise AssertionError("batch security count does not match input")
    os.replace(BUILDING, OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "partitioned_normalized_daily_research_panel",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_daily_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "securities": total_securities,
        "rows": total_rows,
        "years": sorted(
            int(path.name.split("=", 1)[1])
            for path in OUTPUT.glob("year=*")
            if path.is_dir()
        ),
        "parts": sum(len(item["parts"]) for item in summaries),
        "columns": OUTPUT_COLUMNS,
        "path": str(OUTPUT.relative_to(ROOT)),
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    raise SystemExit(main(batch_size=args.batch_size))
