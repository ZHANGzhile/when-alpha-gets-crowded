"""Build each continuous pseudo strategy's leave-one-out structural Crowding state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from alpha_crowding.measurement import (
    compute_crowding_state,
    historical_zscore_by_group_trading_window,
    leave_one_out_placebo_adjustment,
    measure_continuous_pseudo_convergence,
    measure_continuous_pseudo_structure,
)


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
REAL_MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
PSEUDO_MEMBERSHIPS = ROOT / "data" / "processed" / "pseudo_memberships_by_date"
CONFIG = ROOT / "config" / "measurement.yaml"
EXPERIMENTS = ROOT / "config" / "experiments.yaml"
CHECKPOINTS = ROOT / "data" / "processed" / "pseudo_structure_by_date"
FEATURE_OUTPUT = ROOT / "data" / "processed" / "pseudo_structural_features.parquet"
STATE_OUTPUT = ROOT / "data" / "processed" / "pseudo_crowding_state.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "pseudo_crowding_state.json"

FEATURE_COLUMNS = {
    "residual_sync": "excess_sync_historical_z",
    "eigen_concentration": "excess_eigen_historical_z",
    "strategy_convergence": "excess_overlap_historical_z",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _calendar() -> pd.DatetimeIndex:
    pieces = [
        pd.read_parquet(path, columns=["date"])["date"]
        for path in sorted(DAILY.glob("year=*/*.parquet"))
    ]
    if not pieces:
        raise FileNotFoundError("daily research panel partitions are missing")
    return pd.DatetimeIndex(pd.concat(pieces, ignore_index=True).unique()).normalize().sort_values()


def _read_year_window(year: int) -> pd.DataFrame:
    paths = []
    for wanted in (year - 1, year):
        paths.extend(sorted((DAILY / f"year={wanted}").glob("*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no daily partitions for {year - 1}/{year}")
    return pd.concat(
        [pd.read_parquet(path, columns=["date", "code", "daily_return"]) for path in paths],
        ignore_index=True,
    )


def _process_date(
    date: pd.Timestamp,
    candidates: pd.DataFrame,
    daily_window: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    minimum_comparators: int,
) -> dict[str, object]:
    directory = CHECKPOINTS / str(date.date())
    output = directory / "features.parquet"
    marker = directory / "status.json"
    membership_path = PSEUDO_MEMBERSHIPS / str(date.date()) / "memberships.parquet"
    if not membership_path.exists():
        raise FileNotFoundError(f"pseudo memberships missing for {date.date()}")
    membership_sha = _sha256(membership_path)
    signature = {
        "membership_sha256": membership_sha,
        "minimum_comparators": minimum_comparators,
    }
    if marker.exists() and output.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in signature.items()):
            existing["source"] = "checkpoint"
            return existing

    position = int(calendar.get_loc(date))
    window_dates = calendar[max(0, position - 59) : position + 1]
    market = daily_window.loc[daily_window["date"].isin(window_dates)]
    memberships = pd.read_parquet(membership_path)
    structural = measure_continuous_pseudo_structure(
        market,
        candidates,
        memberships,
        decision_at=date,
        minimum_days=40,
        minimum_assets=10,
    )
    convergence = measure_continuous_pseudo_convergence(
        memberships, decision_at=date
    )
    raw = pd.concat([structural, convergence], ignore_index=True, sort=False)
    adjusted = leave_one_out_placebo_adjustment(
        raw,
        minimum_comparators=minimum_comparators,
    )
    _atomic_parquet(adjusted, output)
    payload = {
        **signature,
        "decision_at": str(date.date()),
        "source": "computed",
        "window_start": str(window_dates.min().date()),
        "window_end": str(window_dates.max().date()),
        "feature_rows": len(adjusted),
        "valid_rows": int(adjusted["placebo_valid"].sum()),
        "output_sha256": _sha256(output),
    }
    _atomic_json(payload, marker)
    return payload


def main(*, maximum_dates: int | None) -> int:
    measurement = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    experiments = yaml.safe_load(EXPERIMENTS.read_text(encoding="utf-8"))
    minimum_comparators = int(
        experiments["placebos"]["continuous_minimum_comparators"]
    )
    expected_strategies = int(experiments["placebos"]["continuous_strategy_count"])
    if minimum_comparators >= expected_strategies:
        raise ValueError("continuous_minimum_comparators must be below strategy count")
    real = pd.read_parquet(REAL_MEMBERSHIPS)
    real["date"] = pd.to_datetime(real["date"], errors="raise").dt.normalize()
    dates = pd.DatetimeIndex(sorted(real["date"].unique()))
    if maximum_dates is not None:
        if maximum_dates < 1:
            raise ValueError("maximum_dates must be positive")
        dates = dates[:maximum_dates]
    calendar = _calendar()
    absent = dates.difference(calendar)
    if len(absent):
        raise ValueError(f"decision dates absent from trading calendar: {list(absent[:5])}")

    summaries = []
    loaded_year = None
    daily_window = pd.DataFrame()
    for number, date in enumerate(dates, start=1):
        if loaded_year != date.year:
            daily_window = _read_year_window(date.year)
            daily_window["date"] = pd.to_datetime(
                daily_window["date"], errors="raise"
            ).dt.normalize()
            loaded_year = date.year
        summary = _process_date(
            date,
            real.loc[real["date"].eq(date)],
            daily_window,
            calendar,
            minimum_comparators=minimum_comparators,
        )
        summaries.append(summary)
        print(
            f"pseudo structure {number}/{len(dates)} date={date.date()} "
            f"source={summary['source']} valid={summary['valid_rows']}/{summary['feature_rows']}",
            flush=True,
        )

    features = pd.concat(
        [
            pd.read_parquet(CHECKPOINTS / str(date.date()) / "features.parquet")
            for date in dates
        ],
        ignore_index=True,
    )
    standardized = historical_zscore_by_group_trading_window(
        features,
        calendar,
        value_col="excess",
        group_cols=("original_factor", "pseudo_strategy_id", "leg", "feature"),
        order_col="decision_at",
        lookback_sessions=int(measurement["historical_window_trading_days"]),
        min_periods=int(measurement["historical_min_weekly_observations"]),
    )
    features["excess_historical_z"] = standardized["historical_z"]
    features["excess_history_n"] = standardized["history_n"]
    features["excess_history_window_start"] = standardized["history_window_start"]
    features["excess_history_window_end"] = standardized["history_window_end"]
    _atomic_parquet(features, FEATURE_OUTPUT)

    keys = ["decision_at", "original_factor", "pseudo_strategy_id", "leg"]
    wide = features.pivot(
        index=keys,
        columns="feature",
        values="excess_historical_z",
    ).rename(columns=FEATURE_COLUMNS)
    missing = set(FEATURE_COLUMNS.values()) - set(wide.columns)
    if missing:
        raise ValueError(f"missing pseudo Crowding components: {sorted(missing)}")
    full = compute_crowding_state(wide)
    core = compute_crowding_state(
        wide,
        components=("excess_sync_historical_z", "excess_eigen_historical_z"),
    ).rename(
        columns={
            "crowding_state": "crowding_state_core",
            "crowding_state_valid": "crowding_state_core_valid",
            "crowding_state_component_count": "crowding_state_core_component_count",
        }
    )
    state = pd.concat([wide, full, core], axis=1).reset_index()
    _atomic_parquet(state, STATE_OUTPUT)
    status = "PARTIAL" if maximum_dates is not None else "COMPLETE"
    payload = {
        "schema_version": 1,
        "purpose": "continuous_pseudo_leave_one_out_structural_crowding_state",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_dates": len(dates),
        "strategy_count": expected_strategies,
        "minimum_comparators": minimum_comparators,
        "feature_rows": len(features),
        "valid_cross_sectional_rows": int(features["placebo_valid"].sum()),
        "state_rows": len(state),
        "valid_crowding_rows": int(state["crowding_state_valid"].sum()),
        "checkpoint_count": len(summaries),
        "outputs": {
            str(FEATURE_OUTPUT.relative_to(ROOT)): _sha256(FEATURE_OUTPUT),
            str(STATE_OUTPUT.relative_to(ROOT)): _sha256(STATE_OUTPUT),
        },
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-dates", type=int)
    args = parser.parse_args()
    raise SystemExit(main(maximum_dates=args.maximum_dates))
