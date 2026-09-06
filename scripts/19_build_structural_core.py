"""Build production Residual Sync/Eigen features and matched-placebo audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from alpha_crowding.measurement import (
    ALGORITHM_VERSION,
    historical_zscore_by_group_trading_window,
    measure_structural_placebos,
)


ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "interim" / "daily_market"
MEMBERSHIPS = ROOT / "data" / "processed" / "factor_memberships.parquet"
CONFIG = ROOT / "config" / "measurement.yaml"
CHECKPOINTS = ROOT / "data" / "processed" / "structural_core_by_date"
OUTPUT = ROOT / "data" / "processed" / "structural_features_core.parquet"
MANIFEST = ROOT / "data" / "raw" / "manifests" / "structural_features_core.json"


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


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


def _trading_calendar() -> pd.DatetimeIndex:
    dates = []
    for path in sorted(DAILY.glob("year=*/*.parquet")):
        dates.append(pd.read_parquet(path, columns=["date"])["date"])
    calendar = pd.DatetimeIndex(pd.concat(dates, ignore_index=True).unique()).normalize()
    return calendar.sort_values()


def _invalid_group_rows(
    *, decision_at: pd.Timestamp, factor: str, leg: str, reason: str, draws: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        [
            {
                "decision_at": decision_at,
                "factor": factor,
                "leg": leg,
                "feature": feature,
                "actual": np.nan,
                "placebo_mean": np.nan,
                "placebo_std": np.nan,
                "excess": np.nan,
                "placebo_z": np.nan,
                "valid_draws": 0,
                "total_draws": draws,
                "placebo_valid": False,
                "invalid_reason": reason,
                "actual_valid_days": 0,
                "actual_valid_assets": 0,
                "window_start": pd.NaT,
                "window_end": pd.NaT,
                "candidate_snapshot_sha256": None,
                "algorithm_version": ALGORITHM_VERSION,
            }
            for feature in ("residual_sync", "eigen_concentration")
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "decision_at": decision_at,
                "factor": factor,
                "leg": leg,
                "draw_id": None,
                "valid": False,
                "invalid_reason": reason,
                "algorithm_version": ALGORITHM_VERSION,
            }
        ]
    )
    return features, diagnostics


def _process_date(
    decision_at: pd.Timestamp,
    memberships: pd.DataFrame,
    daily_window: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: dict[str, object],
) -> dict[str, object]:
    directory = CHECKPOINTS / str(decision_at.date())
    marker = directory / "status.json"
    expected = [
        directory / "features.parquet",
        directory / "placebo_features.parquet",
        directory / "placebo_diagnostics.parquet",
        directory / "placebo_memberships.parquet",
    ]
    if marker.exists() and all(path.exists() for path in expected):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["source"] = "checkpoint"
        return payload

    session_position = int(calendar.get_loc(decision_at))
    window_dates = calendar[max(0, session_position - 59) : session_position + 1]
    market = daily_window[daily_window["date"].isin(window_dates)].copy()
    date_memberships = memberships[memberships["date"].eq(decision_at)]
    feature_parts = []
    draw_parts = []
    diagnostic_parts = []
    member_parts = []
    group_failures = []
    placebo = config["placebo"]
    for factor, candidates in date_memberships.groupby("factor", sort=True):
        candidate_frame = candidates[["code", "industry", "lagged_liquidity"]].copy()
        for leg in ("LONG", "SHORT"):
            actual_codes = candidates.loc[candidates["leg"].eq(leg), "code"].tolist()
            if not actual_codes:
                continue
            try:
                result = measure_structural_placebos(
                    market,
                    candidate_frame,
                    actual_codes,
                    decision_at=decision_at,
                    factor=str(factor),
                    leg=leg,
                    draws=int(placebo["draws"]),
                    minimum_valid_draws=int(placebo["minimum_valid_draws"]),
                    base_seed=int(placebo["seed"]),
                    minimum_days=40,
                    minimum_assets=10,
                    temperature=float(placebo["liquidity_match_temperature"]),
                    maximum_mean_absolute_z_difference=float(
                        placebo["maximum_mean_absolute_liquidity_z_difference"]
                    ),
                    maximum_attempts_per_draw=int(placebo["maximum_attempts_per_draw"]),
                )
                feature_parts.append(result.features)
                draw_parts.append(result.placebo_features)
                diagnostic_parts.append(result.placebo_diagnostics)
                member_parts.append(result.placebo_memberships)
            except ValueError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                invalid_features, invalid_diagnostics = _invalid_group_rows(
                    decision_at=decision_at,
                    factor=str(factor),
                    leg=leg,
                    reason=reason,
                    draws=int(placebo["draws"]),
                )
                feature_parts.append(invalid_features)
                diagnostic_parts.append(invalid_diagnostics)
                group_failures.append({"factor": str(factor), "leg": leg, "reason": reason})
    outputs = {
        "features": pd.concat(feature_parts, ignore_index=True),
        "placebo_features": pd.concat(draw_parts, ignore_index=True) if draw_parts else pd.DataFrame(
            columns=["decision_at", "factor", "leg", "draw_id", "feature", "value", "valid_days", "valid_assets", "invalid_reason"]
        ),
        "placebo_diagnostics": pd.concat(diagnostic_parts, ignore_index=True),
        "placebo_memberships": pd.concat(member_parts, ignore_index=True) if member_parts else pd.DataFrame(
            columns=["decision_at", "factor", "leg", "draw_id", "code", "industry", "matched_to_code", "liquidity_z", "actual_liquidity_z", "match_distance", "is_actual_member"]
        ),
    }
    for name, frame in outputs.items():
        _atomic_parquet(frame, directory / f"{name}.parquet")
    payload = {
        "decision_at": str(decision_at.date()),
        "source": "computed",
        "window_start": str(window_dates.min().date()),
        "window_end": str(window_dates.max().date()),
        "window_sessions": len(window_dates),
        "feature_rows": len(outputs["features"]),
        "placebo_feature_rows": len(outputs["placebo_features"]),
        "placebo_membership_rows": len(outputs["placebo_memberships"]),
        "group_failures": group_failures,
    }
    _atomic_json(payload, marker)
    return payload


def main(
    *, start_date: str | None,
    end_date: str | None,
    maximum_dates: int | None,
) -> int:
    if maximum_dates is not None and maximum_dates < 1:
        raise ValueError("maximum_dates must be positive")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    memberships = pd.read_parquet(MEMBERSHIPS)
    memberships["date"] = pd.to_datetime(memberships["date"]).dt.normalize()
    calendar = _trading_calendar()
    dates = pd.DatetimeIndex(sorted(memberships["date"].unique()))
    if start_date is not None:
        dates = dates[dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = dates[dates <= pd.Timestamp(end_date)]
    if maximum_dates is not None:
        dates = dates[:maximum_dates]
    absent = dates.difference(calendar)
    if len(absent):
        raise ValueError(f"decision dates absent from trading calendar: {list(absent[:5])}")

    summaries = []
    loaded_year = None
    daily_window = pd.DataFrame()
    for position, decision_at in enumerate(dates, start=1):
        if loaded_year != decision_at.year:
            daily_window = _read_year_window(decision_at.year)
            daily_window["date"] = pd.to_datetime(daily_window["date"]).dt.normalize()
            loaded_year = decision_at.year
        summary = _process_date(
            decision_at, memberships, daily_window, calendar, config
        )
        summaries.append(summary)
        print(
            f"structural core {position}/{len(dates)} date={decision_at.date()} "
            f"source={summary['source']} failures={len(summary['group_failures'])}",
            flush=True,
        )

    feature_paths = [CHECKPOINTS / str(date.date()) / "features.parquet" for date in dates]
    features = pd.concat([pd.read_parquet(path) for path in feature_paths], ignore_index=True)
    standardized = historical_zscore_by_group_trading_window(
        features,
        calendar,
        value_col="excess",
        group_cols=("factor", "leg", "feature"),
        order_col="decision_at",
        lookback_sessions=int(config["historical_window_trading_days"]),
        min_periods=int(config["historical_min_weekly_observations"]),
    ).add_prefix("excess_")
    actual_standardized = historical_zscore_by_group_trading_window(
        features,
        calendar,
        value_col="actual",
        group_cols=("factor", "leg", "feature"),
        order_col="decision_at",
        lookback_sessions=int(config["historical_window_trading_days"]),
        min_periods=int(config["historical_min_weekly_observations"]),
    ).add_prefix("actual_")
    features = pd.concat(
        [
            features,
            standardized.drop(columns="excess_value"),
            actual_standardized.drop(columns="actual_value"),
        ],
        axis=1,
    )
    _atomic_parquet(features, OUTPUT)
    payload = {
        "schema_version": 1,
        "purpose": "production_structural_core_residual_sync_and_eigen",
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_version": ALGORITHM_VERSION,
        "decision_dates": len(dates),
        "feature_rows": len(features),
        "valid_feature_rows": int(features["placebo_valid"].sum()),
        "group_failures": sum(len(item["group_failures"]) for item in summaries),
        "placebo_draws": int(config["placebo"]["draws"]),
        "historical_window_trading_days": int(config["historical_window_trading_days"]),
        "historical_min_weekly_observations": int(config["historical_min_weekly_observations"]),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "checkpoint_directory": str(CHECKPOINTS.relative_to(ROOT)),
        "scope_note": "Residual Sync and Eigen core only; joint Strategy Convergence is separate.",
    }
    _atomic_json(payload, MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--maximum-dates", type=int)
    args = parser.parse_args()
    raise SystemExit(
        main(
            start_date=args.start_date,
            end_date=args.end_date,
            maximum_dates=args.maximum_dates,
        )
    )
