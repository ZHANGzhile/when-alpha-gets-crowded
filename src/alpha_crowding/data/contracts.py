"""Reusable invariants for point-in-time research tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class AvailabilityViolation:
    """A source row that was unavailable at the claimed decision time."""

    row_index: object
    available_at: pd.Timestamp
    decision_at: pd.Timestamp


def _as_utc_naive(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_datetime(series, errors="coerce", utc=True)
    if values.isna().any():
        bad = series.index[values.isna()].tolist()[:5]
        raise ValueError(f"{name} contains invalid timestamps at rows {bad}")
    return values.dt.tz_convert(None)


def assert_information_available(
    frame: pd.DataFrame,
    *,
    available_col: str = "available_at",
    decision_col: str = "decision_at",
) -> None:
    """Reject rows whose source information became available after the decision."""

    missing = {available_col, decision_col} - set(frame.columns)
    if missing:
        raise KeyError(f"missing availability columns: {sorted(missing)}")
    available = _as_utc_naive(frame[available_col], available_col)
    decision = _as_utc_naive(frame[decision_col], decision_col)
    invalid = available > decision
    if invalid.any():
        sample = [
            AvailabilityViolation(idx, available.loc[idx], decision.loc[idx])
            for idx in frame.index[invalid][:5]
        ]
        raise ValueError(f"{invalid.sum()} point-in-time violations; sample={sample}")


def assert_unique_key(frame: pd.DataFrame, key: Iterable[str]) -> None:
    """Require an explicit unique primary key."""

    key = list(key)
    missing = set(key) - set(frame.columns)
    if missing:
        raise KeyError(f"missing key columns: {sorted(missing)}")
    duplicated = frame.duplicated(key, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, key].head(5).to_dict("records")
        raise ValueError(f"duplicate key rows: {examples}")


def validate_interval(
    frame: pd.DataFrame,
    *,
    start_col: str = "label_start_at",
    end_col: str = "label_end_at",
    cutoff_col: str | None = None,
) -> None:
    """Validate ordered outcome intervals and optional maturity cutoffs."""

    required = {start_col, end_col}
    if cutoff_col:
        required.add(cutoff_col)
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing interval columns: {sorted(missing)}")
    start = _as_utc_naive(frame[start_col], start_col)
    end = _as_utc_naive(frame[end_col], end_col)
    if (end < start).any():
        raise ValueError("label_end_at precedes label_start_at")
    if cutoff_col:
        cutoff = _as_utc_naive(frame[cutoff_col], cutoff_col)
        if (end > cutoff).any():
            raise ValueError("immature outcome interval exceeds its information cutoff")

