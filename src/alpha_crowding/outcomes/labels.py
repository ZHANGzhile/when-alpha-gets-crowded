"""Point-in-time tail-label construction for factor outcomes.

The helpers in this module deliberately treat an outcome as usable history only
after its complete label interval has matured.  This matters for overlapping
20-session factor outcomes: an observation date in the past is not sufficient
to make its future return known at the current information cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatureTailSpec:
    """Configuration for a point-in-time historical tail threshold."""

    quantile: float = 0.10
    lookback_sessions: int = 756
    min_history: int = 104
    quantile_method: str = "linear"

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must lie strictly between zero and one")
        if self.lookback_sessions < 1:
            raise ValueError("lookback_sessions must be positive")
        if self.min_history < 1:
            raise ValueError("min_history must be positive")


def _date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="raise")
    if values.isna().any():
        raise ValueError(f"{column!r} contains missing timestamps")
    return values.dt.normalize()


def _session_index(sessions: Sequence[object]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(list(sessions), errors="raise")).normalize()
    if index.hasnans:
        raise ValueError("sessions contains missing timestamps")
    if index.has_duplicates:
        raise ValueError("sessions must not contain duplicates")
    if not index.is_monotonic_increasing:
        raise ValueError("sessions must be in strictly increasing order")
    if len(index) == 0:
        raise ValueError("sessions must not be empty")
    return index


def mature_historical_quantiles(
    outcomes: pd.DataFrame,
    *,
    sessions: Sequence[object],
    group_cols: Sequence[str] = ("factor", "target_family", "membership_mode"),
    origin_col: str = "decision_at",
    label_start_col: str = "label_start_at",
    label_end_col: str = "label_end_at",
    outcome_col: str = "future_return",
    information_cutoff_col: str | None = None,
    spec: MatureTailSpec | None = None,
    threshold_name: str = "historical_tail_threshold",
    history_count_name: str = "mature_history_count",
) -> pd.DataFrame:
    """Calculate historical quantiles using only fully matured outcomes.

    The lookback is measured on ``sessions`` rather than by DataFrame row count.
    For a row with information cutoff ``a``, eligible history must:

    * belong to the same group (normally factor and outcome family),
    * have a label end on or before ``a``, and
    * have an origin inside the last ``lookback_sessions`` sessions through
      ``a``.

    Dynamic and fixed-membership outcomes should therefore be assigned distinct
    values in ``membership_mode``; research LS and active-long targets remain
    separated by ``target_family``.  The function returns a copy of ``outcomes``.
    """

    spec = spec or MatureTailSpec()
    group_cols = tuple(group_cols)
    required = {
        *group_cols,
        origin_col,
        label_start_col,
        label_end_col,
        outcome_col,
    }
    if information_cutoff_col is not None:
        required.add(information_cutoff_col)
    missing = required.difference(outcomes.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    result = outcomes.copy()
    if result.empty:
        result[threshold_name] = pd.Series(dtype=float, index=result.index)
        result[history_count_name] = pd.Series(dtype="int64", index=result.index)
        return result

    origin = _date_series(result, origin_col)
    label_start = _date_series(result, label_start_col)
    label_end = _date_series(result, label_end_col)
    cutoff = (
        origin.copy()
        if information_cutoff_col is None
        else _date_series(result, information_cutoff_col)
    )
    calendar = _session_index(sessions)

    if (label_end < label_start).any():
        bad = result.index[label_end < label_start].tolist()[:5]
        raise ValueError(f"label_end precedes label_start at rows {bad}")
    if (cutoff > origin).any():
        bad = result.index[cutoff > origin].tolist()[:5]
        raise ValueError(f"information cutoff follows decision date at rows {bad}")

    checked_dates = {
        origin_col: origin,
        label_start_col: label_start,
        label_end_col: label_end,
        information_cutoff_col or origin_col: cutoff,
    }
    for name, values in checked_dates.items():
        outside = ~values.isin(calendar)
        if outside.any():
            examples = values[outside].astype(str).tolist()[:5]
            raise ValueError(f"{name!r} contains dates outside sessions: {examples}")

    duplicate_subset = [*group_cols, origin_col]
    duplicated = result.duplicated(duplicate_subset, keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, duplicate_subset].head().to_dict("records")
        raise ValueError(
            "each group must have at most one outcome per decision date; "
            f"duplicates include {examples}"
        )

    values = pd.to_numeric(result[outcome_col], errors="coerce").to_numpy(dtype=float)
    origin_values = origin.to_numpy(dtype="datetime64[ns]")
    end_values = label_end.to_numpy(dtype="datetime64[ns]")
    cutoff_values = cutoff.to_numpy(dtype="datetime64[ns]")

    thresholds = np.full(len(result), np.nan, dtype=float)
    history_counts = np.zeros(len(result), dtype=np.int64)

    if group_cols:
        grouped_positions = result.groupby(
            list(group_cols), sort=False, dropna=False
        ).indices.values()
    else:
        grouped_positions = (np.arange(len(result), dtype=int),)

    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    for raw_positions in grouped_positions:
        positions = np.asarray(raw_positions, dtype=int)
        for current in positions:
            current_cutoff = cutoff_values[current]
            cutoff_position = int(
                np.searchsorted(calendar_values, current_cutoff, side="right") - 1
            )
            lower_position = max(0, cutoff_position - spec.lookback_sessions + 1)
            lower_date = calendar_values[lower_position]

            eligible = positions[
                (positions != current)
                & (end_values[positions] <= current_cutoff)
                & (origin_values[positions] >= lower_date)
                & (origin_values[positions] <= current_cutoff)
                & np.isfinite(values[positions])
            ]
            history_counts[current] = len(eligible)
            if len(eligible) >= spec.min_history:
                thresholds[current] = float(
                    np.quantile(
                        values[eligible],
                        spec.quantile,
                        method=spec.quantile_method,
                    )
                )

    result[threshold_name] = thresholds
    result[history_count_name] = history_counts
    return result


def build_mature_tail_labels(
    outcomes: pd.DataFrame,
    *,
    sessions: Sequence[object],
    group_cols: Sequence[str] = ("factor", "target_family", "membership_mode"),
    origin_col: str = "decision_at",
    label_start_col: str = "label_start_at",
    label_end_col: str = "label_end_at",
    outcome_col: str = "future_return",
    information_cutoff_col: str | None = None,
    spec: MatureTailSpec | None = None,
    threshold_name: str = "historical_tail_threshold",
    history_count_name: str = "mature_history_count",
    label_name: str = "tail_event",
) -> pd.DataFrame:
    """Add a nullable tail-event label to point-in-time outcome records.

    A label is missing when its current outcome is missing or the required
    matured history is unavailable.  Strict inequality implements the event
    definition ``future_return < historical_quantile``.
    """

    result = mature_historical_quantiles(
        outcomes,
        sessions=sessions,
        group_cols=group_cols,
        origin_col=origin_col,
        label_start_col=label_start_col,
        label_end_col=label_end_col,
        outcome_col=outcome_col,
        information_cutoff_col=information_cutoff_col,
        spec=spec,
        threshold_name=threshold_name,
        history_count_name=history_count_name,
    )
    current = pd.to_numeric(result[outcome_col], errors="coerce")
    threshold = result[threshold_name]
    valid = current.notna() & threshold.notna()
    labels = pd.Series(pd.NA, index=result.index, dtype="boolean")
    labels.loc[valid] = current.loc[valid] < threshold.loc[valid]
    result[label_name] = labels
    return result
