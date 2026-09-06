"""Time-split safeguards for overlapping outcome intervals and lead tests."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _dates(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise KeyError(f"missing required column: {column!r}")
    values = pd.to_datetime(frame[column], errors="raise")
    if values.isna().any():
        raise ValueError(f"{column!r} contains missing timestamps")
    return values.dt.normalize()


def _validated_intervals(
    frame: pd.DataFrame,
    start_col: str,
    end_col: str,
) -> tuple[pd.Series, pd.Series]:
    start = _dates(frame, start_col)
    end = _dates(frame, end_col)
    invalid = end < start
    if invalid.any():
        bad = frame.index[invalid].tolist()[:5]
        raise ValueError(f"interval end precedes start at rows {bad}")
    return start, end


def _merged_intervals(
    start: pd.Series,
    end: pd.Series,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(start) == 0:
        return []
    ordered = sorted(zip(start.tolist(), end.tolist()), key=lambda pair: pair[0])
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start, current_end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    return merged


def purged_training_mask(
    training: pd.DataFrame,
    protected: pd.DataFrame | None = None,
    *,
    fit_cutoff: object | None = None,
    label_start_col: str = "label_start_at",
    label_end_col: str = "label_end_at",
) -> pd.Series:
    """Return rows whose labels are mature and do not overlap protected labels.

    Intervals are closed on both ends.  In particular, a training outcome that
    ends on the same session a protected outcome begins is purged.  A scalar
    ``fit_cutoff`` additionally enforces that every retained training label was
    completely observable when the model was fit.
    """

    train_start, train_end = _validated_intervals(
        training, label_start_col, label_end_col
    )
    keep = pd.Series(True, index=training.index, dtype=bool)

    if fit_cutoff is not None:
        cutoff = pd.Timestamp(fit_cutoff).normalize()
        if pd.isna(cutoff):
            raise ValueError("fit_cutoff must not be missing")
        keep &= train_end <= cutoff

    if protected is not None and not protected.empty:
        protected_start, protected_end = _validated_intervals(
            protected, label_start_col, label_end_col
        )
        for interval_start, interval_end in _merged_intervals(
            protected_start, protected_end
        ):
            overlap = (train_start <= interval_end) & (train_end >= interval_start)
            keep &= ~overlap
    return keep


def purge_training_rows(
    training: pd.DataFrame,
    protected: pd.DataFrame | None = None,
    *,
    fit_cutoff: object | None = None,
    label_start_col: str = "label_start_at",
    label_end_col: str = "label_end_at",
) -> pd.DataFrame:
    """Return a copy of the eligible training rows after interval purging."""

    mask = purged_training_mask(
        training,
        protected,
        fit_cutoff=fit_cutoff,
        label_start_col=label_start_col,
        label_end_col=label_end_col,
    )
    return training.loc[mask].copy()


def assert_no_label_interval_overlap(
    training: pd.DataFrame,
    protected: pd.DataFrame,
    *,
    label_start_col: str = "label_start_at",
    label_end_col: str = "label_end_at",
) -> None:
    """Raise when any closed training interval overlaps a protected interval."""

    mask = purged_training_mask(
        training,
        protected,
        label_start_col=label_start_col,
        label_end_col=label_end_col,
    )
    if not mask.all():
        bad = training.index[~mask].tolist()[:5]
        raise ValueError(f"label intervals overlap protected data at rows {bad}")


def _session_index(sessions: Sequence[object]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(list(sessions), errors="raise")).normalize()
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("sessions must be unique, non-missing, and increasing")
    if len(index) == 0:
        raise ValueError("sessions must not be empty")
    return index


def lead_time_cutoffs(
    target_anchors: pd.Series | Sequence[object],
    *,
    sessions: Sequence[object],
    lead_sessions: int,
) -> pd.Series:
    """Map target anchors to their exact earlier trading-session cutoffs."""

    if lead_sessions < 0:
        raise ValueError("lead_sessions must be non-negative")
    if isinstance(target_anchors, pd.Series):
        index = target_anchors.index
        anchors = pd.to_datetime(target_anchors, errors="raise").dt.normalize()
    else:
        index = pd.RangeIndex(len(target_anchors))
        anchors = pd.Series(pd.to_datetime(list(target_anchors), errors="raise")).dt.normalize()
    if anchors.isna().any():
        raise ValueError("target anchors must not be missing")

    calendar = _session_index(sessions)
    positions = calendar.get_indexer(pd.DatetimeIndex(anchors))
    if (positions < 0).any():
        examples = anchors.iloc[np.flatnonzero(positions < 0)[:5]].astype(str).tolist()
        raise ValueError(f"target anchors are outside sessions: {examples}")
    if (positions < lead_sessions).any():
        examples = anchors.iloc[
            np.flatnonzero(positions < lead_sessions)[:5]
        ].astype(str).tolist()
        raise ValueError(f"insufficient session history for target anchors: {examples}")
    cutoffs = calendar.take(positions - lead_sessions)
    return pd.Series(cutoffs.to_numpy(), index=index, name="expected_information_cutoff")


def assert_lead_time_information_cutoff(
    frame: pd.DataFrame,
    *,
    sessions: Sequence[object],
    lead_sessions: int,
    target_anchor_col: str = "decision_at",
    information_cutoff_col: str = "information_cutoff_at",
    available_at_cols: Sequence[str] = (),
    label_start_col: str | None = "label_start_at",
    require_exact_cutoff: bool = True,
) -> None:
    """Assert that a lead-time experiment uses no post-alert information.

    With ``require_exact_cutoff=True``, the information cutoff must be exactly
    ``lead_sessions`` trading sessions before the target anchor.  When false,
    an older cutoff is accepted.  Every column in ``available_at_cols`` must be
    dated on or before the row's information cutoff; this is intended to cover
    features, membership, historical thresholds, preprocessing, model fit, and
    probability calibration vintages.
    """

    target = _dates(frame, target_anchor_col)
    actual_cutoff = _dates(frame, information_cutoff_col)
    expected_cutoff = lead_time_cutoffs(
        target,
        sessions=sessions,
        lead_sessions=lead_sessions,
    )
    expected_cutoff.index = frame.index

    invalid_cutoff = (
        actual_cutoff != expected_cutoff
        if require_exact_cutoff
        else actual_cutoff > expected_cutoff
    )
    if invalid_cutoff.any():
        bad = frame.index[invalid_cutoff].tolist()[:5]
        relation = "equal" if require_exact_cutoff else "no later than"
        raise ValueError(
            f"information cutoff must be {relation} the expected lead cutoff; "
            f"invalid rows {bad}"
        )

    for column in available_at_cols:
        available_at = _dates(frame, column)
        leaked = available_at > actual_cutoff
        if leaked.any():
            bad = frame.index[leaked].tolist()[:5]
            raise ValueError(
                f"{column!r} is later than the information cutoff at rows {bad}"
            )

    if label_start_col is not None:
        label_start = _dates(frame, label_start_col)
        invalid_start = label_start <= target
        if invalid_start.any():
            bad = frame.index[invalid_start].tolist()[:5]
            raise ValueError(
                "label_start must be strictly after the target anchor at rows "
                f"{bad}"
            )
