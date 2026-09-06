"""Annual expanding-window out-of-sample evaluation.

The fold builder keeps the time contract separate from model fitting.  Every
annual model is fit once, immediately before its evaluation interval, using
only outcomes that were fully mature by that cutoff.  Closed label intervals
that overlap the protected evaluation outcomes are purged as an additional
guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .modeling import (
    NestedPredictionResult,
    fit_nested_logistic_models,
    temporal_regularization_search,
)
from .splits import purge_training_rows


@dataclass(frozen=True)
class AnnualFold:
    """Materialized training and evaluation samples for one calendar fold."""

    name: str
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    fit_cutoff: pd.Timestamp
    training: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class WalkForwardResult:
    """Concatenated out-of-sample predictions and per-fold audit metadata."""

    predictions: pd.DataFrame
    manifest: dict[str, object]


def _normalized_date(value: object, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{name} must not be missing")
    if result.tz is not None:
        result = result.tz_localize(None)
    return result.normalize()


def annual_expanding_folds(
    frame: pd.DataFrame,
    *,
    evaluation_start: object,
    evaluation_end: object,
    raw_data_cutoff: object | None = None,
    decision_col: str = "decision_at",
    label_start_col: str = "label_start_at",
    label_end_col: str = "label_end_at",
    key_columns: Sequence[str] = ("decision_at", "factor"),
) -> list[AnnualFold]:
    """Build annual expanding folds with maturity and interval safeguards.

    The requested evaluation window may begin or end partway through a year.
    Evaluation rows whose labels have not matured by ``raw_data_cutoff`` are
    excluded.  Empty years are omitted, while an empty training sample is a
    hard error because no honest model can be fit for that year.
    """

    required = {decision_col, label_start_col, label_end_col, *key_columns}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing walk-forward columns: {sorted(missing)}")
    if frame.duplicated(list(key_columns)).any():
        raise ValueError("panel must be unique by the walk-forward key columns")

    start = _normalized_date(evaluation_start, name="evaluation_start")
    end = _normalized_date(evaluation_end, name="evaluation_end")
    if end < start:
        raise ValueError("evaluation_end precedes evaluation_start")
    data_cutoff = (
        _normalized_date(raw_data_cutoff, name="raw_data_cutoff")
        if raw_data_cutoff is not None
        else end
    )

    materialized = frame.copy()
    materialized[decision_col] = pd.to_datetime(
        materialized[decision_col], errors="raise"
    ).dt.normalize()
    materialized[label_start_col] = pd.to_datetime(
        materialized[label_start_col], errors="raise"
    ).dt.normalize()
    materialized[label_end_col] = pd.to_datetime(
        materialized[label_end_col], errors="raise"
    ).dt.normalize()
    if materialized[[decision_col, label_start_col, label_end_col]].isna().any().any():
        raise ValueError("walk-forward dates must not be missing")
    if (materialized[label_start_col] <= materialized[decision_col]).any():
        raise ValueError("label_start_at must be after decision_at")
    if (materialized[label_end_col] < materialized[label_start_col]).any():
        raise ValueError("label_end_at precedes label_start_at")

    folds: list[AnnualFold] = []
    for year in range(start.year, end.year + 1):
        fold_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        fold_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        evaluation_mask = (
            materialized[decision_col].between(fold_start, fold_end)
            & (materialized[label_end_col] <= data_cutoff)
        )
        evaluation = materialized.loc[evaluation_mask].copy()
        if evaluation.empty:
            continue

        fit_cutoff = fold_start - pd.Timedelta(days=1)
        candidates = materialized.loc[
            materialized[decision_col] < fold_start
        ].copy()
        training = purge_training_rows(
            candidates,
            evaluation,
            fit_cutoff=fit_cutoff,
            label_start_col=label_start_col,
            label_end_col=label_end_col,
        )
        if training.empty:
            raise ValueError(f"annual fold {year} has no eligible training rows")
        folds.append(
            AnnualFold(
                name=str(year),
                evaluation_start=fold_start,
                evaluation_end=fold_end,
                fit_cutoff=fit_cutoff,
                training=training.reset_index(drop=True),
                evaluation=evaluation.reset_index(drop=True),
            )
        )
    if not folds:
        raise ValueError("no mature evaluation rows exist in the requested window")
    return folds


def run_annual_walk_forward(
    frame: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    evaluation_start: object,
    evaluation_end: object,
    raw_data_cutoff: object | None = None,
    target_col: str = "target",
    key_columns: Sequence[str] = ("decision_at", "factor"),
    c_value: float | Mapping[str, float] = 1.0,
    c_grid: Sequence[float] | None = None,
    inner_validation_weeks: int = 26,
    inner_splits: int = 3,
    minimum_inner_training_weeks: int = 104,
) -> WalkForwardResult:
    """Fit the same nested models independently in each annual outer fold."""

    folds = annual_expanding_folds(
        frame,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        raw_data_cutoff=raw_data_cutoff,
        key_columns=key_columns,
    )
    prediction_parts: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, object]] = []
    for fold in folds:
        tuning_manifest: dict[str, object] | None = None
        fold_c_value = c_value
        if c_grid is not None:
            tuning = temporal_regularization_search(
                fold.training,
                feature_sets,
                c_grid=c_grid,
                target_col=target_col,
                key_columns=key_columns,
                validation_weeks=inner_validation_weeks,
                n_splits=inner_splits,
                minimum_training_weeks=minimum_inner_training_weeks,
            )
            fold_c_value = tuning.selected_c_by_model
            tuning_manifest = {
                **tuning.manifest,
                "scores": tuning.scores.to_dict("records"),
            }
        fitted: NestedPredictionResult = fit_nested_logistic_models(
            fold.training,
            fold.evaluation,
            feature_sets,
            target_col=target_col,
            key_columns=key_columns,
            c_value=fold_c_value,
        )
        predictions = fitted.predictions.copy()
        predictions["fold"] = fold.name
        predictions["fit_cutoff_at"] = fold.fit_cutoff
        prediction_parts.append(predictions)
        fold_manifests.append(
            {
                "fold": fold.name,
                "evaluation_start": fold.evaluation_start.isoformat(),
                "evaluation_end": fold.evaluation_end.isoformat(),
                "fit_cutoff": fold.fit_cutoff.isoformat(),
                "regularization_search": tuning_manifest,
                **fitted.manifest,
            }
        )

    combined = pd.concat(prediction_parts, ignore_index=True)
    if combined.duplicated(list(key_columns)).any():
        raise AssertionError("a row was predicted in more than one outer fold")
    combined = combined.sort_values(list(key_columns)).reset_index(drop=True)
    manifest: dict[str, object] = {
        "scheme": "annual_expanding",
        "evaluation_start": _normalized_date(
            evaluation_start, name="evaluation_start"
        ).isoformat(),
        "evaluation_end": _normalized_date(
            evaluation_end, name="evaluation_end"
        ).isoformat(),
        "raw_data_cutoff": (
            _normalized_date(raw_data_cutoff, name="raw_data_cutoff").isoformat()
            if raw_data_cutoff is not None
            else None
        ),
        "prediction_rows": len(combined),
        "folds": fold_manifests,
    }
    return WalkForwardResult(predictions=combined, manifest=manifest)
