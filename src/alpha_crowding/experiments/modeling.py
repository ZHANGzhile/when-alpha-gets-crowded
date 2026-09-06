"""Small, auditable M0-M4 Logistic model runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class NestedPredictionResult:
    """Predictions and the audit manifest for one nested comparison."""

    predictions: pd.DataFrame
    manifest: dict[str, object]


@dataclass(frozen=True)
class RegularizationSearchResult:
    """Training-only temporal CV result for each nested information set."""

    selected_c_by_model: dict[str, float]
    scores: pd.DataFrame
    manifest: dict[str, object]


def _ordered_union(feature_sets: Mapping[str, Sequence[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for columns in feature_sets.values():
        for column in columns:
            if column not in seen:
                seen.add(column)
                result.append(column)
    return result


def _validate_nested(feature_sets: Mapping[str, Sequence[str]]) -> None:
    names = list(feature_sets)
    if not names:
        raise ValueError("feature_sets must not be empty")
    previous: set[str] = set()
    for name in names:
        columns = list(feature_sets[name])
        if len(columns) != len(set(columns)):
            raise ValueError(f"{name} contains duplicate feature columns")
        current = set(columns)
        if not previous.issubset(current):
            missing = sorted(previous - current)
            raise ValueError(f"{name} is not nested; removed columns {missing}")
        previous = current


def common_complete_case_mask(
    frame: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    required_columns: Sequence[str],
) -> pd.Series:
    """Return one complete-case mask shared by every nested model."""

    _validate_nested(feature_sets)
    columns = list(dict.fromkeys([*required_columns, *_ordered_union(feature_sets)]))
    missing = set(columns) - set(frame.columns)
    if missing:
        raise KeyError(f"missing model columns: {sorted(missing)}")
    finite = frame[columns].notna().all(axis=1)
    numeric = _ordered_union(feature_sets)
    if numeric:
        values = frame[numeric].apply(pd.to_numeric, errors="coerce")
        finite &= np.isfinite(values).all(axis=1)
    return finite.astype(bool)


def _sample_hash(frame: pd.DataFrame, key_columns: Sequence[str]) -> str:
    missing = set(key_columns) - set(frame.columns)
    if missing:
        raise KeyError(f"missing sample key columns: {sorted(missing)}")
    records = frame[list(key_columns)].astype(str).to_dict("records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _pipeline(numeric_columns: Sequence[str], *, c_value: float) -> Pipeline:
    if c_value <= 0 or not np.isfinite(c_value):
        raise ValueError("c_value must be positive and finite")
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), list(numeric_columns)),
            (
                "factor_fe",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                ["factor"],
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("transform", transform),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=2_000,
                    class_weight=None,
                ),
            ),
        ]
    )


def _model_c_values(
    feature_sets: Mapping[str, Sequence[str]],
    c_value: float | Mapping[str, float],
) -> dict[str, float]:
    if isinstance(c_value, Mapping):
        missing = set(feature_sets) - set(c_value)
        extra = set(c_value) - set(feature_sets)
        if missing or extra:
            raise ValueError(
                f"c_value mapping mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        resolved = {name: float(c_value[name]) for name in feature_sets}
    else:
        resolved = {name: float(c_value) for name in feature_sets}
    if any(value <= 0 or not np.isfinite(value) for value in resolved.values()):
        raise ValueError("all c_values must be positive and finite")
    return resolved


def fit_nested_logistic_models(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    target_col: str = "target",
    key_columns: Sequence[str] = ("decision_at", "factor"),
    c_value: float | Mapping[str, float] = 1.0,
) -> NestedPredictionResult:
    """Fit nested pooled Logistic models with one common mask per split."""

    _validate_nested(feature_sets)
    required = [target_col, "factor", *key_columns]
    train_mask = common_complete_case_mask(
        training, feature_sets, required_columns=required
    )
    eval_mask = common_complete_case_mask(
        evaluation, feature_sets, required_columns=required
    )
    train = training.loc[train_mask].copy()
    evaluate = evaluation.loc[eval_mask].copy()
    if train.empty or evaluate.empty:
        raise ValueError("common complete-case sample is empty")
    y_train = pd.to_numeric(train[target_col], errors="raise").astype(int)
    y_eval = pd.to_numeric(evaluate[target_col], errors="raise").astype(int)
    if not y_train.isin([0, 1]).all() or y_train.nunique() != 2:
        raise ValueError("training target must contain both binary classes")
    if not y_eval.isin([0, 1]).all():
        raise ValueError("evaluation target must be binary")

    output = evaluate[list(key_columns)].copy()
    output[target_col] = y_eval.to_numpy()
    fitted_models: dict[str, Pipeline] = {}
    c_values = _model_c_values(feature_sets, c_value)
    for model_name, columns in feature_sets.items():
        model = _pipeline(columns, c_value=c_values[model_name])
        model.fit(train[[*columns, "factor"]], y_train)
        output[f"probability_{model_name}"] = model.predict_proba(
            evaluate[[*columns, "factor"]]
        )[:, 1]
        fitted_models[model_name] = model

    manifest = {
        "model_order": list(feature_sets),
        "feature_sets": {name: list(columns) for name, columns in feature_sets.items()},
        "train_rows": len(train),
        "evaluation_rows": len(evaluate),
        "train_sample_hash": _sample_hash(train, key_columns),
        "evaluation_sample_hash": _sample_hash(evaluate, key_columns),
        "c_values": c_values,
        "c_value": next(iter(c_values.values())) if len(set(c_values.values())) == 1 else None,
        "class_weight": None,
    }
    return NestedPredictionResult(predictions=output.reset_index(drop=True), manifest=manifest)


def temporal_regularization_search(
    training: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    c_grid: Sequence[float],
    target_col: str = "target",
    key_columns: Sequence[str] = ("decision_at", "factor"),
    validation_weeks: int = 26,
    n_splits: int = 3,
    minimum_training_weeks: int = 104,
    decision_col: str = "decision_at",
    label_start_col: str = "label_start_at",
    label_end_col: str = "label_end_at",
) -> RegularizationSearchResult:
    """Choose one C per model with expanding, purged training-only folds."""

    from .splits import purge_training_rows

    _validate_nested(feature_sets)
    if validation_weeks < 1 or n_splits < 1 or minimum_training_weeks < 1:
        raise ValueError("temporal CV sizes must be positive")
    grid = sorted({float(value) for value in c_grid})
    if not grid or any(value <= 0 or not np.isfinite(value) for value in grid):
        raise ValueError("c_grid must contain positive finite values")
    required = {decision_col, label_start_col, label_end_col, target_col, *key_columns}
    missing = required - set(training.columns)
    if missing:
        raise KeyError(f"missing temporal CV columns: {sorted(missing)}")

    frame = training.copy()
    frame[decision_col] = pd.to_datetime(frame[decision_col], errors="raise").dt.normalize()
    unique_weeks = pd.DatetimeIndex(frame[decision_col].drop_duplicates().sort_values())
    needed = minimum_training_weeks + n_splits * validation_weeks
    if len(unique_weeks) < needed:
        raise ValueError(
            f"temporal CV requires at least {needed} unique weeks; found {len(unique_weeks)}"
        )

    first_validation = len(unique_weeks) - n_splits * validation_weeks
    losses: dict[tuple[str, float], list[pd.Series]] = {
        (model, c): [] for model in feature_sets for c in grid
    }
    split_manifest: list[dict[str, object]] = []
    for split_number in range(n_splits):
        start_position = first_validation + split_number * validation_weeks
        end_position = start_position + validation_weeks
        validation_dates = unique_weeks[start_position:end_position]
        validation = frame[frame[decision_col].isin(validation_dates)].copy()
        fit_cutoff = validation_dates[0] - pd.Timedelta(days=1)
        candidates = frame[frame[decision_col] < validation_dates[0]].copy()
        inner_training = purge_training_rows(
            candidates,
            validation,
            fit_cutoff=fit_cutoff,
            label_start_col=label_start_col,
            label_end_col=label_end_col,
        )
        if inner_training[decision_col].nunique() < minimum_training_weeks:
            raise ValueError("purging left fewer than minimum_training_weeks")
        for c in grid:
            result = fit_nested_logistic_models(
                inner_training,
                validation,
                feature_sets,
                target_col=target_col,
                key_columns=key_columns,
                c_value=c,
            )
            for model in feature_sets:
                losses[(model, c)].append(
                    weekly_average_loss(
                        result.predictions,
                        probability_col=f"probability_{model}",
                        target_col=target_col,
                        week_col=decision_col,
                    )
                )
        split_manifest.append(
            {
                "split": split_number + 1,
                "fit_cutoff": fit_cutoff.isoformat(),
                "validation_start": validation_dates[0].isoformat(),
                "validation_end": validation_dates[-1].isoformat(),
                "training_weeks": int(inner_training[decision_col].nunique()),
                "validation_weeks": int(len(validation_dates)),
            }
        )

    score_rows = []
    for model in feature_sets:
        for c in grid:
            combined = pd.concat(losses[(model, c)]).sort_index()
            score_rows.append(
                {
                    "model": model,
                    "c_value": c,
                    "mean_weekly_log_loss": float(combined.mean()),
                    "validation_week_count": int(len(combined)),
                }
            )
    scores = pd.DataFrame(score_rows).sort_values(
        ["model", "mean_weekly_log_loss", "c_value"]
    ).reset_index(drop=True)
    selected = {
        model: float(scores.loc[scores.model == model].iloc[0].c_value)
        for model in feature_sets
    }
    return RegularizationSearchResult(
        selected_c_by_model=selected,
        scores=scores,
        manifest={
            "method": "expanding_purged_temporal_cv",
            "c_grid": grid,
            "validation_weeks_per_split": validation_weeks,
            "n_splits": n_splits,
            "minimum_training_weeks": minimum_training_weeks,
            "splits": split_manifest,
            "selected_c_by_model": selected,
        },
    )


def binary_log_loss(target: Sequence[float], probability: Sequence[float]) -> np.ndarray:
    """Return row-level binary Log Loss with the protocol's numeric clipping."""

    y = np.asarray(target, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape:
        raise ValueError("target and probability shapes differ")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("target must be binary")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probability must be finite and in [0, 1]")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def weekly_average_loss(
    predictions: pd.DataFrame,
    *,
    probability_col: str,
    target_col: str = "target",
    week_col: str = "decision_at",
) -> pd.Series:
    """Average factor losses within each week, then expose the weekly series."""

    required = {probability_col, target_col, week_col, "factor"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"missing prediction columns: {sorted(missing)}")
    frame = predictions.copy()
    frame[week_col] = pd.to_datetime(frame[week_col], errors="raise")
    if frame.duplicated([week_col, "factor"]).any():
        raise ValueError("predictions must be unique by week/factor")
    frame["_loss"] = binary_log_loss(frame[target_col], frame[probability_col])
    return frame.groupby(week_col, sort=True)["_loss"].mean()
