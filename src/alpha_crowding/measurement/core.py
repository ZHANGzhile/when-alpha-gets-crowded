"""Causal and auditable measurement helpers.

The functions here intentionally avoid learned weights.  They validate inputs,
preserve missingness, and expose the sample counts and scale estimates needed to
audit every state value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


CROWDING_COMPONENTS: tuple[str, ...] = (
    "excess_sync_historical_z",
    "excess_eigen_historical_z",
    "excess_overlap_historical_z",
)

GENERIC_RISK_COMPONENTS: tuple[str, ...] = (
    "raw_sync_z",
    "raw_eigen_z",
    "turnover_level_z",
    "illiquidity_level_z",
)

STRESS_COMPONENTS: tuple[str, ...] = (
    "turnover_shock_z",
    "turnover_sync_z",
    "liquidity_stress_z",
    "factor_return_shock_z",
)

_CROWDING_FORBIDDEN_TOKENS = (
    "turnover",
    "liquidity",
    "illiquidity",
    "stress",
    "shock",
)
_GENERIC_FORBIDDEN_TOKENS = ("excess", "placebo")


@dataclass(frozen=True, slots=True)
class PlaceboSummary:
    """Summary of one actual measurement against matched placebo draws."""

    actual: float
    placebo_mean: float
    placebo_std: float
    excess: float
    placebo_z: float
    valid_draws: int
    total_draws: int
    placebo_valid: bool
    invalid_reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_components(components: Sequence[str]) -> tuple[str, ...]:
    if isinstance(components, (str, bytes)):
        raise TypeError("components must be a sequence of column names")
    result = tuple(components)
    if not result:
        raise ValueError("components must not be empty")
    if any(not isinstance(name, str) or not name for name in result):
        raise TypeError("each component name must be a non-empty string")
    if len(set(result)) != len(result):
        raise ValueError("component names must be unique")
    return result


def _validate_family_component_names(
    names: Sequence[str], family: str | None
) -> None:
    normalized_family = None if family is None else family.strip().lower()
    if normalized_family == "crowding":
        forbidden = [
            name
            for name in names
            if any(token in name.lower() for token in _CROWDING_FORBIDDEN_TOKENS)
        ]
        if forbidden:
            raise ValueError(
                "crowding components must describe structural abnormality; "
                f"stress-like components are forbidden: {forbidden}"
            )
    elif normalized_family in {"generic", "generic_risk"}:
        forbidden = [
            name
            for name in names
            if any(token in name.lower() for token in _GENERIC_FORBIDDEN_TOKENS)
        ]
        if forbidden:
            raise ValueError(
                "generic-risk components must be raw portfolio characteristics; "
                f"placebo-adjusted components are forbidden: {forbidden}"
            )
    elif normalized_family not in {None, "stress"}:
        raise ValueError("family must be one of: crowding, generic_risk, stress")


def validate_component_frame(
    frame: pd.DataFrame,
    components: Sequence[str],
    *,
    family: str | None = None,
    require_unique_index: bool = True,
) -> tuple[str, ...]:
    """Validate component columns used to construct a transparent state.

    Missing values are allowed because an unavailable component must propagate to
    an invalid composite.  Infinite values, booleans, and non-numeric component
    columns are rejected.  ``family`` adds economic-separation checks: stress
    terms cannot enter crowding, and placebo-adjusted terms cannot enter generic
    risk.

    Returns the validated component names as an immutable tuple.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    names = _validate_components(components)
    if require_unique_index and not frame.index.is_unique:
        raise ValueError("frame index must be unique")

    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"missing component columns: {missing}")

    for name in names:
        series = frame[name]
        if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
            raise TypeError(f"component '{name}' must be numeric and non-boolean")
        non_missing = series.dropna().to_numpy(dtype=float)
        if not np.isfinite(non_missing).all():
            raise ValueError(f"component '{name}' contains an infinite value")

    _validate_family_component_names(names, family)

    return names


def _normalized_weights(
    components: tuple[str, ...],
    weights: Mapping[str, float] | Sequence[float] | None,
) -> pd.Series:
    if weights is None:
        raw = np.ones(len(components), dtype=float)
    elif isinstance(weights, Mapping):
        missing = set(components) - set(weights)
        extra = set(weights) - set(components)
        if missing or extra:
            raise ValueError(
                "weight keys must exactly match components; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        raw = np.asarray([weights[name] for name in components], dtype=float)
    else:
        if isinstance(weights, (str, bytes)):
            raise TypeError("weights must be numeric")
        raw = np.asarray(tuple(weights), dtype=float)
        if raw.ndim != 1 or len(raw) != len(components):
            raise ValueError("weights must have one value per component")

    if not np.isfinite(raw).all():
        raise ValueError("weights must be finite")
    if (raw < 0).any():
        raise ValueError("weights must be non-negative")
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return pd.Series(raw / total, index=components, dtype=float)


def _complete_case_composite(
    frame: pd.DataFrame,
    components: Sequence[str],
    *,
    output_name: str,
    family: str,
    weights: Mapping[str, float] | Sequence[float] | None,
    column_map: Mapping[str, str] | None,
) -> pd.DataFrame:
    names = _validate_components(components)
    # Validate semantic names as well as mapped storage names so a column map
    # cannot accidentally route a turnover or liquidity field into C.
    _validate_family_component_names(names, family)
    if column_map is None:
        mapped_names = names
    else:
        extra = set(column_map) - set(names)
        if extra:
            raise ValueError(
                f"column_map contains unknown component keys: {sorted(extra)}"
            )
        mapped_names = tuple(column_map.get(name, name) for name in names)
        if any(not isinstance(name, str) or not name for name in mapped_names):
            raise TypeError("column_map values must be non-empty column names")
        if len(set(mapped_names)) != len(mapped_names):
            raise ValueError("column_map must map components to unique columns")

    validate_component_frame(frame, mapped_names, family=family)
    normalized_weights = _normalized_weights(names, weights)
    values = frame.loc[:, mapped_names].astype(float).copy()
    # Weight keys and economic component names remain stable even if storage
    # columns follow a configuration-specific naming convention.
    values.columns = names
    valid = values.notna().all(axis=1)
    component_count = values.notna().sum(axis=1).astype("int64")
    score = values.mul(normalized_weights, axis="columns").sum(axis=1)
    score = score.where(valid, np.nan)

    return pd.DataFrame(
        {
            output_name: score.astype(float),
            f"{output_name}_valid": valid.astype(bool),
            f"{output_name}_component_count": component_count,
        },
        index=frame.index,
    )


def compute_crowding_state(
    frame: pd.DataFrame,
    *,
    components: Sequence[str] = CROWDING_COMPONENTS,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build the structural Crowding State from historical-z excess features.

    The default is an equal-weight complete-case composite.  A missing component
    invalidates the row; weights are never silently renormalized around missing
    data.  Turnover, liquidity, and shock columns are explicitly rejected.
    """

    return _complete_case_composite(
        frame,
        components,
        output_name="crowding_state",
        family="crowding",
        weights=weights,
        column_map=column_map,
    )


def compute_generic_risk(
    frame: pd.DataFrame,
    *,
    components: Sequence[str] = GENERIC_RISK_COMPONENTS,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build a transparent generic portfolio-risk composite.

    Inputs must be comparably scaled raw portfolio characteristics.  Feature
    scaling itself belongs in the historical/fold-specific preprocessing layer;
    factor-specific placebo excess terms are rejected here.
    """

    return _complete_case_composite(
        frame,
        components,
        output_name="generic_risk",
        family="generic_risk",
        weights=weights,
        column_map=column_map,
    )


def compute_stress_trigger(
    frame: pd.DataFrame,
    *,
    components: Sequence[str] = STRESS_COMPONENTS,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build an unwind/stress trigger separately from structural crowding."""

    return _complete_case_composite(
        frame,
        components,
        output_name="stress_trigger",
        family="stress",
        weights=weights,
        column_map=column_map,
    )


def historical_zscore(
    values: pd.Series,
    *,
    lookback: int = 52,
    min_periods: int = 40,
    method: str = "mean_std",
    ddof: int = 1,
) -> pd.DataFrame:
    """Standardize each observation using prior observations only.

    ``lookback`` counts observations, not days.  For weekly decision data, the
    default 52 observations represents roughly one trading year; callers with an
    exchange calendar should first restrict the input to their exact 252-trading-
    day range.  The current value is excluded with an explicit one-row shift.

    The returned audit columns are ``historical_center``, ``historical_scale``,
    ``history_n``, and ``historical_z``.  A zero scale produces ``NaN`` rather
    than an arbitrary epsilon-adjusted score.
    """

    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series")
    if not values.index.is_unique:
        raise ValueError("values index must be unique")
    if not values.index.is_monotonic_increasing:
        raise ValueError("values must be ordered from oldest to newest")
    if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
        raise TypeError("values must be numeric and non-boolean")

    lookback = _validate_positive_integer(lookback, "lookback")
    min_periods = _validate_positive_integer(min_periods, "min_periods")
    if min_periods > lookback:
        raise ValueError("min_periods cannot exceed lookback")
    if isinstance(ddof, bool) or not isinstance(ddof, Integral) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    ddof = int(ddof)
    if min_periods <= ddof:
        raise ValueError("min_periods must exceed ddof")

    numeric = values.astype(float)
    non_missing = numeric.dropna().to_numpy(dtype=float)
    if not np.isfinite(non_missing).all():
        raise ValueError("values contains an infinite value")

    past = numeric.shift(1)
    rolling = past.rolling(window=lookback, min_periods=min_periods)
    normalized_method = method.strip().lower()
    if normalized_method == "mean_std":
        center = rolling.mean()
        scale = rolling.std(ddof=ddof)
    elif normalized_method == "median_mad":
        center = rolling.median()

        def scaled_mad(window_values: np.ndarray) -> float:
            median = float(np.nanmedian(window_values))
            return float(1.4826 * np.nanmedian(np.abs(window_values - median)))

        scale = rolling.apply(scaled_mad, raw=True)
    else:
        raise ValueError("method must be 'mean_std' or 'median_mad'")

    # ``Rolling.count`` inherits ``min_periods`` in pandas 3 and therefore
    # returns NaN during warm-up.  The audit count should still report how much
    # prior history was present, even when it is below the validity threshold.
    history_n = (
        past.rolling(window=lookback, min_periods=1).count().fillna(0).astype("int64")
    )
    valid_scale = scale.notna() & np.isfinite(scale) & (scale > 0)
    zscore = ((numeric - center) / scale).where(valid_scale, np.nan)

    return pd.DataFrame(
        {
            "value": numeric,
            "historical_center": center.astype(float),
            "historical_scale": scale.astype(float),
            "history_n": history_n,
            "historical_z": zscore.astype(float),
        },
        index=values.index,
    )


def historical_zscore_by_group(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_cols: Sequence[str] = ("factor", "leg"),
    order_col: str = "decision_at",
    lookback: int = 52,
    min_periods: int = 40,
    method: str = "mean_std",
    ddof: int = 1,
) -> pd.DataFrame:
    """Apply :func:`historical_zscore` independently within ordered groups.

    Results retain the input index and never borrow history across factors or
    legs.  Duplicate decision times within a group are rejected.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    groups = _validate_components(group_cols)
    required = (*groups, order_col, value_col)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame.loc[:, [*groups, order_col]].isna().any().any():
        raise ValueError("group and order columns cannot contain missing values")
    if is_bool_dtype(frame[value_col].dtype) or not is_numeric_dtype(
        frame[value_col].dtype
    ):
        raise TypeError(f"value column '{value_col}' must be numeric")

    output_columns = (
        "value",
        "historical_center",
        "historical_scale",
        "history_n",
        "historical_z",
    )
    result = pd.DataFrame(index=frame.index, columns=output_columns)
    ordered = frame.sort_values([*groups, order_col], kind="mergesort")
    grouper: str | list[str]
    grouper = groups[0] if len(groups) == 1 else list(groups)

    for _, group in ordered.groupby(grouper, sort=False, dropna=False):
        if group[order_col].duplicated().any():
            raise ValueError("decision times must be unique within each group")
        ordered_values = pd.Series(
            group[value_col].to_numpy(dtype=float),
            index=pd.Index(group[order_col], name=order_col),
            name=value_col,
        )
        standardized = historical_zscore(
            ordered_values,
            lookback=lookback,
            min_periods=min_periods,
            method=method,
            ddof=ddof,
        )
        standardized.index = group.index
        result.loc[group.index, output_columns] = standardized.loc[
            group.index, output_columns
        ]

    result["value"] = result["value"].astype(float)
    result["historical_center"] = result["historical_center"].astype(float)
    result["historical_scale"] = result["historical_scale"].astype(float)
    result["history_n"] = result["history_n"].astype("int64")
    result["historical_z"] = result["historical_z"].astype(float)
    return result


def historical_zscore_trading_window(
    values: pd.Series,
    trading_calendar: Sequence[object],
    *,
    lookback_sessions: int = 252,
    min_periods: int = 40,
    method: str = "mean_std",
    ddof: int = 1,
) -> pd.DataFrame:
    """Standardize weekly observations over an exact past trading-session range.

    ``values`` is indexed by decision session.  For a decision at calendar
    position ``p``, the baseline contains observations whose sessions fall in
    ``[p-lookback_sessions, p-1]``.  This differs deliberately from a rolling
    number of weekly rows and keeps the current observation out of the baseline.
    """

    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series")
    if not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError("values index must be unique and ordered oldest to newest")
    if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
        raise TypeError("values must be numeric and non-boolean")
    lookback_sessions = _validate_positive_integer(
        lookback_sessions, "lookback_sessions"
    )
    min_periods = _validate_positive_integer(min_periods, "min_periods")
    if isinstance(ddof, bool) or not isinstance(ddof, Integral) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    ddof = int(ddof)
    if min_periods <= ddof:
        raise ValueError("min_periods must exceed ddof")

    calendar = pd.DatetimeIndex(pd.to_datetime(list(trading_calendar))).normalize()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("trading_calendar must be nonempty, unique, and ordered")
    decisions = pd.DatetimeIndex(pd.to_datetime(values.index)).normalize()
    positions = calendar.get_indexer(decisions)
    if (positions < 0).any():
        absent = decisions[positions < 0]
        raise ValueError(
            "all decision dates must be trading sessions; "
            f"absent examples={list(absent[:3].date)}"
        )
    numeric = values.astype(float)
    finite = numeric.dropna().to_numpy(dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("values contains an infinite value")

    normalized_method = method.strip().lower()
    if normalized_method not in {"mean_std", "median_mad"}:
        raise ValueError("method must be 'mean_std' or 'median_mad'")
    records: list[dict[str, object]] = []
    numeric_array = numeric.to_numpy(dtype=float)
    for row_number, position in enumerate(positions):
        lower_position = max(0, int(position) - lookback_sessions)
        prior_mask = (
            (positions[:row_number] >= lower_position)
            & (positions[:row_number] < position)
        )
        history = numeric_array[:row_number][prior_mask]
        history = history[np.isfinite(history)]
        history_n = int(history.size)
        center = float("nan")
        scale = float("nan")
        if history_n >= min_periods:
            if normalized_method == "mean_std":
                center = float(np.mean(history))
                scale = float(np.std(history, ddof=ddof))
            else:
                center = float(np.median(history))
                scale = float(1.4826 * np.median(np.abs(history - center)))
        value = numeric_array[row_number]
        historical_z = (
            float((value - center) / scale)
            if np.isfinite(value) and np.isfinite(scale) and scale > 0
            else float("nan")
        )
        records.append(
            {
                "value": value,
                "historical_center": center,
                "historical_scale": scale,
                "history_n": history_n,
                "historical_z": historical_z,
                "history_window_start": calendar[lower_position],
                "history_window_end": (
                    calendar[position - 1] if position > 0 else pd.NaT
                ),
            }
        )
    return pd.DataFrame(records, index=values.index)


def historical_zscore_by_group_trading_window(
    frame: pd.DataFrame,
    trading_calendar: Sequence[object],
    *,
    value_col: str,
    group_cols: Sequence[str] = ("factor", "leg"),
    order_col: str = "decision_at",
    lookback_sessions: int = 252,
    min_periods: int = 40,
    method: str = "mean_std",
    ddof: int = 1,
) -> pd.DataFrame:
    """Apply the exact trading-session historical window within each group."""

    if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique:
        raise ValueError("frame must be a DataFrame with a unique index")
    groups = _validate_components(group_cols)
    required = [*groups, order_col, value_col]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    result: pd.DataFrame | None = None
    ordered = frame.sort_values([*groups, order_col], kind="mergesort")
    grouper: str | list[str] = groups[0] if len(groups) == 1 else list(groups)
    for _, group in ordered.groupby(grouper, sort=False, dropna=False):
        if group[order_col].duplicated().any():
            raise ValueError("decision times must be unique within each group")
        series = pd.Series(
            group[value_col].to_numpy(dtype=float),
            index=pd.DatetimeIndex(group[order_col]),
        )
        standardized = historical_zscore_trading_window(
            series,
            trading_calendar,
            lookback_sessions=lookback_sessions,
            min_periods=min_periods,
            method=method,
            ddof=ddof,
        )
        standardized.index = group.index
        if result is None:
            result = pd.DataFrame(index=frame.index, columns=standardized.columns)
        result.loc[group.index, standardized.columns] = standardized
    if result is None:
        return pd.DataFrame(index=frame.index)
    for column in ("value", "historical_center", "historical_scale", "historical_z"):
        result[column] = result[column].astype(float)
    result["history_n"] = result["history_n"].astype("int64")
    result["history_window_start"] = pd.to_datetime(result["history_window_start"])
    result["history_window_end"] = pd.to_datetime(result["history_window_end"])
    return result


def _coerce_actual(actual: float) -> float:
    if isinstance(actual, bool) or not isinstance(actual, Real):
        raise TypeError("actual must be a finite real number")
    actual_float = float(actual)
    if not np.isfinite(actual_float):
        raise ValueError("actual must be finite")
    return actual_float


def _coerce_placebo_values(placebo_values: Sequence[float]) -> np.ndarray:
    if isinstance(placebo_values, (str, bytes)):
        raise TypeError("placebo_values must be a one-dimensional numeric sequence")
    try:
        values = np.asarray(placebo_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("placebo_values must be numeric") from exc
    if values.ndim != 1 or values.size == 0:
        raise ValueError("placebo_values must be a non-empty one-dimensional sequence")
    if np.isinf(values).any():
        raise ValueError("placebo_values cannot contain infinite values")
    return values


def summarize_matched_placebo(
    actual: float,
    placebo_values: Sequence[float],
    *,
    min_valid_draws: int = 80,
    ddof: int = 1,
) -> PlaceboSummary:
    """Compare one actual feature value with matched placebo draws.

    NaN placebo draws are counted as invalid.  An insufficient draw count or a
    zero placebo standard deviation marks the summary invalid and leaves
    ``placebo_z`` missing; no epsilon is inserted.
    """

    actual_float = _coerce_actual(actual)
    draws = _coerce_placebo_values(placebo_values)
    min_valid_draws = _validate_positive_integer(min_valid_draws, "min_valid_draws")
    if min_valid_draws > draws.size:
        raise ValueError("min_valid_draws cannot exceed total placebo draws")
    if isinstance(ddof, bool) or not isinstance(ddof, Integral) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    ddof = int(ddof)

    valid_values = draws[~np.isnan(draws)]
    valid_draws = int(valid_values.size)
    placebo_mean = (
        float(np.mean(valid_values)) if valid_draws > 0 else float("nan")
    )
    placebo_std = (
        float(np.std(valid_values, ddof=ddof))
        if valid_draws > ddof
        else float("nan")
    )
    excess = (
        float(actual_float - placebo_mean)
        if np.isfinite(placebo_mean)
        else float("nan")
    )

    invalid_reason: str | None
    if valid_draws < min_valid_draws:
        invalid_reason = "insufficient_valid_draws"
    elif not np.isfinite(placebo_std) or placebo_std <= 0:
        invalid_reason = "zero_or_undefined_placebo_std"
    else:
        invalid_reason = None

    placebo_valid = invalid_reason is None
    placebo_z = (
        float(excess / placebo_std) if placebo_valid else float("nan")
    )
    return PlaceboSummary(
        actual=actual_float,
        placebo_mean=placebo_mean,
        placebo_std=placebo_std,
        excess=excess,
        placebo_z=placebo_z,
        valid_draws=valid_draws,
        total_draws=int(draws.size),
        placebo_valid=placebo_valid,
        invalid_reason=invalid_reason,
    )


def summarize_matched_placebos(
    actual: pd.Series,
    placebo_draws: pd.DataFrame,
    *,
    min_valid_draws: int = 80,
    ddof: int = 1,
) -> pd.DataFrame:
    """Vectorized row-wise matched-placebo summaries with strict alignment."""

    if not isinstance(actual, pd.Series):
        raise TypeError("actual must be a pandas Series")
    if not isinstance(placebo_draws, pd.DataFrame):
        raise TypeError("placebo_draws must be a pandas DataFrame")
    if not actual.index.is_unique or not placebo_draws.index.is_unique:
        raise ValueError("actual and placebo indexes must be unique")
    if placebo_draws.columns.duplicated().any():
        raise ValueError("placebo draw identifiers must be unique")
    if not actual.index.equals(placebo_draws.index):
        raise ValueError("actual and placebo_draws indexes must match exactly")
    if len(actual) == 0 or placebo_draws.shape[1] == 0:
        raise ValueError("actual and placebo_draws must not be empty")
    if is_bool_dtype(actual.dtype) or not is_numeric_dtype(actual.dtype):
        raise TypeError("actual must be numeric and non-boolean")
    for column in placebo_draws.columns:
        series = placebo_draws[column]
        if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
            raise TypeError(f"placebo draw '{column}' must be numeric")

    actual_values = actual.astype(float).to_numpy()
    if np.isnan(actual_values).any() or np.isinf(actual_values).any():
        raise ValueError("actual values must be finite")
    placebo_values = placebo_draws.astype(float).to_numpy()
    if np.isinf(placebo_values).any():
        raise ValueError("placebo draws cannot contain infinite values")

    summaries = [
        summarize_matched_placebo(
            actual_value,
            placebo_values[row_number],
            min_valid_draws=min_valid_draws,
            ddof=ddof,
        ).to_dict()
        for row_number, actual_value in enumerate(actual_values)
    ]
    return pd.DataFrame(summaries, index=actual.index)
