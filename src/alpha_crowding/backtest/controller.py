"""Controller policies that turn strictly historical risk ranks into exposure.

The functions in this module deliberately do not inspect timestamps or model
features.  Their input is already-out-of-sample model output in chronological
order.  Keeping the policy small makes the no-look-ahead boundary easy to
audit.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd


# ``percentile < upper_bound`` selects a band.  The final bound includes 1.0.
DEFAULT_RISK_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.60, 1.00),
    (0.80, 0.75),
    (0.90, 0.50),
    (1.00, 0.25),
)


def _as_probability(value: float, *, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _validate_bands(
    bands: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    if not bands:
        raise ValueError("bands must not be empty")

    validated = []
    previous_bound = 0.0
    for upper_bound, active_weight in bands:
        bound = _as_probability(upper_bound, name="band upper bound")
        weight = _as_probability(active_weight, name="active weight")
        if bound <= previous_bound:
            raise ValueError("band upper bounds must be strictly increasing")
        validated.append((bound, weight))
        previous_bound = bound

    if validated[-1][0] != 1.0:
        raise ValueError("the final band upper bound must be 1.0")
    return tuple(validated)


def risk_percentile_to_active_weight(
    risk_percentile: float,
    *,
    bands: Sequence[Tuple[float, float]] = DEFAULT_RISK_BANDS,
) -> float:
    """Map a historical risk percentile to the frozen V2 exposure tiers.

    The intervals are ``[0, 0.60)``, ``[0.60, 0.80)``,
    ``[0.80, 0.90)``, and ``[0.90, 1]`` under the default policy.  Boundary
    handling is explicit because the design says *below* the 60th percentile.
    """

    percentile = _as_probability(risk_percentile, name="risk_percentile")
    validated_bands = _validate_bands(bands)
    for index, (upper_bound, active_weight) in enumerate(validated_bands):
        is_last = index == len(validated_bands) - 1
        if percentile < upper_bound or (is_last and percentile <= upper_bound):
            return active_weight
    raise AssertionError("validated bands did not cover the percentile")


def historical_risk_percentile(
    current_probability: float,
    past_probabilities: Iterable[float],
    *,
    min_history: int = 1,
) -> Optional[float]:
    """Return the current probability's midrank among prior predictions only.

    A midrank empirical percentile is used so a flat sequence of identical
    probabilities maps to 0.5 rather than spuriously entering the highest-risk
    tier.  ``None`` marks an honest warm-up period with insufficient history.
    """

    if isinstance(min_history, bool) or not isinstance(min_history, int):
        raise TypeError("min_history must be an integer")
    if min_history < 1:
        raise ValueError("min_history must be at least 1")

    current = _as_probability(current_probability, name="current_probability")
    history = [
        _as_probability(value, name="past probability")
        for value in past_probabilities
    ]
    if len(history) < min_history:
        return None

    less = sum(value < current for value in history)
    equal = sum(value == current for value in history)
    return (less + 0.5 * equal) / len(history)


def active_weights_from_oos_probabilities(
    probabilities: Iterable[float],
    *,
    min_history: int,
    bands: Sequence[Tuple[float, float]] = DEFAULT_RISK_BANDS,
) -> Tuple[Optional[float], ...]:
    """Produce chronological weights without ever ranking on future values.

    The probability at position ``t`` is ranked only against positions before
    ``t``.  The current observation is appended to history after its weight is
    decided.  Warm-up observations return ``None`` and remain in the history.
    """

    values = [
        _as_probability(value, name="probability") for value in probabilities
    ]
    result = []
    history = []
    for probability in values:
        percentile = historical_risk_percentile(
            probability, history, min_history=min_history
        )
        result.append(
            None
            if percentile is None
            else risk_percentile_to_active_weight(percentile, bands=bands)
        )
        history.append(probability)
    return tuple(result)


def probability_to_active_weight(
    probability: float,
    *,
    minimum_weight: float = 0.25,
    maximum_weight: float = 1.0,
) -> float:
    """Implement the report's ``clip(1 - p, 0.25, 1)`` sensitivity policy."""

    risk = _as_probability(probability, name="probability")
    lower = _as_probability(minimum_weight, name="minimum_weight")
    upper = _as_probability(maximum_weight, name="maximum_weight")
    if lower > upper:
        raise ValueError("minimum_weight must not exceed maximum_weight")
    return min(upper, max(lower, 1.0 - risk))


def build_volatility_control_weights(
    decisions: pd.DataFrame,
    active_returns: pd.DataFrame,
    *,
    lookback_sessions: int,
    minimum_observations: int,
    annualized_target_volatility: float,
    minimum_active_weight: float = 0.25,
    maximum_active_weight: float = 1.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build close-known volatility-control weights from trailing active returns."""

    decision_required = {"decision_at", "factor"}
    return_required = {"date", "factor", "active_return"}
    for name, frame, required in (
        ("decision", decisions, decision_required),
        ("active return", active_returns, return_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} fields: {sorted(missing)}")
    if (
        isinstance(lookback_sessions, bool)
        or isinstance(minimum_observations, bool)
        or lookback_sessions < 2
        or minimum_observations < 2
        or minimum_observations > lookback_sessions
    ):
        raise ValueError("volatility window and minimum observations are invalid")
    if isinstance(periods_per_year, bool) or periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    target = float(annualized_target_volatility)
    lower = _as_probability(minimum_active_weight, name="minimum_active_weight")
    upper = _as_probability(maximum_active_weight, name="maximum_active_weight")
    if not isfinite(target) or target <= 0.0:
        raise ValueError("annualized_target_volatility must be finite and positive")
    if lower > upper:
        raise ValueError("minimum_active_weight must not exceed maximum_active_weight")

    dates = decisions[["decision_at", "factor"]].copy()
    dates["decision_at"] = pd.to_datetime(
        dates["decision_at"], errors="raise"
    ).dt.normalize()
    dates["factor"] = dates["factor"].astype(str)
    if dates.duplicated(["decision_at", "factor"]).any():
        raise ValueError("decisions must be unique by date/factor")
    returns = active_returns[["date", "factor", "active_return"]].copy()
    returns["date"] = pd.to_datetime(returns["date"], errors="raise").dt.normalize()
    returns["factor"] = returns["factor"].astype(str)
    returns["active_return"] = pd.to_numeric(
        returns["active_return"], errors="coerce"
    )
    if returns.duplicated(["date", "factor"]).any():
        raise ValueError("active returns must be unique by date/factor")
    histories = {
        factor: group.sort_values("date").reset_index(drop=True)
        for factor, group in returns.groupby("factor", sort=False)
    }
    rows = []
    for current in dates.itertuples(index=False):
        history = histories.get(current.factor)
        if history is None:
            window = pd.DataFrame(columns=returns.columns)
        else:
            window = history.loc[history["date"].le(current.decision_at)].tail(
                lookback_sessions
            )
        complete = (
            len(window) >= minimum_observations
            and window["active_return"].notna().all()
        )
        if complete:
            volatility = float(window["active_return"].std(ddof=1)) * sqrt(
                periods_per_year
            )
            weight = upper if volatility == 0.0 else target / volatility
            weight = min(upper, max(lower, weight))
        else:
            volatility = None
            weight = None
        rows.append(
            {
                "decision_at": current.decision_at,
                "factor": current.factor,
                "active_weight_volatility_control": weight,
                "volatility_control_estimate": volatility,
                "volatility_control_observations": len(window),
                "volatility_control_window_start": (
                    window["date"].min() if len(window) else pd.NaT
                ),
                "volatility_control_window_end": (
                    window["date"].max() if len(window) else pd.NaT
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision_at", "factor"]
    ).reset_index(drop=True)


def add_expost_mean_exposure_diagnostic(
    schedule: pd.DataFrame,
    *,
    source_column: str = "active_weight_M3",
    output_column: str = "active_weight_M3_expost_mean",
) -> pd.DataFrame:
    """Add an explicitly non-executable full-period mean-exposure diagnostic."""

    required = {"decision_at", "factor", source_column}
    missing = required - set(schedule.columns)
    if missing:
        raise KeyError(f"ex-post exposure diagnostic lacks fields: {sorted(missing)}")
    if output_column in schedule.columns:
        raise ValueError(f"output column already exists: {output_column}")
    result = schedule.copy()
    result[source_column] = pd.to_numeric(result[source_column], errors="coerce")
    if result[source_column].isna().any():
        raise ValueError("ex-post exposure diagnostic requires a complete common sample")
    if not result[source_column].between(0.0, 1.0).all():
        raise ValueError("source exposure must be in [0, 1]")
    result[output_column] = result.groupby("factor", sort=False)[
        source_column
    ].transform("mean")
    return result


def build_oos_exposure_schedule(
    predictions: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    *,
    probability_columns: Sequence[str],
    minimum_history_weeks: int,
    bands: Sequence[Tuple[float, float]] = DEFAULT_RISK_BANDS,
    date_col: str = "decision_at",
    factor_col: str = "factor",
) -> pd.DataFrame:
    """Build next-session exposures from strictly earlier same-factor OOS history."""

    columns = tuple(probability_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("probability_columns must be nonempty and unique")
    if minimum_history_weeks < 1:
        raise ValueError("minimum_history_weeks must be positive")
    required = {date_col, factor_col, *columns}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"missing exposure-schedule columns: {sorted(missing)}")
    frame = predictions[[date_col, factor_col, *columns]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise").dt.normalize()
    if frame.duplicated([date_col, factor_col]).any():
        raise ValueError("predictions must be unique by decision date and factor")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    calendar = calendar.drop_duplicates().sort_values()
    locations = calendar.get_indexer(frame[date_col])
    if (locations < 0).any() or (locations + 1 >= len(calendar)).any():
        raise ValueError("every decision needs a next trading session")
    frame["effective_at"] = calendar[locations + 1]

    rows = []
    for factor, group in frame.groupby(factor_col, sort=True):
        ordered = group.sort_values(date_col)
        histories = {column: [] for column in columns}
        for _, current in ordered.iterrows():
            row = {
                "decision_at": current[date_col],
                "effective_at": current["effective_at"],
                "factor": factor,
            }
            for column in columns:
                probability = _as_probability(current[column], name=column)
                percentile = historical_risk_percentile(
                    probability,
                    histories[column],
                    min_history=minimum_history_weeks,
                )
                model = column.removeprefix("probability_")
                row[f"probability_{model}"] = probability
                row[f"risk_percentile_{model}"] = percentile
                row[f"active_weight_{model}"] = (
                    None
                    if percentile is None
                    else risk_percentile_to_active_weight(percentile, bands=bands)
                )
                histories[column].append(probability)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["decision_at", "factor"]).reset_index(drop=True)
