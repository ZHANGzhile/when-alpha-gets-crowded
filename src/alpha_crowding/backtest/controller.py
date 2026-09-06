"""Controller policies that turn strictly historical risk ranks into exposure.

The functions in this module deliberately do not inspect timestamps or model
features.  Their input is already-out-of-sample model output in chronological
order.  Keeping the policy small makes the no-look-ahead boundary easy to
audit.
"""

from __future__ import annotations

from math import isfinite
from typing import Iterable, Optional, Sequence, Tuple


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
