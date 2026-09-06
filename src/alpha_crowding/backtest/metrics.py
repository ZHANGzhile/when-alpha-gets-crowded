"""Relative-to-benchmark paths and evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, nan, sqrt
from statistics import mean, stdev
from typing import Iterable, Tuple


def _return_pair(
    factor_long_returns: Iterable[float],
    benchmark_returns: Iterable[float],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    factor = tuple(float(value) for value in factor_long_returns)
    benchmark = tuple(float(value) for value in benchmark_returns)
    if len(factor) != len(benchmark):
        raise ValueError("factor and benchmark returns must have equal length")
    if not factor:
        raise ValueError("return series must not be empty")
    for value in factor + benchmark:
        if not isfinite(value):
            raise ValueError("returns must be finite")
        if value <= -1.0:
            raise ValueError("returns must be greater than -1")
    return factor, benchmark


def arithmetic_active_returns(
    factor_long_returns: Iterable[float],
    benchmark_returns: Iterable[float],
) -> Tuple[float, ...]:
    """Return daily ``FactorLong - Benchmark`` for TE and IR calculations."""

    factor, benchmark = _return_pair(factor_long_returns, benchmark_returns)
    return tuple(long_return - benchmark_return for long_return, benchmark_return in zip(factor, benchmark))


def relative_nav(
    factor_long_returns: Iterable[float],
    benchmark_returns: Iterable[float],
    *,
    initial_value: float = 1.0,
) -> Tuple[float, ...]:
    """Build ``wealth(FactorLong) / wealth(Benchmark)``, including time zero."""

    factor, benchmark = _return_pair(factor_long_returns, benchmark_returns)
    start = float(initial_value)
    if not isfinite(start) or start <= 0.0:
        raise ValueError("initial_value must be finite and positive")

    path = [start]
    value = start
    for long_return, benchmark_return in zip(factor, benchmark):
        value *= (1.0 + long_return) / (1.0 + benchmark_return)
        path.append(value)
    return tuple(path)


def max_drawdown(nav: Iterable[float]) -> float:
    """Return a non-negative maximum drawdown; the supplied initial NAV counts."""

    values = tuple(float(value) for value in nav)
    if not values:
        raise ValueError("nav must not be empty")
    if any(not isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("nav values must be finite and positive")

    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, 1.0 - value / peak)
    return worst


def tracking_error(
    active_returns: Iterable[float],
    *,
    periods_per_year: int = 252,
) -> float:
    """Annualized sample standard deviation of arithmetic active returns."""

    values = tuple(float(value) for value in active_returns)
    if len(values) < 2:
        raise ValueError("at least two active returns are required")
    if any(not isfinite(value) for value in values):
        raise ValueError("active returns must be finite")
    if isinstance(periods_per_year, bool) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return stdev(values) * sqrt(periods_per_year)


def information_ratio(
    active_returns: Iterable[float],
    *,
    periods_per_year: int = 252,
) -> float:
    """Annualized arithmetic active return divided by tracking error."""

    values = tuple(float(value) for value in active_returns)
    te = tracking_error(values, periods_per_year=periods_per_year)
    if te == 0.0:
        return nan
    return mean(values) * periods_per_year / te


def active_cvar(active_returns: Iterable[float], *, alpha: float = 0.05) -> float:
    """Return positive expected shortfall of the worst ``alpha`` active returns."""

    values = tuple(float(value) for value in active_returns)
    if not values:
        raise ValueError("active returns must not be empty")
    if any(not isfinite(value) for value in values):
        raise ValueError("active returns must be finite")
    tail_probability = float(alpha)
    if not isfinite(tail_probability) or not 0.0 < tail_probability <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    tail_size = max(1, ceil(len(values) * tail_probability))
    tail_mean = mean(sorted(values)[:tail_size])
    return max(0.0, -tail_mean)


@dataclass(frozen=True)
class RelativePerformance:
    observations: int
    relative_total_return: float
    annualized_relative_return: float
    tracking_error: float
    information_ratio: float
    active_max_drawdown: float
    active_cvar: float


def evaluate_relative_performance(
    factor_long_returns: Iterable[float],
    benchmark_returns: Iterable[float],
    *,
    periods_per_year: int = 252,
    cvar_alpha: float = 0.05,
) -> RelativePerformance:
    """Compute the V2 core active metrics from aligned daily return series."""

    factor, benchmark = _return_pair(factor_long_returns, benchmark_returns)
    active = arithmetic_active_returns(factor, benchmark)
    path = relative_nav(factor, benchmark)
    observations = len(factor)
    relative_total = path[-1] - 1.0
    annualized_relative = path[-1] ** (periods_per_year / observations) - 1.0
    return RelativePerformance(
        observations=observations,
        relative_total_return=relative_total,
        annualized_relative_return=annualized_relative,
        tracking_error=tracking_error(active, periods_per_year=periods_per_year),
        information_ratio=information_ratio(
            active, periods_per_year=periods_per_year
        ),
        active_max_drawdown=max_drawdown(path),
        active_cvar=active_cvar(active, alpha=cvar_alpha),
    )
