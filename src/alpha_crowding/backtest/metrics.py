"""Relative-to-benchmark paths and evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite, nan, sqrt
from statistics import mean, stdev
from typing import Iterable, Tuple

import pandas as pd


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
    return tuple(
        long_return - benchmark_return
        for long_return, benchmark_return in zip(factor, benchmark)
    )


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
class AbsolutePerformance:
    observations: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    worst_horizon_return: float


def evaluate_absolute_performance(
    returns: Iterable[float],
    *,
    periods_per_year: int = 252,
    worst_horizon_sessions: int = 20,
) -> AbsolutePerformance:
    """Compute the frozen absolute metrics from a daily net-return path."""

    values = tuple(float(value) for value in returns)
    if not values:
        raise ValueError("return series must not be empty")
    if any(not isfinite(value) or value <= -1.0 for value in values):
        raise ValueError("returns must be finite and greater than -1")
    if isinstance(periods_per_year, bool) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if (
        isinstance(worst_horizon_sessions, bool)
        or not isinstance(worst_horizon_sessions, int)
        or worst_horizon_sessions < 1
    ):
        raise ValueError("worst_horizon_sessions must be a positive integer")

    nav = [1.0]
    for value in values:
        nav.append(nav[-1] * (1.0 + value))
    observations = len(values)
    total_return = nav[-1] - 1.0
    annualized_return = nav[-1] ** (periods_per_year / observations) - 1.0
    annualized_volatility = (
        stdev(values) * sqrt(periods_per_year) if observations >= 2 else nan
    )
    sharpe = (
        mean(values) * periods_per_year / annualized_volatility
        if annualized_volatility > 0.0
        else nan
    )
    downside_deviation = sqrt(
        mean(min(value, 0.0) ** 2 for value in values) * periods_per_year
    )
    sortino = (
        mean(values) * periods_per_year / downside_deviation
        if downside_deviation > 0.0
        else nan
    )
    if observations < worst_horizon_sessions:
        worst_horizon = nan
    else:
        horizon_returns = []
        for start in range(observations - worst_horizon_sessions + 1):
            wealth = 1.0
            for value in values[start : start + worst_horizon_sessions]:
                wealth *= 1.0 + value
            horizon_returns.append(wealth - 1.0)
        worst_horizon = min(horizon_returns)
    return AbsolutePerformance(
        observations=observations,
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown(nav),
        worst_horizon_return=worst_horizon,
    )


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


def summarize_controller_performance(
    portfolio_paths: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    execution_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize absolute, active, exposure, turnover, and rejection metrics."""

    path_required = {
        "date",
        "factor",
        "policy",
        "cost_bps",
        "net_return",
        "active_weight",
        "half_l1_turnover",
        "transaction_cost",
    }
    benchmark_required = {"date", "daily_return"}
    ledger_required = {"factor", "policy", "cost_bps", "reject_reason"}
    for name, frame, required in (
        ("portfolio path", portfolio_paths, path_required),
        ("benchmark", benchmark_returns, benchmark_required),
        ("execution ledger", execution_ledger, ledger_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} fields: {sorted(missing)}")
    paths = portfolio_paths.copy()
    paths["date"] = pd.to_datetime(paths["date"], errors="raise").dt.normalize()
    group_key = ["factor", "policy", "cost_bps"]
    if paths.duplicated(["date", *group_key]).any():
        raise ValueError("portfolio paths contain duplicate group dates")
    benchmark = benchmark_returns[["date", "daily_return"]].copy()
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="raise"
    ).dt.normalize()
    if benchmark["date"].duplicated().any():
        raise ValueError("benchmark returns contain duplicate dates")
    benchmark["daily_return"] = pd.to_numeric(
        benchmark["daily_return"], errors="coerce"
    )

    rows = []
    for keys, group in paths.groupby(group_key, sort=True):
        aligned = group.merge(
            benchmark,
            on="date",
            how="left",
            validate="one_to_one",
        )
        if aligned["daily_return"].isna().any():
            raise ValueError("controller path lacks aligned benchmark returns")
        absolute = evaluate_absolute_performance(aligned["net_return"])
        relative = evaluate_relative_performance(
            aligned["net_return"], aligned["daily_return"]
        )
        factor, policy, cost_bps = keys
        ledger_mask = (
            execution_ledger["factor"].eq(factor)
            & execution_ledger["policy"].eq(policy)
            & execution_ledger["cost_bps"].eq(cost_bps)
        )
        rejected = execution_ledger.loc[ledger_mask, "reject_reason"].fillna("").ne("")
        rows.append(
            {
                "factor": factor,
                "policy": policy,
                "cost_bps": cost_bps,
                **asdict(absolute),
                **asdict(relative),
                "mean_active_weight": float(aligned["active_weight"].mean()),
                "active_weight_q25": float(aligned["active_weight"].quantile(0.25)),
                "active_weight_median": float(aligned["active_weight"].median()),
                "active_weight_q75": float(aligned["active_weight"].quantile(0.75)),
                "mean_daily_half_l1_turnover": float(
                    aligned["half_l1_turnover"].mean()
                ),
                "total_half_l1_turnover": float(
                    aligned["half_l1_turnover"].sum()
                ),
                "total_transaction_cost": float(aligned["transaction_cost"].sum()),
                "rejected_ledger_rows": int(rejected.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_key).reset_index(drop=True)
