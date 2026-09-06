"""Active-performance measures based on relative wealth."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


ReturnVector = pd.Series | Sequence[float] | np.ndarray


def _validated_return_pair(
    portfolio_returns: ReturnVector,
    benchmark_returns: ReturnVector,
) -> tuple[np.ndarray, np.ndarray, pd.Index | None]:
    if isinstance(portfolio_returns, pd.Series) and isinstance(
        benchmark_returns, pd.Series
    ):
        if not portfolio_returns.index.equals(benchmark_returns.index):
            raise ValueError("portfolio and benchmark Series must have identical indices")
        index: pd.Index | None = portfolio_returns.index
    elif isinstance(portfolio_returns, pd.Series):
        index = portfolio_returns.index
    elif isinstance(benchmark_returns, pd.Series):
        index = benchmark_returns.index
    else:
        index = None

    portfolio = np.asarray(portfolio_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if portfolio.ndim != 1 or benchmark.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if portfolio.shape != benchmark.shape:
        raise ValueError("portfolio and benchmark returns must have equal lengths")
    if not np.isfinite(portfolio).all() or not np.isfinite(benchmark).all():
        raise ValueError("returns must be finite; handle missing observations explicitly")
    if (portfolio < -1.0).any():
        raise ValueError("portfolio returns cannot be below -100%")
    if (benchmark <= -1.0).any():
        raise ValueError("benchmark returns must be greater than -100%")
    return portfolio, benchmark, index


def _restore_type(values: np.ndarray, index: pd.Index | None, name: str) -> ReturnVector:
    if index is None:
        return values
    return pd.Series(values, index=index, name=name)


def relative_nav(
    portfolio_returns: ReturnVector,
    benchmark_returns: ReturnVector,
    *,
    initial_value: float = 1.0,
) -> ReturnVector:
    """Return portfolio wealth divided by benchmark wealth.

    This is deliberately not ``cumprod(1 + portfolio - benchmark)``.  Relative
    wealth is the ratio of the two independently compounded NAV series.
    """

    if not np.isfinite(initial_value) or initial_value <= 0.0:
        raise ValueError("initial_value must be finite and positive")
    portfolio, benchmark, index = _validated_return_pair(
        portfolio_returns, benchmark_returns
    )
    portfolio_nav = np.cumprod(1.0 + portfolio)
    benchmark_nav = np.cumprod(1.0 + benchmark)
    values = initial_value * portfolio_nav / benchmark_nav
    return _restore_type(values, index, "relative_nav")


def active_drawdown_series(
    portfolio_returns: ReturnVector,
    benchmark_returns: ReturnVector,
    *,
    initial_value: float = 1.0,
) -> ReturnVector:
    """Return the drawdown series of relative NAV.

    The initial relative NAV is included as a peak, so a loss on the first day
    is represented correctly.  Values are zero at a high-water mark and
    negative during a drawdown.
    """

    relative = relative_nav(
        portfolio_returns, benchmark_returns, initial_value=initial_value
    )
    index = relative.index if isinstance(relative, pd.Series) else None
    values = np.asarray(relative, dtype=float)
    if len(values) == 0:
        return _restore_type(values.copy(), index, "active_drawdown")
    running_peak = np.maximum.accumulate(
        np.concatenate(([float(initial_value)], values))
    )[1:]
    drawdown = values / running_peak - 1.0
    return _restore_type(drawdown, index, "active_drawdown")


def active_max_drawdown(
    portfolio_returns: ReturnVector,
    benchmark_returns: ReturnVector,
    *,
    initial_value: float = 1.0,
) -> float:
    """Return active maximum drawdown as a non-negative loss magnitude."""

    drawdown = np.asarray(
        active_drawdown_series(
            portfolio_returns,
            benchmark_returns,
            initial_value=initial_value,
        ),
        dtype=float,
    )
    if len(drawdown) == 0:
        return 0.0
    return float(-np.min(drawdown))


def active_mdd(
    portfolio_returns: ReturnVector,
    benchmark_returns: ReturnVector,
    *,
    initial_value: float = 1.0,
) -> float:
    """Alias with the terminology used by the research protocol."""

    return active_max_drawdown(
        portfolio_returns, benchmark_returns, initial_value=initial_value
    )
