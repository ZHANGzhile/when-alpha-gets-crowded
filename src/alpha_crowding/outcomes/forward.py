"""Exact-session forward outcomes for research and active targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .active import active_mdd, relative_nav


def _maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    return float(-np.min(wealth / peaks - 1.0))


def forward_window_outcome(
    portfolio_returns: pd.Series,
    trading_calendar: pd.DatetimeIndex,
    *,
    decision_at: object,
    horizon_sessions: int,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, object]:
    """Compute a future return/MDD over the next exact trading sessions."""

    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be positive")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    decision = pd.Timestamp(decision_at).normalize()
    position = int(calendar.get_indexer([decision])[0])
    if position < 0:
        raise ValueError("decision_at must be a trading session")
    label_start = calendar[position + 1] if position + 1 < len(calendar) else pd.NaT
    if position + horizon_sessions >= len(calendar):
        return {
            "decision_at": decision,
            "label_start_at": label_start,
            "label_end_at": pd.NaT,
            "future_return": np.nan,
            "future_mdd": np.nan,
            "horizon_sessions": horizon_sessions,
            "outcome_mature": False,
            "invalid_reason": "horizon_beyond_raw_cutoff",
        }
    dates = calendar[position + 1 : position + horizon_sessions + 1]
    returns = pd.Series(
        pd.to_numeric(portfolio_returns, errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(portfolio_returns.index)).normalize(),
    ).reindex(dates)
    if returns.isna().any():
        return {
            "decision_at": decision,
            "label_start_at": dates.min(),
            "label_end_at": dates.max(),
            "future_return": np.nan,
            "future_mdd": np.nan,
            "horizon_sessions": horizon_sessions,
            "outcome_mature": False,
            "invalid_reason": "missing_portfolio_return",
        }
    if benchmark_returns is None:
        values = returns.to_numpy(dtype=float)
        future_return = float(np.prod(1.0 + values) - 1.0)
        future_mdd = _maximum_drawdown(values)
    else:
        benchmark = pd.Series(
            pd.to_numeric(benchmark_returns, errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(benchmark_returns.index)).normalize(),
        ).reindex(dates)
        if benchmark.isna().any():
            return {
                "decision_at": decision,
                "label_start_at": dates.min(),
                "label_end_at": dates.max(),
                "future_return": np.nan,
                "future_mdd": np.nan,
                "horizon_sessions": horizon_sessions,
                "outcome_mature": False,
                "invalid_reason": "missing_benchmark_return",
            }
        relative = np.asarray(relative_nav(returns, benchmark), dtype=float)
        future_return = float(relative[-1] - 1.0)
        future_mdd = active_mdd(returns, benchmark)
    return {
        "decision_at": decision,
        "label_start_at": dates.min(),
        "label_end_at": dates.max(),
        "future_return": future_return,
        "future_mdd": future_mdd,
        "horizon_sessions": horizon_sessions,
        "outcome_mature": True,
        "invalid_reason": None,
    }
