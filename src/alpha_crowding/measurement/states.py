"""Causal trailing market and factor return-state primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_return_state(
    returns: pd.Series,
    trading_calendar: pd.DatetimeIndex,
    *,
    decision_at: object,
    return_sessions: int = 20,
    volatility_short_sessions: int = 20,
    volatility_long_sessions: int = 60,
    drawdown_sessions: int = 252,
) -> dict[str, object]:
    """Compute trailing return, volatilities, and current drawdown through t."""

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    decision = pd.Timestamp(decision_at).normalize()
    position = int(calendar.get_indexer([decision])[0])
    if position < 0:
        raise ValueError("decision_at must be a trading session")
    series = pd.Series(
        pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(returns.index)).normalize(),
    )
    if not series.index.is_unique:
        raise ValueError("returns must be unique by session")
    aligned = series.reindex(calendar[: position + 1])

    def tail(count: int) -> pd.Series:
        if count < 1:
            raise ValueError("state windows must be positive")
        return aligned.iloc[-count:] if len(aligned) >= count else pd.Series(dtype=float)

    return_window = tail(return_sessions)
    short_window = tail(volatility_short_sessions)
    long_window = tail(volatility_long_sessions)
    drawdown_window = tail(drawdown_sessions)
    trailing_return = (
        float(np.prod(1.0 + return_window.to_numpy()) - 1.0)
        if len(return_window) == return_sessions and return_window.notna().all()
        else float("nan")
    )
    short_volatility = (
        float(short_window.std(ddof=1))
        if len(short_window) == volatility_short_sessions and short_window.notna().all()
        else float("nan")
    )
    long_volatility = (
        float(long_window.std(ddof=1))
        if len(long_window) == volatility_long_sessions and long_window.notna().all()
        else float("nan")
    )
    current_drawdown = float("nan")
    if len(drawdown_window) == drawdown_sessions and drawdown_window.notna().all():
        wealth = np.cumprod(1.0 + drawdown_window.to_numpy(dtype=float))
        current_drawdown = float(wealth[-1] / max(1.0, float(wealth.max())) - 1.0)
    return {
        "decision_at": decision,
        "trailing_return": trailing_return,
        "volatility_short": short_volatility,
        "volatility_long": long_volatility,
        "current_drawdown": current_drawdown,
        "return_window_start": return_window.index.min() if len(return_window) else pd.NaT,
        "volatility_long_window_start": long_window.index.min() if len(long_window) else pd.NaT,
        "drawdown_window_start": drawdown_window.index.min() if len(drawdown_window) else pd.NaT,
        "window_end": decision,
    }


def causal_rank_ic_state(
    rank_ic: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    *,
    lookback_observations: int = 52,
    minimum_observations: int = 26,
) -> pd.DataFrame:
    """Aggregate only RankIC observations realized by each decision date."""

    required = {"signal_at", "realized_at", "rank_ic"}
    missing = required - set(rank_ic.columns)
    if missing:
        raise KeyError(f"missing RankIC fields: {sorted(missing)}")
    observed = rank_ic.copy()
    observed["realized_at"] = pd.to_datetime(observed["realized_at"]).dt.normalize()
    observed["rank_ic"] = pd.to_numeric(observed["rank_ic"], errors="coerce")
    rows = []
    for decision in pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize():
        history = observed.loc[
            observed["realized_at"].le(decision) & observed["rank_ic"].notna()
        ].sort_values("realized_at").tail(lookback_observations)
        valid = len(history) >= minimum_observations
        rows.append(
            {
                "decision_at": decision,
                "rank_ic_mean": float(history["rank_ic"].mean()) if valid else np.nan,
                "rank_ic_volatility": float(history["rank_ic"].std(ddof=1)) if valid else np.nan,
                "rank_ic_history_n": len(history),
                "rank_ic_latest_realized_at": history["realized_at"].max() if len(history) else pd.NaT,
            }
        )
    return pd.DataFrame(rows)
