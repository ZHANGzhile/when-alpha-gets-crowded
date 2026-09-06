"""Portfolio-level generic risk and short-horizon stress characteristics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _raw_mad_z(current: float, baseline: pd.Series, minimum: int) -> float:
    history = pd.to_numeric(baseline, errors="coerce").dropna().to_numpy(dtype=float)
    if history.size < minimum or not np.isfinite(current):
        return float("nan")
    center = float(np.median(history))
    scale = float(np.median(np.abs(history - center)))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float((current - center) / scale)


def portfolio_risk_characteristics(
    daily: pd.DataFrame,
    actual_codes: list[str],
    trading_calendar: pd.DatetimeIndex,
    *,
    decision_at: object,
    turnover_recent_sessions: int = 5,
    turnover_baseline_sessions: int = 60,
    turnover_recent_minimum: int = 3,
    turnover_baseline_minimum: int = 40,
    turnover_sync_threshold: float = 1.5,
    liquidity_recent_sessions: int = 20,
    liquidity_baseline_sessions: int = 252,
    liquidity_recent_minimum: int = 15,
    liquidity_baseline_minimum: int = 126,
) -> dict[str, object]:
    """Compute G levels and S innovations with non-overlapping baselines."""

    required = {"date", "code", "turn", "daily_illiquidity"}
    missing = required - set(daily.columns)
    if missing:
        raise KeyError(f"missing portfolio-risk fields: {sorted(missing)}")
    codes = sorted(set(str(code) for code in actual_codes))
    if not codes:
        raise ValueError("actual_codes must not be empty")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    decision = pd.Timestamp(decision_at).normalize()
    position = int(calendar.get_indexer([decision])[0])
    if position < 0:
        raise ValueError("decision_at must be a trading session")
    needed = max(
        turnover_recent_sessions + turnover_baseline_sessions,
        liquidity_recent_sessions + liquidity_baseline_sessions,
    )
    if position + 1 < needed:
        raise ValueError(f"only {position + 1} sessions available; {needed} required")
    turn_recent_dates = calendar[
        position - turnover_recent_sessions + 1 : position + 1
    ]
    turn_baseline_dates = calendar[
        position - turnover_recent_sessions - turnover_baseline_sessions + 1 :
        position - turnover_recent_sessions + 1
    ]
    liquidity_recent_dates = calendar[
        position - liquidity_recent_sessions + 1 : position + 1
    ]
    liquidity_baseline_dates = calendar[
        position - liquidity_recent_sessions - liquidity_baseline_sessions + 1 :
        position - liquidity_recent_sessions + 1
    ]
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = frame["code"].astype(str)
    frame = frame[frame["code"].isin(codes)]
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("daily panel must be unique by date/code")
    rows = []
    for code in codes:
        stock = frame[frame["code"].eq(code)].set_index("date")
        turn_recent = pd.to_numeric(
            stock["turn"].reindex(turn_recent_dates), errors="coerce"
        ).dropna()
        turn_baseline = pd.to_numeric(
            stock["turn"].reindex(turn_baseline_dates), errors="coerce"
        )
        illiq_recent = pd.to_numeric(
            stock["daily_illiquidity"].reindex(liquidity_recent_dates), errors="coerce"
        ).dropna()
        illiq_baseline = pd.to_numeric(
            stock["daily_illiquidity"].reindex(liquidity_baseline_dates), errors="coerce"
        )
        turn_level = (
            float(turn_recent.median())
            if len(turn_recent) >= turnover_recent_minimum
            else float("nan")
        )
        liquidity_level = (
            float(illiq_recent.median())
            if len(illiq_recent) >= liquidity_recent_minimum
            else float("nan")
        )
        rows.append(
            {
                "code": code,
                "turnover_level": turn_level,
                "turnover_shock": _raw_mad_z(
                    turn_level, turn_baseline, turnover_baseline_minimum
                ),
                "liquidity_level": liquidity_level,
                "liquidity_shock": _raw_mad_z(
                    liquidity_level, illiq_baseline, liquidity_baseline_minimum
                ),
                "turnover_recent_n": len(turn_recent),
                "turnover_baseline_n": int(turn_baseline.notna().sum()),
                "liquidity_recent_n": len(illiq_recent),
                "liquidity_baseline_n": int(illiq_baseline.notna().sum()),
            }
        )
    stocks = pd.DataFrame(rows)

    def median(column: str) -> float:
        values = stocks[column].dropna()
        return float(values.median()) if len(values) else float("nan")

    valid_turn_shocks = stocks["turnover_shock"].dropna()
    turn_levels = stocks["turnover_level"].dropna()
    valid_liquidity_shocks = stocks["liquidity_shock"].dropna()
    result = {
        "decision_at": decision,
        "member_count": len(codes),
        "turnover_level": median("turnover_level"),
        "turnover_dispersion": (
            float(turn_levels.quantile(0.75) - turn_levels.quantile(0.25))
            if len(turn_levels)
            else float("nan")
        ),
        "illiquidity_level": median("liquidity_level"),
        "turnover_shock": median("turnover_shock"),
        "turnover_sync": (
            float((valid_turn_shocks > turnover_sync_threshold).mean())
            if len(valid_turn_shocks)
            else float("nan")
        ),
        "liquidity_shock": median("liquidity_shock"),
        "turnover_level_coverage": len(turn_levels) / len(codes),
        "turnover_shock_coverage": len(valid_turn_shocks) / len(codes),
        "illiquidity_level_coverage": stocks["liquidity_level"].notna().mean(),
        "liquidity_shock_coverage": len(valid_liquidity_shocks) / len(codes),
        "turnover_recent_start": turn_recent_dates.min(),
        "turnover_recent_end": turn_recent_dates.max(),
        "turnover_baseline_start": turn_baseline_dates.min(),
        "turnover_baseline_end": turn_baseline_dates.max(),
        "liquidity_recent_start": liquidity_recent_dates.min(),
        "liquidity_recent_end": liquidity_recent_dates.max(),
        "liquidity_baseline_start": liquidity_baseline_dates.min(),
        "liquidity_baseline_end": liquidity_baseline_dates.max(),
    }
    return {"portfolio": result, "stocks": stocks}


def factor_return_shock(
    returns: pd.Series,
    trading_calendar: pd.DatetimeIndex,
    *,
    decision_at: object,
    recent_sessions: int = 5,
    baseline_sessions: int = 252,
    baseline_minimum: int = 126,
) -> dict[str, object]:
    """Measure a negative recent factor return against past-only rolling returns."""

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a Series indexed by trading session")
    series = pd.Series(
        pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(returns.index)).normalize(),
    )
    if not series.index.is_unique:
        raise ValueError("factor returns must be unique by session")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    decision = pd.Timestamp(decision_at).normalize()
    position = int(calendar.get_indexer([decision])[0])
    if position < 0:
        raise ValueError("decision_at must be a trading session")
    required_sessions = baseline_sessions + 2 * recent_sessions - 1
    if position + 1 < required_sessions:
        raise ValueError("insufficient sessions for factor-return shock baseline")
    aligned = series.reindex(calendar)
    rolling_return = (1.0 + aligned).rolling(
        recent_sessions, min_periods=recent_sessions
    ).apply(np.prod, raw=True) - 1.0
    current_return = float(rolling_return.iloc[position])
    baseline_end_position = position - recent_sessions
    baseline_start_position = baseline_end_position - baseline_sessions + 1
    past_losses = -rolling_return.iloc[
        baseline_start_position : baseline_end_position + 1
    ]
    shock = _raw_mad_z(-current_return, past_losses, baseline_minimum)
    return {
        "decision_at": decision,
        "factor_return_recent": current_return,
        "factor_return_shock": shock,
        "factor_return_history_n": int(past_losses.notna().sum()),
        "factor_return_recent_start": calendar[position - recent_sessions + 1],
        "factor_return_recent_end": calendar[position],
        "factor_return_baseline_start": calendar[baseline_start_position],
        "factor_return_baseline_end": calendar[baseline_end_position],
    }
