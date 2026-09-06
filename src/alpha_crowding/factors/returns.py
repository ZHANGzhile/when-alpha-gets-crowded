"""Daily factor-leg accounting with delayed weekly execution and weight drift."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from alpha_crowding.backtest.accounting import drift_weights


class MissingHeldReturnError(RuntimeError):
    """Raised when an existing holding lacks a return or settlement observation."""


def _next_session(date: pd.Timestamp, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    location = sessions.searchsorted(date, side="right")
    if location >= len(sessions):
        raise ValueError(f"no execution session exists after decision {date.date()}")
    return sessions[location]


def simulate_factor_leg_returns(
    memberships: pd.DataFrame,
    security_returns: pd.DataFrame,
    trading_sessions: Sequence[object],
    *,
    decision_col: str = "date",
    return_col: str = "daily_return",
) -> pd.DataFrame:
    """Simulate long-only factor legs; targets take effect on the next session."""

    membership_required = {decision_col, "factor", "leg", "code", "weight"}
    return_required = {"date", "code", return_col}
    missing_membership = membership_required - set(memberships.columns)
    missing_returns = return_required - set(security_returns.columns)
    if missing_membership:
        raise KeyError(f"missing membership fields: {sorted(missing_membership)}")
    if missing_returns:
        raise KeyError(f"missing security-return fields: {sorted(missing_returns)}")
    selected = memberships[memberships["leg"].isin(["LONG", "SHORT"])].copy()
    if selected.empty:
        raise ValueError("memberships contain no LONG or SHORT rows")
    if selected.duplicated([decision_col, "factor", "leg", "code"]).any():
        raise ValueError("factor memberships contain duplicate target keys")
    selected[decision_col] = pd.to_datetime(selected[decision_col]).dt.normalize()
    selected["weight"] = pd.to_numeric(selected["weight"], errors="raise")
    target_sums = selected.groupby([decision_col, "factor", "leg"])["weight"].sum()
    if not np.allclose(target_sums.to_numpy(), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("every factor leg target must sum to one")

    sessions = pd.DatetimeIndex(pd.to_datetime(list(trading_sessions))).normalize()
    sessions = sessions.drop_duplicates().sort_values()
    if sessions.empty:
        raise ValueError("trading_sessions must not be empty")
    returns = security_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    returns[return_col] = pd.to_numeric(returns[return_col], errors="coerce")
    if returns.duplicated(["date", "code"]).any():
        raise ValueError("security returns must be unique by date/code")
    return_lookup = returns.set_index(["date", "code"])[return_col]

    results = []
    for (factor, leg), group in selected.groupby(["factor", "leg"], sort=True):
        schedules: dict[pd.Timestamp, tuple[pd.Timestamp, dict[str, float]]] = {}
        for decision, rows in group.groupby(decision_col, sort=True):
            decision_at = pd.Timestamp(decision).normalize()
            execution_at = _next_session(decision_at, sessions)
            if execution_at in schedules:
                raise ValueError("multiple decisions map to one execution session")
            schedules[execution_at] = (
                decision_at,
                dict(zip(rows["code"].astype(str), rows["weight"].astype(float))),
            )
        first_execution = min(schedules)
        weights: dict[str, float] | None = None
        active_decision: pd.Timestamp | None = None
        for date in sessions[sessions >= first_execution]:
            rebalance = date in schedules
            turnover = 0.0
            if rebalance:
                active_decision, target = schedules[date]
                if weights is None:
                    turnover = 1.0
                else:
                    assets = set(weights) | set(target)
                    turnover = sum(
                        abs(target.get(code, 0.0) - weights.get(code, 0.0))
                        for code in assets
                    )
                weights = target
            if weights is None:
                continue
            keys = pd.MultiIndex.from_tuples([(date, code) for code in weights])
            daily = return_lookup.reindex(keys)
            missing = [code for code, value in zip(weights, daily) if pd.isna(value)]
            if missing:
                raise MissingHeldReturnError(
                    f"{factor}/{leg} held returns missing at {date.date()}: {missing[:10]}"
                )
            asset_returns = {
                code: float(value) for code, value in zip(weights, daily.to_numpy())
            }
            portfolio_return = float(
                sum(weights[code] * asset_returns[code] for code in weights)
            )
            results.append(
                {
                    "date": date,
                    "factor": factor,
                    "leg": leg,
                    "decision_at": active_decision,
                    "daily_return": portfolio_return,
                    "rebalance": rebalance,
                    "gross_traded_fraction": turnover,
                    "holding_count": len(weights),
                }
            )
            weights = drift_weights(weights, asset_returns)
    return pd.DataFrame(results).sort_values(["date", "factor", "leg"]).reset_index(drop=True)


def combine_long_short_returns(leg_returns: pd.DataFrame) -> pd.DataFrame:
    """Combine matched long and short leg returns using +1 long / -1 short."""

    required = {"date", "factor", "leg", "daily_return"}
    missing = required - set(leg_returns.columns)
    if missing:
        raise KeyError(f"missing leg-return fields: {sorted(missing)}")
    selected = leg_returns[leg_returns["leg"].isin(["LONG", "SHORT"])].copy()
    if selected.duplicated(["date", "factor", "leg"]).any():
        raise ValueError("leg returns must be unique by date/factor/leg")
    wide = selected.pivot(index=["date", "factor"], columns="leg", values="daily_return")
    if set(wide.columns) != {"LONG", "SHORT"} or wide.isna().any().any():
        raise ValueError("long and short leg returns must both exist on every factor date")
    result = wide.rename(
        columns={"LONG": "long_return", "SHORT": "short_return"}
    ).reset_index()
    result["long_short_return"] = result["long_return"] - result["short_return"]
    return result.sort_values(["date", "factor"]).reset_index(drop=True)
