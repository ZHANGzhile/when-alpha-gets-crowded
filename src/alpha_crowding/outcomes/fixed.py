"""Fixed-membership forward paths for the footprint mechanism test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_crowding.backtest.accounting import drift_weights
from alpha_crowding.factors.returns import MissingHeldReturnError


def fixed_membership_leg_path(
    members: pd.DataFrame,
    security_returns: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
    *,
    decision_at: object,
    horizon_sessions: int = 20,
) -> pd.DataFrame:
    """Hold decision-date members for the exact future horizon with drift."""

    required_members = {"code", "weight"}
    required_returns = {"date", "code", "daily_return"}
    if missing := required_members - set(members.columns):
        raise KeyError(f"missing fixed-member fields: {sorted(missing)}")
    if missing := required_returns - set(security_returns.columns):
        raise KeyError(f"missing security-return fields: {sorted(missing)}")
    if members.empty or members["code"].duplicated().any():
        raise ValueError("fixed members must be nonempty and unique")
    weights = dict(
        zip(members["code"].astype(str), pd.to_numeric(members["weight"], errors="raise"))
    )
    if not np.isclose(sum(weights.values()), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("fixed-member weights must sum to one")
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    decision = pd.Timestamp(decision_at).normalize()
    position = int(calendar.get_indexer([decision])[0])
    if position < 0:
        raise ValueError("decision_at must be a trading session")
    if position + horizon_sessions >= len(calendar):
        return pd.DataFrame(
            columns=["date", "decision_at", "daily_return", "holding_count"]
        )
    dates = calendar[position + 1 : position + horizon_sessions + 1]
    returns = security_returns.copy()
    returns["date"] = pd.to_datetime(returns["date"]).dt.normalize()
    returns["code"] = returns["code"].astype(str)
    if returns.duplicated(["date", "code"]).any():
        raise ValueError("security returns must be unique by date/code")
    lookup = returns.set_index(["date", "code"])["daily_return"]
    rows = []
    for date in dates:
        daily = lookup.reindex(pd.MultiIndex.from_tuples([(date, code) for code in weights]))
        missing_codes = [code for code, value in zip(weights, daily) if pd.isna(value)]
        if missing_codes:
            raise MissingHeldReturnError(
                f"fixed members missing returns at {date.date()}: {missing_codes[:10]}"
            )
        asset_returns = {
            code: float(value) for code, value in zip(weights, daily.to_numpy())
        }
        portfolio_return = float(
            sum(weights[code] * asset_returns[code] for code in weights)
        )
        rows.append(
            {
                "date": date,
                "decision_at": decision,
                "daily_return": portfolio_return,
                "holding_count": len(weights),
            }
        )
        weights = drift_weights(weights, asset_returns)
    return pd.DataFrame(rows)
