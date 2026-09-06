"""Normalize BaoStock raw fields into point-in-time research primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_baostock_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Create audited returns, return index, PB and float-share estimates.

    BaoStock leaves ``pctChg`` blank on some suspended rows while preserving an
    unchanged close.  Those rows receive a zero holding return and remain
    non-tradable.  A missing return on a row marked tradable is a hard error.
    """

    required = {
        "date", "code", "close", "volume", "amount", "turn", "tradestatus",
        "pctChg", "pbMRQ", "isST",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing normalization fields: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    if out.duplicated(["date", "code"]).any():
        raise ValueError("raw daily data must be unique by date/code")
    out = out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    numeric = ["close", "volume", "amount", "turn", "pctChg", "pbMRQ"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    out["tradestatus"] = pd.to_numeric(out["tradestatus"], errors="raise").astype(int)
    out["isST"] = pd.to_numeric(out["isST"], errors="raise").astype(int)
    if not out["tradestatus"].isin([0, 1]).all() or not out["isST"].isin([0, 1]).all():
        raise ValueError("tradestatus and isST must be binary")
    if (out["close"].dropna() <= 0).any():
        raise ValueError("close must be positive")

    out["daily_return"] = out["pctChg"] / 100.0
    suspended_missing = out["tradestatus"].eq(0) & out["daily_return"].isna()
    out.loc[suspended_missing, "daily_return"] = 0.0
    traded_missing = out["tradestatus"].eq(1) & out["daily_return"].isna()
    if traded_missing.any():
        examples = out.loc[traded_missing, ["date", "code"]].head().to_dict("records")
        raise ValueError(f"tradable rows contain missing returns: {examples}")
    if (out["daily_return"] <= -1).any():
        raise ValueError("daily return at or below -100% requires terminal-event handling")
    out["return_index"] = (1.0 + out["daily_return"]).groupby(out["code"]).cumprod()

    current_float = out["volume"] / (out["turn"] / 100.0)
    valid_float = (
        out["tradestatus"].eq(1)
        & (out["volume"] > 0)
        & (out["turn"] > 0)
        & np.isfinite(current_float)
        & (current_float > 0)
    )
    out["float_shares_observed"] = current_float.where(valid_float)
    out["float_shares_estimate"] = out.groupby("code", sort=False)[
        "float_shares_observed"
    ].ffill()
    out["float_market_cap"] = out["close"] * out["float_shares_estimate"]
    out["pb_mrq"] = out["pbMRQ"].where(out["pbMRQ"] > 0)
    out["eligible_for_new_position"] = (
        out["tradestatus"].eq(1) & out["isST"].eq(0)
    )
    return out


def add_lagged_matching_characteristics(
    normalized: pd.DataFrame,
    *,
    information_lag_days: int = 20,
    liquidity_window_days: int = 20,
    minimum_liquidity_days: int = 15,
) -> pd.DataFrame:
    """Add placebo matching fields whose estimation window ends at t-lag."""

    required = {"date", "code", "daily_return", "amount", "float_market_cap"}
    missing = required - set(normalized.columns)
    if missing:
        raise KeyError(f"missing matching-characteristic fields: {sorted(missing)}")
    if information_lag_days < 1 or liquidity_window_days < 1:
        raise ValueError("matching lag and window must be positive")
    if not 1 <= minimum_liquidity_days <= liquidity_window_days:
        raise ValueError("minimum_liquidity_days must be within the window")
    out = normalized.copy().sort_values(["code", "date"], kind="mergesort")
    daily_return = pd.to_numeric(out["daily_return"], errors="coerce")
    amount = pd.to_numeric(out["amount"], errors="coerce")
    out["daily_illiquidity"] = (daily_return.abs() / amount.where(amount > 0)).where(
        np.isfinite(daily_return) & np.isfinite(amount)
    )

    def lagged_liquidity(series: pd.Series) -> pd.Series:
        return series.shift(information_lag_days).rolling(
            liquidity_window_days, min_periods=minimum_liquidity_days
        ).median()

    out["lagged_liquidity"] = out.groupby("code", sort=False)[
        "daily_illiquidity"
    ].transform(lagged_liquidity)
    out["lagged_float_market_cap"] = out.groupby("code", sort=False)[
        "float_market_cap"
    ].shift(information_lag_days)
    return out.reset_index(drop=True)
