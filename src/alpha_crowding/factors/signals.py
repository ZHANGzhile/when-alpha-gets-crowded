"""Transparent V1 factor definitions with deterministic portfolio assignment."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_BAR_COLUMNS = {"date", "code", "close", "pb_mrq"}


def compute_raw_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute V1 raw signals without cross-sectional ranking.

    `close` must be the point-in-time, corporate-action-consistent return price.
    Rows are sorted internally by code/date. The returned frame preserves the
    original row index in `_source_index` for auditability.
    """

    missing = REQUIRED_BAR_COLUMNS - set(bars.columns)
    if missing:
        raise KeyError(f"missing bar columns: {sorted(missing)}")
    if bars.duplicated(["date", "code"]).any():
        raise ValueError("bars must be unique by date/code")

    out = bars.copy()
    out["_source_index"] = out.index
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out = out.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)

    close = pd.to_numeric(out["close"], errors="coerce")
    if (close.dropna() <= 0).any():
        raise ValueError("close must be positive where present")
    out["close"] = close

    by_code = out.groupby("code", sort=False)["close"]
    out["daily_return"] = by_code.transform(lambda s: s.pct_change(fill_method=None))
    out["momentum"] = by_code.shift(21) / by_code.shift(252) - 1.0
    out["reversal"] = -(out["close"] / by_code.shift(20) - 1.0)
    out["low_volatility"] = -out.groupby("code", sort=False)["daily_return"].transform(
        lambda s: s.rolling(60, min_periods=60).std(ddof=1)
    )

    pb = pd.to_numeric(out["pb_mrq"], errors="coerce")
    out["value"] = np.where(pb > 0, -np.log(pb), np.nan)
    return out


def mad_winsorize(values: pd.Series, *, threshold: float = 3.5) -> pd.Series:
    """Clip a series around its median using raw MAD; preserve missing values."""

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    center = float(valid.median())
    mad = float((valid - center).abs().median())
    if not np.isfinite(mad) or mad == 0:
        return numeric
    return numeric.clip(center - threshold * mad, center + threshold * mad)


def _assign_one_date(
    frame: pd.DataFrame,
    *,
    score_col: str,
    code_col: str,
    quantile: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["leg"] = "MIDDLE"
    valid = result[score_col].notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        result["weight"] = 0.0
        return result
    n_leg = max(1, math.floor(n_valid * quantile))
    if 2 * n_leg > n_valid:
        raise ValueError("portfolio quantile creates overlapping legs")

    ranked = result.loc[valid].sort_values(
        [score_col, code_col], ascending=[True, True], kind="mergesort"
    )
    short_idx = ranked.index[:n_leg]
    long_idx = ranked.index[-n_leg:]
    result.loc[short_idx, "leg"] = "SHORT"
    result.loc[long_idx, "leg"] = "LONG"
    result["weight"] = 0.0
    result.loc[short_idx, "weight"] = 1.0 / n_leg
    result.loc[long_idx, "weight"] = 1.0 / n_leg
    return result


def assign_legs(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    date_col: str = "date",
    code_col: str = "code",
    industry_col: str = "industry",
    quantile: float = 0.10,
    mad_threshold: float = 3.5,
) -> pd.DataFrame:
    """Winsorize by date, rank within industry, and deterministically assign legs."""

    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    required = {date_col, code_col, industry_col, signal_col}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing membership columns: {sorted(missing)}")

    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="raise")
    if out.duplicated([date_col, code_col]).any():
        raise ValueError("membership candidates must be unique by date/code")
    out["signal_winsorized"] = out.groupby(date_col, group_keys=False)[signal_col].transform(
        lambda s: mad_winsorize(s, threshold=mad_threshold)
    )
    out["industry_rank"] = out.groupby(
        [date_col, industry_col], dropna=False
    )["signal_winsorized"].rank(method="average", pct=True)

    pieces = [
        _assign_one_date(
            group,
            score_col="industry_rank",
            code_col=code_col,
            quantile=quantile,
        )
        for _, group in out.groupby(date_col, sort=True, group_keys=False)
    ]
    return pd.concat(pieces).sort_values([date_col, code_col]).reset_index(drop=True)

