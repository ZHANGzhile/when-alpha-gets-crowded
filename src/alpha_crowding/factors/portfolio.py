"""Production factor-leg construction from point-in-time weekly security rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .signals import assign_legs


DEFAULT_FACTORS = {
    "MOM": "momentum",
    "REV": "reversal",
    "LOWVOL": "low_volatility",
    "VALUE": "value",
}


def build_factor_memberships(
    weekly: pd.DataFrame,
    *,
    factors: Mapping[str, str] = DEFAULT_FACTORS,
    enabled_factors: Sequence[str] = ("MOM", "REV", "LOWVOL", "VALUE"),
    date_col: str = "date",
    code_col: str = "code",
    industry_col: str = "industry",
    quantile: float = 0.10,
) -> pd.DataFrame:
    """Assign deterministic long/short legs using only eligible PIT rows."""

    enabled = tuple(enabled_factors)
    unknown = set(enabled) - set(factors)
    if unknown:
        raise ValueError(f"unknown enabled factors: {sorted(unknown)}")
    required = {
        date_col, code_col, industry_col, "base_eligible", "industry_is_fresh",
        *(factors[name] for name in enabled),
    }
    missing = required - set(weekly.columns)
    if missing:
        raise KeyError(f"missing factor-membership fields: {sorted(missing)}")
    if weekly.duplicated([date_col, code_col]).any():
        raise ValueError("weekly security rows must be unique by date/code")

    outputs = []
    for factor in enabled:
        signal_col = factors[factor]
        numeric_signal = pd.to_numeric(weekly[signal_col], errors="coerce")
        eligible = (
            weekly["base_eligible"].fillna(False).astype(bool)
            & weekly["industry_is_fresh"].fillna(False).astype(bool)
            & weekly[industry_col].notna()
            & np.isfinite(numeric_signal)
        )
        passthrough = [
            column
            for column in (
                "lagged_liquidity",
                "lagged_float_market_cap",
                "float_market_cap",
                "amount",
                "turn",
            )
            if column in weekly.columns
        ]
        candidates = weekly.loc[
            eligible, [date_col, code_col, industry_col, signal_col, *passthrough]
        ].copy()
        if candidates.empty:
            continue
        assigned = assign_legs(
            candidates,
            signal_col=signal_col,
            date_col=date_col,
            code_col=code_col,
            industry_col=industry_col,
            quantile=quantile,
        ).rename(columns={signal_col: "raw_signal"})
        assigned["factor"] = factor
        assigned["signed_weight"] = np.select(
            [assigned["leg"].eq("LONG"), assigned["leg"].eq("SHORT")],
            [assigned["weight"], -assigned["weight"]],
            default=0.0,
        )
        outputs.append(assigned)
    if not outputs:
        return pd.DataFrame()
    result = pd.concat(outputs, ignore_index=True).sort_values(
        [date_col, "factor", code_col]
    ).reset_index(drop=True)
    legs = result[result["leg"].isin(["LONG", "SHORT"])]
    sums = legs.groupby([date_col, "factor", "leg"])["weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("factor leg weights do not sum to one")
    overlap = legs.groupby([date_col, "factor", code_col])["leg"].nunique()
    if (overlap > 1).any():
        raise AssertionError("a security appears in both factor legs")
    return result
