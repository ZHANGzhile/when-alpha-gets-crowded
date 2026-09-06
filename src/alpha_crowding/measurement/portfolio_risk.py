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


def stock_risk_characteristics(
    daily: pd.DataFrame,
    actual_codes: list[str],
    trading_calendar: pd.DatetimeIndex,
    *,
    decision_at: object,
    turnover_recent_sessions: int = 5,
    turnover_baseline_sessions: int = 60,
    turnover_recent_minimum: int = 3,
    turnover_baseline_minimum: int = 40,
    liquidity_recent_sessions: int = 20,
    liquidity_baseline_sessions: int = 252,
    liquidity_recent_minimum: int = 15,
    liquidity_baseline_minimum: int = 126,
) -> pd.DataFrame:
    """Compute reusable stock-level G/S inputs with non-overlapping baselines."""

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
    stocks.insert(0, "decision_at", decision)
    for name, values in (
        ("turnover_recent", turn_recent_dates),
        ("turnover_baseline", turn_baseline_dates),
        ("liquidity_recent", liquidity_recent_dates),
        ("liquidity_baseline", liquidity_baseline_dates),
    ):
        stocks[f"{name}_start"] = values.min()
        stocks[f"{name}_end"] = values.max()
    return stocks


def aggregate_stock_risk_characteristics(
    stock_features: pd.DataFrame,
    actual_codes: list[str],
    *,
    decision_at: object,
    turnover_sync_threshold: float = 1.5,
) -> dict[str, object]:
    """Aggregate cached stock-level risk inputs to one equal-weight portfolio leg."""

    required = {
        "decision_at",
        "code",
        "turnover_level",
        "turnover_shock",
        "liquidity_level",
        "liquidity_shock",
        "turnover_recent_start",
        "turnover_recent_end",
        "turnover_baseline_start",
        "turnover_baseline_end",
        "liquidity_recent_start",
        "liquidity_recent_end",
        "liquidity_baseline_start",
        "liquidity_baseline_end",
    }
    missing = required - set(stock_features.columns)
    if missing:
        raise KeyError(f"missing stock-risk fields: {sorted(missing)}")
    codes = sorted(set(str(code) for code in actual_codes))
    if not codes:
        raise ValueError("actual_codes must not be empty")
    decision = pd.Timestamp(decision_at).normalize()
    stocks = stock_features.copy()
    stocks["decision_at"] = pd.to_datetime(stocks["decision_at"]).dt.normalize()
    stocks["code"] = stocks["code"].astype(str)
    stocks = stocks.loc[
        stocks["decision_at"].eq(decision) & stocks["code"].isin(codes)
    ]
    if stocks.duplicated(["decision_at", "code"]).any():
        raise ValueError("stock risk features must be unique by decision_at/code")
    missing_codes = set(codes) - set(stocks["code"])
    if missing_codes:
        raise ValueError(f"stock risk features lack members: {sorted(missing_codes)[:5]}")

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
        "turnover_recent_start": stocks["turnover_recent_start"].iloc[0],
        "turnover_recent_end": stocks["turnover_recent_end"].iloc[0],
        "turnover_baseline_start": stocks["turnover_baseline_start"].iloc[0],
        "turnover_baseline_end": stocks["turnover_baseline_end"].iloc[0],
        "liquidity_recent_start": stocks["liquidity_recent_start"].iloc[0],
        "liquidity_recent_end": stocks["liquidity_recent_end"].iloc[0],
        "liquidity_baseline_start": stocks["liquidity_baseline_start"].iloc[0],
        "liquidity_baseline_end": stocks["liquidity_baseline_end"].iloc[0],
    }
    return result


def aggregate_stock_risk_groups(
    stock_features: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    group_cols: tuple[str, ...],
    decision_at: object,
    turnover_sync_threshold: float = 1.5,
) -> pd.DataFrame:
    """Vectorize the same stock-level aggregation across many portfolio legs."""

    required_memberships = {"code", *group_cols}
    missing = required_memberships - set(memberships.columns)
    if missing:
        raise KeyError(f"missing grouped-membership fields: {sorted(missing)}")
    if memberships.duplicated([*group_cols, "code"]).any():
        raise ValueError("membership codes must be unique within every portfolio group")
    decision = pd.Timestamp(decision_at).normalize()
    stocks = stock_features.copy()
    stocks["decision_at"] = pd.to_datetime(stocks["decision_at"]).dt.normalize()
    stocks["code"] = stocks["code"].astype(str)
    stocks = stocks.loc[stocks["decision_at"].eq(decision)]
    members = memberships[[*group_cols, "code"]].copy()
    members["code"] = members["code"].astype(str)
    joined = members.merge(stocks, on="code", how="left", validate="many_to_one")
    if joined["decision_at"].isna().any():
        missing_codes = sorted(joined.loc[joined["decision_at"].isna(), "code"].unique())
        raise ValueError(f"stock risk features lack members: {missing_codes[:5]}")

    grouped = joined.groupby(list(group_cols), sort=True, dropna=False)
    result = grouped.agg(
        member_count=("code", "size"),
        turnover_level=("turnover_level", "median"),
        illiquidity_level=("liquidity_level", "median"),
        turnover_shock=("turnover_shock", "median"),
        liquidity_shock=("liquidity_shock", "median"),
        turnover_level_coverage=("turnover_level", lambda values: values.notna().mean()),
        turnover_shock_coverage=("turnover_shock", lambda values: values.notna().mean()),
        illiquidity_level_coverage=("liquidity_level", lambda values: values.notna().mean()),
        liquidity_shock_coverage=("liquidity_shock", lambda values: values.notna().mean()),
    )
    result["turnover_dispersion"] = grouped["turnover_level"].apply(
        lambda values: values.quantile(0.75) - values.quantile(0.25)
        if values.notna().any()
        else np.nan
    )
    result["turnover_sync"] = grouped["turnover_shock"].apply(
        lambda values: (values.dropna() > turnover_sync_threshold).mean()
        if values.notna().any()
        else np.nan
    )
    result["decision_at"] = decision
    for column in (
        "turnover_recent_start",
        "turnover_recent_end",
        "turnover_baseline_start",
        "turnover_baseline_end",
        "liquidity_recent_start",
        "liquidity_recent_end",
        "liquidity_baseline_start",
        "liquidity_baseline_end",
    ):
        result[column] = stocks[column].iloc[0]
    return result.reset_index()


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
    """Compute portfolio G/S inputs through the reusable stock-level path."""

    stocks = stock_risk_characteristics(
        daily,
        actual_codes,
        trading_calendar,
        decision_at=decision_at,
        turnover_recent_sessions=turnover_recent_sessions,
        turnover_baseline_sessions=turnover_baseline_sessions,
        turnover_recent_minimum=turnover_recent_minimum,
        turnover_baseline_minimum=turnover_baseline_minimum,
        liquidity_recent_sessions=liquidity_recent_sessions,
        liquidity_baseline_sessions=liquidity_baseline_sessions,
        liquidity_recent_minimum=liquidity_recent_minimum,
        liquidity_baseline_minimum=liquidity_baseline_minimum,
    )
    portfolio = aggregate_stock_risk_characteristics(
        stocks,
        actual_codes,
        decision_at=decision_at,
        turnover_sync_threshold=turnover_sync_threshold,
    )
    return {"portfolio": portfolio, "stocks": stocks.drop(columns="decision_at")}


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
