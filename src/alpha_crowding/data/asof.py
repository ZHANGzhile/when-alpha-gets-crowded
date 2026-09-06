"""Leakage-safe as-of joins for slowly changing point-in-time fields."""

from __future__ import annotations

import pandas as pd


def join_industry_asof(
    weekly: pd.DataFrame,
    industry: pd.DataFrame,
    *,
    decision_col: str = "date",
    snapshot_col: str = "requested_date",
    code_col: str = "code",
    maximum_age_days: int = 45,
) -> pd.DataFrame:
    """Join each security-week to its latest nonfuture industry snapshot."""

    if maximum_age_days < 0:
        raise ValueError("maximum_age_days must be non-negative")
    required_weekly = {decision_col, code_col}
    required_industry = {snapshot_col, code_col, "industry", "stable_sector"}
    if required_weekly - set(weekly.columns):
        raise KeyError("weekly frame lacks decision/code columns")
    if required_industry - set(industry.columns):
        raise KeyError("industry frame lacks PIT classification columns")
    left = weekly.copy()
    right = industry.copy()
    left[decision_col] = pd.to_datetime(left[decision_col], errors="raise").dt.normalize()
    right[snapshot_col] = pd.to_datetime(right[snapshot_col], errors="raise").dt.normalize()
    if left.duplicated([decision_col, code_col]).any():
        raise ValueError("weekly rows must be unique by date/code")
    if right.duplicated([snapshot_col, code_col]).any():
        raise ValueError("industry rows must be unique by snapshot/code")
    left["_row_order"] = range(len(left))
    joined = pd.merge_asof(
        left.sort_values([decision_col, code_col]),
        right.sort_values([snapshot_col, code_col]),
        left_on=decision_col,
        right_on=snapshot_col,
        by=code_col,
        direction="backward",
        allow_exact_matches=True,
    )
    joined["industry_age_days"] = (
        joined[decision_col] - joined[snapshot_col]
    ).dt.days
    if (joined["industry_age_days"].dropna() < 0).any():
        raise AssertionError("future industry snapshot joined to weekly row")
    joined["industry_is_fresh"] = joined["industry_age_days"].between(
        0, maximum_age_days
    )
    return joined.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
