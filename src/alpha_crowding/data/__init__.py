"""Point-in-time data contracts and validation."""

from .contracts import (
    AvailabilityViolation,
    assert_information_available,
    assert_unique_key,
    validate_interval,
)
from .baostock_source import (
    BaoStockQueryError,
    fetch_constituents,
    fetch_daily_bars,
    fetch_industry,
    fetch_index_bars,
    fetch_trade_calendar,
    result_to_frame,
    session,
)
from .industry_taxonomy import (
    classify_industry_snapshot,
    industry_snapshot_summary,
    stable_industry_sector,
)
from .quality import (
    compare_adjustments,
    daily_history_acceptance_failures,
    summarize_daily_history,
)
from .membership import (
    classify_missing_membership_market_rows,
    repair_merger_membership_gaps,
    validate_constituent_snapshot,
    weekly_last_sessions,
)
from .normalize import add_lagged_matching_characteristics, normalize_baostock_daily
from .asof import join_industry_asof

__all__ = [
    "AvailabilityViolation",
    "assert_information_available",
    "assert_unique_key",
    "validate_interval",
    "BaoStockQueryError",
    "fetch_constituents",
    "fetch_daily_bars",
    "fetch_industry",
    "fetch_index_bars",
    "fetch_trade_calendar",
    "result_to_frame",
    "session",
    "classify_industry_snapshot",
    "industry_snapshot_summary",
    "stable_industry_sector",
    "compare_adjustments",
    "daily_history_acceptance_failures",
    "summarize_daily_history",
    "repair_merger_membership_gaps",
    "classify_missing_membership_market_rows",
    "validate_constituent_snapshot",
    "weekly_last_sessions",
    "normalize_baostock_daily",
    "add_lagged_matching_characteristics",
    "join_industry_asof",
]
