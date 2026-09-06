"""Auditable foundations for active-exposure backtests."""

from .accounting import (
    RebalanceResult,
    calculate_rebalance,
    combine_stock_target_weights,
    drift_weights,
    rebalance_after_drift,
)
from .controller import (
    DEFAULT_RISK_BANDS,
    active_weights_from_oos_probabilities,
    historical_risk_percentile,
    probability_to_active_weight,
    risk_percentile_to_active_weight,
)
from .metrics import (
    RelativePerformance,
    active_cvar,
    arithmetic_active_returns,
    evaluate_relative_performance,
    information_ratio,
    max_drawdown,
    relative_nav,
    tracking_error,
)

__all__ = [
    "DEFAULT_RISK_BANDS",
    "RebalanceResult",
    "RelativePerformance",
    "active_cvar",
    "active_weights_from_oos_probabilities",
    "arithmetic_active_returns",
    "calculate_rebalance",
    "combine_stock_target_weights",
    "drift_weights",
    "evaluate_relative_performance",
    "historical_risk_percentile",
    "information_ratio",
    "max_drawdown",
    "probability_to_active_weight",
    "rebalance_after_drift",
    "relative_nav",
    "risk_percentile_to_active_weight",
    "tracking_error",
]
