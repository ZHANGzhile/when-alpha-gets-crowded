"""Transparent measurement primitives for alpha-crowding research.

The public API deliberately keeps the three economic feature families separate:

``crowding``
    Factor-specific structural abnormality after matched-placebo and historical
    adjustment.
``generic risk``
    Standardized levels of the observed portfolio's ordinary risk features.
``stress``
    Turnover, liquidity, market, and return shocks that may trigger an unwind.

Nothing in this module estimates institutional holdings or supports a causal
interpretation by itself.
"""

from .core import (
    CROWDING_COMPONENTS,
    GENERIC_RISK_COMPONENTS,
    STRESS_COMPONENTS,
    PlaceboSummary,
    compute_crowding_state,
    compute_generic_risk,
    compute_stress_trigger,
    historical_zscore,
    historical_zscore_by_group,
    historical_zscore_by_group_trading_window,
    historical_zscore_trading_window,
    summarize_matched_placebo,
    summarize_matched_placebos,
    validate_component_frame,
)
from .statistics import (
    CorrelationFeatures,
    amihud_illiquidity,
    industry_residual_returns,
    robust_past_shock,
    shrunk_correlation_features,
)
from .placebo import MatchedPlaceboResult, draw_matched_placebos, stable_group_seed
from .structural import (
    ALGORITHM_VERSION,
    StructuralPlaceboResult,
    candidate_snapshot_sha256,
    measure_structural_placebos,
)
from .convergence import (
    CONVERGENCE_ALGORITHM_VERSION,
    ConvergencePlaceboResult,
    measure_strategy_convergence_placebos,
    pairwise_jaccard,
)
from .portfolio_risk import factor_return_shock, portfolio_risk_characteristics
from .states import causal_rank_ic_state, trailing_return_state
from .pseudo_pipeline import (
    leave_one_out_placebo_adjustment,
    measure_continuous_pseudo_convergence,
    measure_continuous_pseudo_structure,
)

__all__ = [
    "CROWDING_COMPONENTS",
    "GENERIC_RISK_COMPONENTS",
    "STRESS_COMPONENTS",
    "PlaceboSummary",
    "compute_crowding_state",
    "compute_generic_risk",
    "compute_stress_trigger",
    "historical_zscore",
    "historical_zscore_by_group",
    "historical_zscore_by_group_trading_window",
    "historical_zscore_trading_window",
    "summarize_matched_placebo",
    "summarize_matched_placebos",
    "validate_component_frame",
    "CorrelationFeatures",
    "amihud_illiquidity",
    "industry_residual_returns",
    "robust_past_shock",
    "shrunk_correlation_features",
    "MatchedPlaceboResult",
    "draw_matched_placebos",
    "stable_group_seed",
    "ALGORITHM_VERSION",
    "StructuralPlaceboResult",
    "candidate_snapshot_sha256",
    "measure_structural_placebos",
    "CONVERGENCE_ALGORITHM_VERSION",
    "ConvergencePlaceboResult",
    "measure_strategy_convergence_placebos",
    "pairwise_jaccard",
    "portfolio_risk_characteristics",
    "factor_return_shock",
    "causal_rank_ic_state",
    "trailing_return_state",
    "leave_one_out_placebo_adjustment",
    "measure_continuous_pseudo_convergence",
    "measure_continuous_pseudo_structure",
]
