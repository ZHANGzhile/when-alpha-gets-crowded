"""Time-valid model comparison and inference utilities."""

from .bootstrap import BlockBootstrapResult, moving_block_bootstrap_mean
from .modeling import (
    NestedPredictionResult,
    RegularizationSearchResult,
    binary_log_loss,
    common_complete_case_mask,
    fit_nested_logistic_models,
    temporal_regularization_search,
    weekly_average_loss,
)
from .splits import (
    assert_lead_time_information_cutoff,
    purge_training_rows,
    purged_training_mask,
)
from .walk_forward import (
    AnnualFold,
    WalkForwardResult,
    annual_expanding_folds,
    run_annual_walk_forward,
)
from .protocol import require_protocol_freeze

__all__ = [
    "BlockBootstrapResult",
    "AnnualFold",
    "NestedPredictionResult",
    "RegularizationSearchResult",
    "WalkForwardResult",
    "annual_expanding_folds",
    "assert_lead_time_information_cutoff",
    "binary_log_loss",
    "common_complete_case_mask",
    "fit_nested_logistic_models",
    "moving_block_bootstrap_mean",
    "purge_training_rows",
    "purged_training_mask",
    "run_annual_walk_forward",
    "temporal_regularization_search",
    "weekly_average_loss",
    "require_protocol_freeze",
]
