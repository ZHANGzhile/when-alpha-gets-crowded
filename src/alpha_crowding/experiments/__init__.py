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
from .lead import attach_lead_features, build_lead_labels
from .falsification import (
    discriminant_validity_table,
    multivariate_residual_diagnostic,
    time_misalign_features,
)
from .pseudo import (
    ContinuousPseudoMembershipResult,
    PSEUDO_ALGORITHM_VERSION,
    build_continuous_permuted_memberships,
    stable_pseudo_seed,
)
from .protocol import require_protocol_freeze

__all__ = [
    "BlockBootstrapResult",
    "ContinuousPseudoMembershipResult",
    "AnnualFold",
    "NestedPredictionResult",
    "PSEUDO_ALGORITHM_VERSION",
    "RegularizationSearchResult",
    "WalkForwardResult",
    "annual_expanding_folds",
    "attach_lead_features",
    "assert_lead_time_information_cutoff",
    "binary_log_loss",
    "build_continuous_permuted_memberships",
    "build_lead_labels",
    "common_complete_case_mask",
    "discriminant_validity_table",
    "fit_nested_logistic_models",
    "moving_block_bootstrap_mean",
    "multivariate_residual_diagnostic",
    "purge_training_rows",
    "purged_training_mask",
    "run_annual_walk_forward",
    "stable_pseudo_seed",
    "temporal_regularization_search",
    "time_misalign_features",
    "weekly_average_loss",
    "require_protocol_freeze",
]
