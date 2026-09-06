"""Factor signal and portfolio construction primitives."""

from .signals import assign_legs, compute_raw_signals, mad_winsorize
from .portfolio import DEFAULT_FACTORS, build_factor_memberships
from .returns import (
    MissingHeldReturnError,
    combine_long_short_returns,
    simulate_factor_leg_returns,
)

__all__ = [
    "DEFAULT_FACTORS",
    "assign_legs",
    "build_factor_memberships",
    "compute_raw_signals",
    "mad_winsorize",
    "MissingHeldReturnError",
    "combine_long_short_returns",
    "simulate_factor_leg_returns",
]
