"""Outcome and label utilities for the alpha-crowding study."""

from .active import active_drawdown_series, active_max_drawdown, active_mdd, relative_nav
from .forward import forward_window_outcome
from .fixed import fixed_membership_leg_path
from .events import build_event_study_panel, merge_crash_episodes
from .labels import MatureTailSpec, build_mature_tail_labels, mature_historical_quantiles

__all__ = [
    "MatureTailSpec",
    "active_drawdown_series",
    "active_max_drawdown",
    "active_mdd",
    "forward_window_outcome",
    "fixed_membership_leg_path",
    "build_mature_tail_labels",
    "build_event_study_panel",
    "mature_historical_quantiles",
    "merge_crash_episodes",
    "relative_nav",
]
