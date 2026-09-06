"""Build time-valid lead experiment panels from weekly state vintages."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from alpha_crowding.outcomes.labels import MatureTailSpec, build_mature_tail_labels

from .splits import assert_lead_time_information_cutoff, lead_time_cutoffs


def build_lead_labels(
    outcomes: pd.DataFrame,
    *,
    sessions: Sequence[object],
    lead_sessions: int,
    spec: MatureTailSpec,
    group_cols: Sequence[str] = (
        "factor",
        "target_family",
        "membership_mode",
        "horizon_sessions",
    ),
) -> pd.DataFrame:
    """Rebuild the target threshold using information available at alert time.

    ``target_anchor_at`` remains the factor decision whose following return
    window defines the outcome.  ``decision_at`` is the earlier date on which
    the alert is issued and is therefore the date used by every model split.
    """

    required = {
        *group_cols,
        "decision_at",
        "label_start_at",
        "label_end_at",
        "future_return",
    }
    missing = required - set(outcomes.columns)
    if missing:
        raise KeyError(f"missing lead outcome columns: {sorted(missing)}")
    if lead_sessions < 0:
        raise ValueError("lead_sessions must be non-negative")

    frame = outcomes.copy()
    frame["target_anchor_at"] = pd.to_datetime(
        frame["decision_at"], errors="raise"
    ).dt.normalize()
    calendar = pd.DatetimeIndex(pd.to_datetime(list(sessions), errors="raise")).normalize()
    positions = calendar.get_indexer(pd.DatetimeIndex(frame["target_anchor_at"]))
    if (positions < 0).any():
        raise ValueError("target anchors must belong to the trading calendar")
    frame = frame.loc[positions >= lead_sessions].copy()
    if frame.empty:
        raise ValueError("no outcome has enough trading-session history for this lead")
    frame["issue_at"] = lead_time_cutoffs(
        frame["target_anchor_at"],
        sessions=sessions,
        lead_sessions=lead_sessions,
    ).to_numpy()
    frame["information_cutoff_at"] = frame["issue_at"]
    frame["threshold_available_at"] = frame["issue_at"]

    labelled = build_mature_tail_labels(
        frame,
        sessions=sessions,
        group_cols=group_cols,
        information_cutoff_col="information_cutoff_at",
        spec=spec,
        threshold_name="historical_tail_threshold",
        history_count_name="mature_history_count",
        label_name="target",
    )
    labelled = labelled.loc[labelled["target"].notna()].copy()
    labelled["target"] = labelled["target"].astype(int)
    labelled["decision_at"] = labelled["issue_at"]
    labelled["lead_sessions"] = int(lead_sessions)
    if labelled.duplicated(["decision_at", "factor"]).any():
        raise ValueError("lead labels must be unique by issue date and factor")
    return labelled.reset_index(drop=True)


def attach_lead_features(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    sessions: Sequence[object],
) -> pd.DataFrame:
    """Attach the latest weekly state vintage available by each alert date."""

    required_labels = {
        "decision_at",
        "target_anchor_at",
        "factor",
        "label_start_at",
        "label_end_at",
        "threshold_available_at",
        "lead_sessions",
    }
    missing_labels = required_labels - set(labels.columns)
    if missing_labels:
        raise KeyError(f"missing lead label columns: {sorted(missing_labels)}")
    if not {"decision_at", "factor"}.issubset(features.columns):
        raise KeyError("features require decision_at and factor")
    if features.duplicated(["decision_at", "factor"]).any():
        raise ValueError("features must be unique by decision date and factor")
    leads = labels["lead_sessions"].drop_duplicates()
    if len(leads) != 1:
        raise ValueError("one lead horizon is required per feature attachment")
    lead_sessions = int(leads.iloc[0])

    left = labels.copy()
    left["decision_at"] = pd.to_datetime(
        left["decision_at"], errors="raise"
    ).dt.normalize()
    right = features.copy().rename(columns={"decision_at": "feature_source_at"})
    right["feature_source_at"] = pd.to_datetime(
        right["feature_source_at"], errors="raise"
    ).dt.normalize()

    parts: list[pd.DataFrame] = []
    feature_factors = set(right["factor"].astype(str))
    for factor, group in left.groupby("factor", sort=True):
        if str(factor) not in feature_factors:
            raise ValueError(f"no feature vintages exist for factor {factor!r}")
        factor_features = right.loc[right["factor"].eq(factor)].drop(columns="factor")
        merged = pd.merge_asof(
            group.sort_values("decision_at"),
            factor_features.sort_values("feature_source_at"),
            left_on="decision_at",
            right_on="feature_source_at",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["factor"] = factor
        parts.append(merged)
    panel = pd.concat(parts, ignore_index=True)
    if panel["feature_source_at"].isna().any():
        bad = panel.loc[panel["feature_source_at"].isna(), ["decision_at", "factor"]]
        examples = bad.head().to_dict("records")
        raise ValueError(f"no feature vintage exists by issue date: {examples}")
    panel["feature_available_at"] = panel["feature_source_at"]
    assert_lead_time_information_cutoff(
        panel,
        sessions=sessions,
        lead_sessions=lead_sessions,
        target_anchor_col="target_anchor_at",
        information_cutoff_col="decision_at",
        available_at_cols=("feature_available_at", "threshold_available_at"),
    )
    if (panel["feature_source_at"] > panel["decision_at"]).any():
        raise AssertionError("lead feature attachment selected a future vintage")
    calendar = pd.DatetimeIndex(pd.to_datetime(list(sessions), errors="raise")).normalize()
    issue_positions = calendar.get_indexer(pd.DatetimeIndex(panel["decision_at"]))
    source_positions = calendar.get_indexer(pd.DatetimeIndex(panel["feature_source_at"]))
    if (issue_positions < 0).any() or (source_positions < 0).any():
        raise ValueError("issue and feature source dates must belong to the trading calendar")
    panel["feature_staleness_sessions"] = issue_positions - source_positions
    if panel.duplicated(["decision_at", "factor"]).any():
        raise ValueError("lead model panel must be unique by issue date and factor")
    return panel.sort_values(["decision_at", "factor"]).reset_index(drop=True)
