"""Structural measurements for continuous full-pipeline pseudo strategies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .convergence import pairwise_jaccard
from .core import compute_generic_risk, compute_stress_trigger
from .statistics import industry_residual_returns, shrunk_correlation_features


def leave_one_out_placebo_adjustment(
    features: pd.DataFrame,
    *,
    value_col: str = "actual",
    group_cols: Sequence[str] = ("decision_at", "original_factor", "leg", "feature"),
    identity_col: str = "pseudo_strategy_id",
    minimum_comparators: int = 80,
) -> pd.DataFrame:
    """Compare every pseudo strategy with all other same-date pseudo strategies."""

    if minimum_comparators < 2:
        raise ValueError("minimum_comparators must be at least two")
    required = {*group_cols, identity_col, value_col}
    missing = required - set(features.columns)
    if missing:
        raise KeyError(f"missing leave-one-out columns: {sorted(missing)}")
    if features.duplicated([*group_cols, identity_col]).any():
        raise ValueError("leave-one-out feature identities must be unique within groups")

    result = features.copy()
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce")
    mean_values = np.full(len(result), np.nan)
    std_values = np.full(len(result), np.nan)
    excess_values = np.full(len(result), np.nan)
    z_values = np.full(len(result), np.nan)
    comparator_counts = np.zeros(len(result), dtype=np.int64)
    valid_values = np.zeros(len(result), dtype=bool)
    reasons = np.full(len(result), "insufficient_comparators", dtype=object)

    grouped_positions = result.groupby(
        list(group_cols), sort=False, dropna=False
    ).indices.values()
    all_values = result[value_col].to_numpy(dtype=float)
    for raw_positions in grouped_positions:
        positions = np.asarray(raw_positions, dtype=int)
        for current in positions:
            others = positions[positions != current]
            other_values = all_values[others]
            other_values = other_values[np.isfinite(other_values)]
            comparator_counts[current] = len(other_values)
            if not np.isfinite(all_values[current]):
                reasons[current] = "invalid_current_value"
                continue
            if len(other_values) < minimum_comparators:
                continue
            mean = float(np.mean(other_values))
            std = float(np.std(other_values, ddof=1))
            mean_values[current] = mean
            std_values[current] = std
            excess_values[current] = all_values[current] - mean
            if not np.isfinite(std) or std <= 0:
                reasons[current] = "zero_or_undefined_comparator_std"
                continue
            z_values[current] = excess_values[current] / std
            valid_values[current] = True
            reasons[current] = None

    result["placebo_mean"] = mean_values
    result["placebo_std"] = std_values
    result["excess"] = excess_values
    result["placebo_z"] = z_values
    result["valid_comparators"] = comparator_counts
    result["placebo_valid"] = valid_values
    result["invalid_reason"] = reasons
    return result


def measure_continuous_pseudo_structure(
    daily_returns: pd.DataFrame,
    candidates: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    decision_at: object,
    minimum_days: int = 40,
    minimum_assets: int = 10,
) -> pd.DataFrame:
    """Measure Residual Sync and Eigen Concentration for all pseudo legs at one date."""

    required_daily = {"date", "code", "daily_return"}
    required_candidates = {"factor", "code", "industry"}
    required_memberships = {"original_factor", "pseudo_strategy_id", "leg", "code"}
    for name, required, frame in (
        ("daily", required_daily, daily_returns),
        ("candidates", required_candidates, candidates),
        ("memberships", required_memberships, memberships),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} pseudo-structure columns: {sorted(missing)}")
    decision = pd.Timestamp(decision_at).normalize()
    rows: list[dict[str, object]] = []
    for factor, factor_candidates in candidates.groupby("factor", sort=True):
        candidate_frame = factor_candidates[["code", "industry"]].drop_duplicates("code")
        candidate_frame["code"] = candidate_frame["code"].astype(str)
        panel = daily_returns[["date", "code", "daily_return"]].copy()
        panel["code"] = panel["code"].astype(str)
        panel = panel.loc[panel["code"].isin(candidate_frame["code"].astype(str))]
        panel = panel.merge(candidate_frame, on="code", validate="many_to_one")
        residuals = industry_residual_returns(panel)
        matrix = residuals.pivot(
            index="date", columns="code", values="industry_residual_return"
        )
        selected = memberships.loc[memberships["original_factor"].eq(str(factor))]
        for (pseudo_id, leg), group in selected.groupby(
            ["pseudo_strategy_id", "leg"], sort=True
        ):
            codes = sorted(group["code"].astype(str).unique())
            reason = None
            try:
                absent = set(codes) - set(matrix.columns.astype(str))
                if absent:
                    raise ValueError(f"return window lacks pseudo assets: {sorted(absent)[:5]}")
                measured = shrunk_correlation_features(
                    matrix.loc[:, codes],
                    minimum_days=minimum_days,
                    minimum_assets=minimum_assets,
                )
                values = {
                    "residual_sync": measured.residual_sync,
                    "eigen_concentration": measured.eigen_concentration,
                }
                valid_days = measured.valid_days
                valid_assets = measured.valid_assets
            except ValueError as exc:
                reason = f"{type(exc).__name__}: {exc}"
                values = {"residual_sync": np.nan, "eigen_concentration": np.nan}
                valid_days = 0
                valid_assets = 0
            for feature, value in values.items():
                rows.append(
                    {
                        "decision_at": decision,
                        "original_factor": str(factor),
                        "pseudo_strategy_id": str(pseudo_id),
                        "leg": str(leg),
                        "feature": feature,
                        "actual": value,
                        "valid_days": valid_days,
                        "valid_assets": valid_assets,
                        "measurement_valid": reason is None,
                        "measurement_invalid_reason": reason,
                    }
                )
    return pd.DataFrame(rows)


def measure_continuous_pseudo_convergence(
    memberships: pd.DataFrame,
    *,
    decision_at: object,
) -> pd.DataFrame:
    """Measure same-ID cross-factor overlap for every pseudo strategy and leg."""

    required = {"original_factor", "pseudo_strategy_id", "leg", "code"}
    missing = required - set(memberships.columns)
    if missing:
        raise KeyError(f"missing pseudo-convergence columns: {sorted(missing)}")
    decision = pd.Timestamp(decision_at).normalize()
    rows: list[dict[str, object]] = []
    for (pseudo_id, leg), group in memberships.groupby(
        ["pseudo_strategy_id", "leg"], sort=True
    ):
        member_sets = {
            str(factor): set(factor_rows["code"].astype(str))
            for factor, factor_rows in group.groupby("original_factor", sort=True)
        }
        edges = pairwise_jaccard(member_sets)
        counts: dict[str, int] = {}
        for members in member_sets.values():
            for code in members:
                counts[code] = counts.get(code, 0) + 1
        for factor, members in sorted(member_sets.items()):
            incident = edges.loc[
                edges["factor_left"].eq(factor) | edges["factor_right"].eq(factor)
            ]
            rows.append(
                {
                    "decision_at": decision,
                    "original_factor": factor,
                    "pseudo_strategy_id": str(pseudo_id),
                    "leg": str(leg),
                    "feature": "strategy_convergence",
                    "actual": float(incident["jaccard"].mean()),
                    "maximum_edge_weight": float(incident["jaccard"].max()),
                    "mean_stock_crowding_count": float(
                        np.mean([counts[code] for code in members])
                    ),
                    "factor_count": len(member_sets),
                    "measurement_valid": True,
                    "measurement_invalid_reason": None,
                }
            )
    return pd.DataFrame(rows)


def assemble_continuous_pseudo_state(
    crowding: pd.DataFrame,
    structural: pd.DataFrame,
    risk: pd.DataFrame,
    return_stress: pd.DataFrame,
) -> pd.DataFrame:
    """Join comparable pseudo C/G/S inputs and apply the production aggregators."""

    keys = ["decision_at", "original_factor", "pseudo_strategy_id", "leg"]
    for name, frame in (
        ("crowding", crowding),
        ("risk", risk),
        ("return_stress", return_stress),
    ):
        missing = set(keys) - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} state keys: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} state keys must be unique")
    structural_required = {*keys, "feature", "actual_historical_z"}
    missing = structural_required - set(structural.columns)
    if missing:
        raise KeyError(f"missing structural state fields: {sorted(missing)}")
    if structural.duplicated([*keys, "feature"]).any():
        raise ValueError("structural state keys must be unique")

    raw = structural.loc[
        structural["feature"].isin(["residual_sync", "eigen_concentration"])
    ].pivot(index=keys, columns="feature", values="actual_historical_z").rename(
        columns={"residual_sync": "raw_sync_z", "eigen_concentration": "raw_eigen_z"}
    ).reset_index()
    required_raw = {"raw_sync_z", "raw_eigen_z"}
    if required_raw - set(raw.columns):
        raise ValueError("pseudo structural inputs lack raw sync or eigen history")
    panel = crowding.merge(raw, on=keys, how="outer", validate="one_to_one")
    panel = panel.merge(risk, on=keys, how="outer", validate="one_to_one")
    return_columns = keys + [
        "factor_return_recent",
        "factor_return_shock",
        "factor_return_history_n",
    ]
    missing_return = set(return_columns) - set(return_stress.columns)
    if missing_return:
        raise KeyError(f"missing return-stress fields: {sorted(missing_return)}")
    panel = panel.merge(
        return_stress[return_columns],
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    panel["turnover_level_z"] = panel["turnover_level_historical_z"]
    panel["illiquidity_level_z"] = panel["illiquidity_level_historical_z"]
    panel["turnover_shock_z"] = panel["turnover_shock_historical_z"]
    panel["turnover_sync_z"] = panel["turnover_sync_historical_z"]
    panel["liquidity_stress_z"] = panel["liquidity_shock_historical_z"]
    panel["factor_return_shock_z"] = panel["factor_return_shock"]
    generic = compute_generic_risk(panel)
    stress = compute_stress_trigger(panel)
    return pd.concat([panel, generic, stress], axis=1).sort_values(keys).reset_index(drop=True)
