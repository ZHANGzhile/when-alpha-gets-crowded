"""Cross-factor strategy-convergence measurements with joint placebo draws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import summarize_matched_placebo
from .placebo import draw_matched_placebos, stable_group_seed
from .structural import candidate_snapshot_sha256


CONVERGENCE_ALGORITHM_VERSION = "joint_strategy_convergence_v1"


@dataclass(frozen=True, slots=True)
class ConvergencePlaceboResult:
    features: pd.DataFrame
    edges: pd.DataFrame
    placebo_features: pd.DataFrame
    placebo_diagnostics: pd.DataFrame
    placebo_memberships: pd.DataFrame


def pairwise_jaccard(member_sets: dict[str, set[str]]) -> pd.DataFrame:
    """Return every undirected pairwise Jaccard edge in stable factor order."""

    factors = sorted(member_sets)
    if len(factors) < 2:
        raise ValueError("strategy convergence requires at least two factors")
    if any(not member_sets[factor] for factor in factors):
        raise ValueError("every factor requires a nonempty member set")
    rows = []
    for left_position, left in enumerate(factors[:-1]):
        for right in factors[left_position + 1 :]:
            intersection = member_sets[left] & member_sets[right]
            union = member_sets[left] | member_sets[right]
            rows.append(
                {
                    "factor_left": left,
                    "factor_right": right,
                    "jaccard": len(intersection) / len(union),
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                }
            )
    return pd.DataFrame(rows)


def _factor_statistics(
    member_sets: dict[str, set[str]], edges: pd.DataFrame
) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for members in member_sets.values():
        for code in members:
            counts[code] = counts.get(code, 0) + 1
    rows = []
    for factor, members in sorted(member_sets.items()):
        incident = edges[
            edges["factor_left"].eq(factor) | edges["factor_right"].eq(factor)
        ]
        rows.append(
            {
                "factor": factor,
                "strategy_convergence": float(incident["jaccard"].mean()),
                "maximum_edge_weight": float(incident["jaccard"].max()),
                "mean_stock_crowding_count": float(
                    np.mean([counts[code] for code in members])
                ),
                "member_count": len(members),
            }
        )
    return pd.DataFrame(rows)


def measure_strategy_convergence_placebos(
    date_memberships: pd.DataFrame,
    *,
    decision_at: object,
    leg: str,
    draws: int = 100,
    minimum_valid_draws: int = 80,
    base_seed: int = 42,
    temperature: float = 0.75,
    maximum_mean_absolute_z_difference: float = 0.35,
    maximum_attempts_per_draw: int = 50,
    exclude_actual: bool = False,
) -> ConvergencePlaceboResult:
    """Measure factor overlap against jointly generated matched portfolios."""

    required = {"factor", "code", "industry", "lagged_liquidity", "leg"}
    missing = required - set(date_memberships.columns)
    if missing:
        raise KeyError(f"missing convergence fields: {sorted(missing)}")
    decision = pd.Timestamp(decision_at).normalize()
    factors = sorted(date_memberships["factor"].astype(str).unique())
    if len(factors) < 2:
        raise ValueError("strategy convergence requires at least two factors")

    actual_sets: dict[str, set[str]] = {}
    sampled_memberships = []
    sampled_diagnostics = []
    snapshot_hashes = {}
    for factor in factors:
        candidates = date_memberships[
            date_memberships["factor"].astype(str).eq(factor)
        ][["code", "industry", "lagged_liquidity"]].copy()
        actual = set(
            date_memberships.loc[
                date_memberships["factor"].astype(str).eq(factor)
                & date_memberships["leg"].eq(leg),
                "code",
            ].astype(str)
        )
        if not actual:
            raise ValueError(f"factor {factor} has no {leg} members")
        actual_sets[factor] = actual
        snapshot_hashes[factor] = candidate_snapshot_sha256(candidates)
        seed = stable_group_seed(
            base_seed, decision_at=decision, factor=factor, leg=leg
        )
        sampled = draw_matched_placebos(
            candidates,
            sorted(actual),
            draws=draws,
            seed=seed,
            exclude_actual=exclude_actual,
            temperature=temperature,
            maximum_mean_absolute_z_difference=maximum_mean_absolute_z_difference,
            maximum_attempts_per_draw=maximum_attempts_per_draw,
        )
        members = sampled.memberships.copy()
        members.insert(0, "decision_at", decision)
        members.insert(1, "leg", leg)
        members.insert(2, "factor", factor)
        sampled_memberships.append(members)
        diagnostics = sampled.diagnostics.copy()
        diagnostics.insert(0, "decision_at", decision)
        diagnostics.insert(1, "leg", leg)
        diagnostics.insert(2, "factor", factor)
        diagnostics["group_seed"] = seed
        diagnostics["candidate_snapshot_sha256"] = snapshot_hashes[factor]
        diagnostics["algorithm_version"] = CONVERGENCE_ALGORITHM_VERSION
        sampled_diagnostics.append(diagnostics)
    memberships = pd.concat(sampled_memberships, ignore_index=True)
    diagnostics = pd.concat(sampled_diagnostics, ignore_index=True)

    actual_edges = pairwise_jaccard(actual_sets)
    actual_edges.insert(0, "decision_at", decision)
    actual_edges.insert(1, "leg", leg)
    actual_statistics = _factor_statistics(actual_sets, actual_edges)

    placebo_rows = []
    for draw_number in range(draws):
        draw_id = f"draw_{draw_number:03d}"
        joint_sets = {
            factor: set(
                memberships.loc[
                    memberships["factor"].eq(factor)
                    & memberships["draw_id"].eq(draw_id),
                    "code",
                ].astype(str)
            )
            for factor in factors
        }
        if any(not members for members in joint_sets.values()):
            for factor in factors:
                placebo_rows.append(
                    {
                        "decision_at": decision,
                        "leg": leg,
                        "factor": factor,
                        "draw_id": draw_id,
                        "value": np.nan,
                        "maximum_edge_weight": np.nan,
                        "mean_stock_crowding_count": np.nan,
                        "joint_valid": False,
                        "invalid_reason": "one_or_more_factor_draws_invalid",
                    }
                )
            continue
        draw_statistics = _factor_statistics(joint_sets, pairwise_jaccard(joint_sets))
        for row in draw_statistics.to_dict("records"):
            placebo_rows.append(
                {
                    "decision_at": decision,
                    "leg": leg,
                    "factor": row["factor"],
                    "draw_id": draw_id,
                    "value": row["strategy_convergence"],
                    "maximum_edge_weight": row["maximum_edge_weight"],
                    "mean_stock_crowding_count": row["mean_stock_crowding_count"],
                    "joint_valid": True,
                    "invalid_reason": None,
                }
            )
    placebo_features = pd.DataFrame(placebo_rows)
    feature_rows = []
    for actual in actual_statistics.to_dict("records"):
        factor = actual["factor"]
        values = placebo_features.loc[
            placebo_features["factor"].eq(factor), "value"
        ].to_numpy(dtype=float)
        summary = summarize_matched_placebo(
            actual["strategy_convergence"],
            values,
            min_valid_draws=minimum_valid_draws,
        ).to_dict()
        feature_rows.append(
            {
                "decision_at": decision,
                "leg": leg,
                "factor": factor,
                "feature": "strategy_convergence",
                **summary,
                "actual_maximum_edge_weight": actual["maximum_edge_weight"],
                "actual_mean_stock_crowding_count": actual[
                    "mean_stock_crowding_count"
                ],
                "factor_count": len(factors),
                "candidate_snapshot_sha256": snapshot_hashes[factor],
                "algorithm_version": CONVERGENCE_ALGORITHM_VERSION,
            }
        )
    return ConvergencePlaceboResult(
        features=pd.DataFrame(feature_rows),
        edges=actual_edges,
        placebo_features=placebo_features,
        placebo_diagnostics=diagnostics,
        placebo_memberships=memberships,
    )
