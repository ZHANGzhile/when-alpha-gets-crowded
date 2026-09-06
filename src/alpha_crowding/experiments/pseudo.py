"""Continuous within-industry rank-permutation strategy construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_crowding.factors.signals import assign_legs


PSEUDO_ALGORITHM_VERSION = "within_industry_rank_permutation_v1"


@dataclass(frozen=True)
class ContinuousPseudoMembershipResult:
    memberships: pd.DataFrame
    diagnostics: pd.DataFrame


def stable_pseudo_seed(
    base_seed: int,
    *,
    pseudo_strategy_id: str,
    decision_at: object,
    factor: str,
) -> int:
    """Derive a reproducible seed for one strategy's weekly decision."""

    payload = (
        f"{int(base_seed)}|{pseudo_strategy_id}|"
        f"{pd.Timestamp(decision_at).date()}|{factor}|{PSEUDO_ALGORITHM_VERSION}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_continuous_permuted_memberships(
    candidates: pd.DataFrame,
    *,
    strategy_count: int,
    base_seed: int = 1729,
    quantile: float = 0.10,
) -> ContinuousPseudoMembershipResult:
    """Create stable pseudo strategies by permuting ranks within industry each week.

    Each ``pseudo_strategy_id`` denotes one deterministic rule across every
    decision date.  Signals are permuted only among securities in the same
    date/factor/industry cell.  Leg construction then uses the exact production
    winsorization, ranking, tie, weighting, and next-stage accounting contract.
    """

    if strategy_count < 1:
        raise ValueError("strategy_count must be positive")
    required = {"date", "factor", "code", "industry", "raw_signal"}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"missing pseudo-strategy candidate columns: {sorted(missing)}")
    if candidates.duplicated(["date", "factor", "code"]).any():
        raise ValueError("pseudo-strategy candidates must be unique by date/factor/code")
    if candidates[["date", "factor", "code", "industry", "raw_signal"]].isna().any().any():
        raise ValueError("pseudo-strategy candidate keys and signals must not be missing")

    passthrough = [
        column
        for column in (
            "lagged_liquidity",
            "lagged_float_market_cap",
            "float_market_cap",
            "amount",
            "turn",
        )
        if column in candidates.columns
    ]
    membership_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for (decision_at, factor), group in candidates.groupby(["date", "factor"], sort=True):
        frame = group[["date", "code", "industry", "raw_signal", *passthrough]].copy()
        real_signal_hash = hashlib.sha256(
            frame.sort_values("code")[["code", "raw_signal"]]
            .to_csv(index=False)
            .encode("utf-8")
        ).hexdigest()
        for number in range(strategy_count):
            pseudo_id = f"pseudo_{number:03d}"
            seed = stable_pseudo_seed(
                base_seed,
                pseudo_strategy_id=pseudo_id,
                decision_at=decision_at,
                factor=str(factor),
            )
            rng = np.random.default_rng(seed)
            permuted = frame.copy()
            permuted["permuted_signal"] = permuted.groupby(
                "industry", sort=True, group_keys=False
            )["raw_signal"].transform(
                lambda values: rng.permutation(values.to_numpy())
            )
            assigned = assign_legs(
                permuted,
                signal_col="permuted_signal",
                quantile=quantile,
            )
            legs = assigned.loc[assigned["leg"].isin(["LONG", "SHORT"])].copy()
            if legs.empty or set(legs["leg"]) != {"LONG", "SHORT"}:
                raise ValueError(
                    f"pseudo strategy lacks both legs: {decision_at}/{factor}/{pseudo_id}"
                )
            if legs.duplicated(["date", "code"]).any():
                raise AssertionError("a pseudo security appears in both legs")
            key = f"{factor}|{pseudo_id}"
            legs["original_factor"] = str(factor)
            legs["pseudo_strategy_id"] = pseudo_id
            legs["strategy_key"] = key
            legs["factor"] = key
            legs["permutation_seed"] = seed
            legs["algorithm_version"] = PSEUDO_ALGORITHM_VERSION
            membership_parts.append(legs)
            counts = legs.groupby("leg").size()
            diagnostics.append(
                {
                    "decision_at": pd.Timestamp(decision_at).normalize(),
                    "original_factor": str(factor),
                    "pseudo_strategy_id": pseudo_id,
                    "strategy_key": key,
                    "permutation_seed": seed,
                    "candidate_count": len(frame),
                    "long_count": int(counts.get("LONG", 0)),
                    "short_count": int(counts.get("SHORT", 0)),
                    "long_short_overlap": 0,
                    "real_signal_sha256": real_signal_hash,
                    "algorithm_version": PSEUDO_ALGORITHM_VERSION,
                    "valid": True,
                    "invalid_reason": None,
                }
            )
    memberships = pd.concat(membership_parts, ignore_index=True)
    sums = memberships.groupby(["date", "strategy_key", "leg"])["weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("pseudo leg weights do not sum to one")
    return ContinuousPseudoMembershipResult(
        memberships=memberships.sort_values(
            ["date", "strategy_key", "leg", "code"]
        ).reset_index(drop=True),
        diagnostics=pd.DataFrame(diagnostics).sort_values(
            ["decision_at", "original_factor", "pseudo_strategy_id"]
        ).reset_index(drop=True),
    )
