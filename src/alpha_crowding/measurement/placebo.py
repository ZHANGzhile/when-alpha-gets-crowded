"""Same-date matched placebo portfolios for structural measurements."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MatchedPlaceboResult:
    memberships: pd.DataFrame
    diagnostics: pd.DataFrame


def stable_group_seed(
    base_seed: int, *, decision_at: object, factor: str, leg: str
) -> int:
    """Derive a stable per-group RNG seed without Python's randomized hash."""

    payload = f"{int(base_seed)}|{pd.Timestamp(decision_at).date()}|{factor}|{leg}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _standardized_log_liquidity(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    logged = np.log(numeric.where(numeric > 0))
    center = float(logged.median())
    scale = float((logged - center).abs().median())
    if not np.isfinite(scale) or scale <= 0:
        scale = float(logged.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (logged - center) / scale


def _weighted_choice(
    rng: np.random.Generator,
    candidates: pd.DataFrame,
    target_z: float,
    *,
    temperature: float,
) -> Hashable:
    distances = (candidates["_liquidity_z"] - target_z).abs().to_numpy(dtype=float)
    weights = np.exp(-distances / temperature)
    weights = weights / weights.sum()
    return rng.choice(candidates.index.to_numpy(), p=weights)


def draw_matched_placebos(
    candidates: pd.DataFrame,
    actual_codes: Sequence[str],
    *,
    draws: int = 100,
    seed: int = 42,
    industry_col: str = "industry",
    liquidity_col: str = "lagged_liquidity",
    exclude_actual: bool = False,
    temperature: float = 0.75,
    maximum_mean_absolute_z_difference: float = 0.35,
    maximum_attempts_per_draw: int = 50,
) -> MatchedPlaceboResult:
    """Generate industry-exact, liquidity-matched portfolios without replacement."""

    required = {"code", industry_col, liquidity_col}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"missing placebo candidate fields: {sorted(missing)}")
    if candidates["code"].duplicated().any():
        raise ValueError("placebo candidates must be unique by code")
    if draws < 1 or maximum_attempts_per_draw < 1:
        raise ValueError("draws and maximum_attempts_per_draw must be positive")
    if temperature <= 0 or maximum_mean_absolute_z_difference < 0:
        raise ValueError("temperature must be positive and balance threshold non-negative")
    codes = tuple(str(code) for code in actual_codes)
    if not codes or len(set(codes)) != len(codes):
        raise ValueError("actual_codes must be nonempty and unique")
    indexed = candidates.copy().set_index("code", drop=False)
    absent = set(codes) - set(indexed.index)
    if absent:
        raise ValueError(f"actual members absent from candidates: {sorted(absent)}")
    if indexed.loc[list(codes), industry_col].isna().any():
        raise ValueError("actual members require point-in-time industry values")
    eligible = indexed[indexed[industry_col].notna()].copy()
    eligible["_liquidity_z"] = eligible.groupby(industry_col, group_keys=False)[
        liquidity_col
    ].transform(_standardized_log_liquidity)
    eligible = eligible[eligible["_liquidity_z"].notna()].copy()
    if exclude_actual:
        eligible = eligible[~eligible["code"].isin(codes)]
    actual = indexed.loc[list(codes)].copy()
    actual["_liquidity_z"] = actual.groupby(industry_col, group_keys=False)[
        liquidity_col
    ].transform(_standardized_log_liquidity)
    # Use candidate-universe industry scales for both actual and sampled stocks.
    for industry, rows in actual.groupby(industry_col):
        universe = indexed[indexed[industry_col].eq(industry)]
        universe_z = _standardized_log_liquidity(universe[liquidity_col])
        actual.loc[rows.index, "_liquidity_z"] = universe_z.reindex(rows.index)
    if actual["_liquidity_z"].isna().any():
        raise ValueError("actual members require positive, variable lagged liquidity")
    needed = actual.groupby(industry_col).size()
    available = eligible.groupby(industry_col).size()
    shortages = {
        str(industry): int(count - available.get(industry, 0))
        for industry, count in needed.items()
        if available.get(industry, 0) < count
    }
    if shortages:
        raise ValueError(f"insufficient industry-matched candidates: {shortages}")

    rng = np.random.default_rng(seed)
    membership_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for draw_number in range(draws):
        accepted_rows = None
        accepted_balance = float("nan")
        for attempt in range(1, maximum_attempts_per_draw + 1):
            selected_rows = []
            for industry, actual_group in actual.groupby(industry_col, sort=True):
                pool = eligible[eligible[industry_col].eq(industry)].copy()
                order = rng.permutation(actual_group.index.to_numpy())
                for actual_code in order:
                    target_z = float(actual_group.loc[actual_code, "_liquidity_z"])
                    chosen_index = _weighted_choice(
                        rng, pool, target_z, temperature=temperature
                    )
                    chosen = pool.loc[chosen_index]
                    selected_rows.append(
                        {
                            "code": str(chosen["code"]),
                            "industry": industry,
                            "matched_to_code": str(actual_code),
                            "liquidity_z": float(chosen["_liquidity_z"]),
                            "actual_liquidity_z": target_z,
                            "match_distance": abs(float(chosen["_liquidity_z"]) - target_z),
                        }
                    )
                    pool = pool.drop(index=chosen_index)
            draw_frame = pd.DataFrame(selected_rows)
            balance = float(
                (draw_frame["liquidity_z"] - draw_frame["actual_liquidity_z"])
                .abs()
                .mean()
            )
            if balance <= maximum_mean_absolute_z_difference:
                accepted_rows = draw_frame
                accepted_balance = balance
                break
        draw_id = f"draw_{draw_number:03d}"
        valid = accepted_rows is not None
        diagnostic_rows.append(
            {
                "draw_id": draw_id,
                "seed": seed,
                "attempts": attempt,
                "valid": valid,
                "member_count": 0 if accepted_rows is None else len(accepted_rows),
                "mean_absolute_liquidity_z_difference": accepted_balance,
                "invalid_reason": None if valid else "liquidity_balance_threshold",
            }
        )
        if valid:
            accepted_rows = accepted_rows.copy()
            accepted_rows.insert(0, "draw_id", draw_id)
            membership_rows.extend(accepted_rows.to_dict("records"))
    return MatchedPlaceboResult(
        memberships=pd.DataFrame(membership_rows),
        diagnostics=pd.DataFrame(diagnostic_rows),
    )
