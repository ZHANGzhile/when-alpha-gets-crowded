"""Falsification diagnostics for time alignment and construct validity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloRankResult:
    observed: float
    placebo_mean: float
    placebo_std: float
    percentile_midrank: float
    probability_placebo_at_least_observed: float
    valid_placebos: int


def monte_carlo_incremental_rank(
    observed: float,
    placebo_values: Sequence[float],
    *,
    minimum_placebos: int = 80,
) -> MonteCarloRankResult:
    """Rank a real incremental value against continuous-strategy placebos."""

    values = np.asarray(placebo_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(observed):
        raise ValueError("observed incremental value must be finite")
    if len(values) < minimum_placebos:
        raise ValueError(
            f"at least {minimum_placebos} finite placebo values are required; found {len(values)}"
        )
    less = int(np.sum(values < observed))
    equal = int(np.sum(values == observed))
    return MonteCarloRankResult(
        observed=float(observed),
        placebo_mean=float(np.mean(values)),
        placebo_std=float(np.std(values, ddof=1)),
        percentile_midrank=float((less + 0.5 * equal) / len(values)),
        probability_placebo_at_least_observed=float(
            (1 + np.sum(values >= observed)) / (len(values) + 1)
        ),
        valid_placebos=len(values),
    )


def time_misalign_features(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    lag_weeks: int,
    group_cols: Sequence[str] = ("factor",),
    date_col: str = "decision_at",
) -> pd.DataFrame:
    """Replace selected fields with their exact earlier within-group row vintage.

    The function retains the current outcome and all nonselected controls.  It
    records the source date so the falsification can prove that the shifted
    Crowding fields came from exactly ``lag_weeks`` earlier observations.
    """

    if lag_weeks < 1:
        raise ValueError("lag_weeks must be positive")
    selected = tuple(columns)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("columns must be nonempty and unique")
    required = {date_col, *group_cols, *selected}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"missing time-misalignment columns: {sorted(missing)}")
    if frame.duplicated([*group_cols, date_col]).any():
        raise ValueError("time-misalignment panel keys must be unique")

    result = frame.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="raise").dt.normalize()
    result = result.sort_values([*group_cols, date_col]).reset_index(drop=True)
    grouped = result.groupby(list(group_cols), sort=False, dropna=False)
    source_dates = grouped[date_col].shift(lag_weeks)
    shifted = grouped[list(selected)].shift(lag_weeks)
    for column in selected:
        result[column] = shifted[column]
    result["misaligned_source_at"] = source_dates
    result["misalignment_weeks"] = int(lag_weeks)
    available = result["misaligned_source_at"].notna()
    if (result.loc[available, "misaligned_source_at"] >= result.loc[available, date_col]).any():
        raise AssertionError("time-misaligned features must be strictly historical")
    return result


def discriminant_validity_table(
    frame: pd.DataFrame,
    crowding_columns: Sequence[str],
    comparator_columns: Sequence[str],
    *,
    minimum_observations: int = 30,
) -> pd.DataFrame:
    """Report pairwise Pearson/Spearman association and univariate R-squared."""

    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")
    requested = {*crowding_columns, *comparator_columns}
    missing = requested - set(frame.columns)
    if missing:
        raise KeyError(f"missing discriminant-validity columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for crowding in crowding_columns:
        for comparator in comparator_columns:
            pair = frame[[crowding, comparator]].apply(pd.to_numeric, errors="coerce")
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            n = len(pair)
            pearson = spearman = slope = intercept = r_squared = np.nan
            valid = n >= minimum_observations
            reason = None
            if valid:
                x = pair[comparator].to_numpy(dtype=float)
                y = pair[crowding].to_numpy(dtype=float)
                if np.std(x, ddof=1) <= 0 or np.std(y, ddof=1) <= 0:
                    valid = False
                    reason = "zero_variance"
                else:
                    pearson = float(np.corrcoef(x, y)[0, 1])
                    spearman = float(pair[comparator].corr(pair[crowding], method="spearman"))
                    design = np.column_stack([np.ones(n), x])
                    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
                    fitted = intercept + slope * x
                    denominator = float(np.sum((y - y.mean()) ** 2))
                    r_squared = float(1.0 - np.sum((y - fitted) ** 2) / denominator)
            elif reason is None:
                reason = "insufficient_observations"
            rows.append(
                {
                    "crowding_variable": crowding,
                    "comparator_variable": comparator,
                    "observations": n,
                    "pearson": pearson,
                    "spearman": spearman,
                    "intercept": float(intercept),
                    "slope": float(slope),
                    "r_squared": r_squared,
                    "valid": valid,
                    "invalid_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def multivariate_residual_diagnostic(
    frame: pd.DataFrame,
    specifications: Mapping[str, Sequence[str]],
    *,
    minimum_observations: int = 50,
) -> pd.DataFrame:
    """Regress each crowding proxy on generic-risk comparators and report residual scale."""

    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")
    rows: list[dict[str, object]] = []
    for crowding, comparators_raw in specifications.items():
        comparators = list(comparators_raw)
        if not comparators:
            raise ValueError(f"{crowding!r} requires at least one comparator")
        columns = [crowding, *comparators]
        missing = set(columns) - set(frame.columns)
        if missing:
            raise KeyError(f"missing residual-diagnostic columns: {sorted(missing)}")
        sample = frame[columns].apply(pd.to_numeric, errors="coerce")
        sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
        n = len(sample)
        valid = n >= max(minimum_observations, len(comparators) + 2)
        reason = None if valid else "insufficient_observations"
        r_squared = adjusted = residual_std = np.nan
        rank = 0
        if valid:
            y = sample[crowding].to_numpy(dtype=float)
            x = sample[comparators].to_numpy(dtype=float)
            scales = x.std(axis=0, ddof=1)
            if np.any(scales <= 0) or np.std(y, ddof=1) <= 0:
                valid = False
                reason = "zero_variance"
            else:
                x = (x - x.mean(axis=0)) / scales
                design = np.column_stack([np.ones(n), x])
                rank = int(np.linalg.matrix_rank(design))
                if rank < design.shape[1]:
                    valid = False
                    reason = "rank_deficient_design"
                else:
                    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
                    residual = y - design @ coefficients
                    denominator = float(np.sum((y - y.mean()) ** 2))
                    r_squared = float(1.0 - np.sum(residual**2) / denominator)
                    adjusted = float(
                        1.0 - (1.0 - r_squared) * (n - 1) / (n - len(comparators) - 1)
                    )
                    residual_std = float(np.std(residual, ddof=len(comparators) + 1))
        rows.append(
            {
                "crowding_variable": crowding,
                "comparators": comparators,
                "observations": n,
                "design_rank": rank,
                "r_squared": r_squared,
                "adjusted_r_squared": adjusted,
                "residual_std": residual_std,
                "valid": valid,
                "invalid_reason": reason,
            }
        )
    return pd.DataFrame(rows)
