"""Production statistics underlying structural and generic-risk measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True, slots=True)
class CorrelationFeatures:
    residual_sync: float
    eigen_concentration: float
    valid_days: int
    valid_assets: int
    estimator: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def industry_residual_returns(
    panel: pd.DataFrame,
    *,
    return_col: str = "daily_return",
    industry_col: str = "industry",
    minimum_industry_assets: int = 2,
) -> pd.DataFrame:
    """Demean each stock return by its same-date point-in-time industry mean."""

    required = {"date", "code", return_col, industry_col}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"missing residual-return fields: {sorted(missing)}")
    if minimum_industry_assets < 2:
        raise ValueError("minimum_industry_assets must be at least two")
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    if out.duplicated(["date", "code"]).any():
        raise ValueError("daily panel must be unique by date/code")
    out[return_col] = pd.to_numeric(out[return_col], errors="coerce")
    groups = out.groupby(["date", industry_col], dropna=False)[return_col]
    out["industry_asset_count"] = groups.transform("count").astype("int64")
    out["industry_return"] = groups.transform("mean")
    valid = (
        out[industry_col].notna()
        & out[return_col].notna()
        & out["industry_asset_count"].ge(minimum_industry_assets)
    )
    out["industry_residual_return"] = (
        out[return_col] - out["industry_return"]
    ).where(valid)
    return out


def shrunk_correlation_features(
    return_matrix: pd.DataFrame,
    *,
    minimum_days: int = 40,
    minimum_assets: int = 10,
) -> CorrelationFeatures:
    """Estimate mean off-diagonal correlation and leading-eigenvalue share."""

    if not isinstance(return_matrix, pd.DataFrame):
        raise TypeError("return_matrix must be a DataFrame")
    if return_matrix.index.has_duplicates or return_matrix.columns.has_duplicates:
        raise ValueError("return matrix dates and assets must be unique")
    if minimum_days < 2 or minimum_assets < 2:
        raise ValueError("minimum_days and minimum_assets must be at least two")
    numeric = return_matrix.apply(pd.to_numeric, errors="coerce")
    eligible_columns = numeric.columns[numeric.notna().sum().ge(minimum_days)]
    complete = numeric.loc[:, eligible_columns].dropna(axis=0, how="any")
    if complete.shape[0] < minimum_days:
        raise ValueError(
            f"insufficient complete days: {complete.shape[0]} < {minimum_days}"
        )
    if complete.shape[1] < minimum_assets:
        raise ValueError(
            f"insufficient assets: {complete.shape[1]} < {minimum_assets}"
        )
    values = complete.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("return matrix contains infinite values")
    covariance = LedoitWolf(assume_centered=False).fit(values).covariance_
    standard_deviation = np.sqrt(np.diag(covariance))
    if (standard_deviation <= 0).any() or not np.isfinite(standard_deviation).all():
        raise ValueError("shrunk covariance contains a zero or invalid variance")
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    correlation = np.clip(correlation, -1.0, 1.0)
    asset_count = correlation.shape[0]
    residual_sync = float(
        (correlation.sum() - np.trace(correlation)) / (asset_count * (asset_count - 1))
    )
    eigenvalues = np.linalg.eigvalsh(correlation)
    trace = float(eigenvalues.sum())
    if trace <= 0:
        raise ValueError("correlation matrix has nonpositive trace")
    return CorrelationFeatures(
        residual_sync=residual_sync,
        eigen_concentration=float(eigenvalues[-1] / trace),
        valid_days=int(complete.shape[0]),
        valid_assets=int(complete.shape[1]),
        estimator="ledoit_wolf_covariance_to_correlation",
    )


def amihud_illiquidity(
    daily_return: pd.Series, amount: pd.Series
) -> pd.Series:
    """Compute abs(return)/amount while preserving zero-amount invalidity."""

    returns = pd.to_numeric(daily_return, errors="coerce")
    traded_amount = pd.to_numeric(amount, errors="coerce")
    result = returns.abs() / traded_amount.where(traded_amount > 0)
    return result.where(np.isfinite(result))


def robust_past_shock(
    history: pd.Series,
    current: float,
    *,
    minimum_history: int,
) -> float:
    """Standardize current value against a past-only median/raw-MAD baseline."""

    past = pd.to_numeric(history, errors="coerce").dropna().astype(float)
    if len(past) < minimum_history:
        return float("nan")
    center = float(past.median())
    mad = float((past - center).abs().median())
    value = float(current)
    if not np.isfinite(value) or not np.isfinite(mad) or mad <= 0:
        return float("nan")
    return (value - center) / mad
