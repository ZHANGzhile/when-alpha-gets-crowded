"""Point-in-time contract for a tradable benchmark replication basket."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_benchmark_replication_weights(
    weights: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Validate historical target weights without inventing an equal-weight CSI800."""

    required = ["decision_at", "available_at", "code", "weight"]
    missing = set(required) - set(weights.columns)
    if missing:
        raise KeyError(f"missing benchmark-weight fields: {sorted(missing)}")
    result = weights[required].copy()
    result["decision_at"] = pd.to_datetime(result["decision_at"], errors="raise").dt.normalize()
    result["available_at"] = pd.to_datetime(result["available_at"], errors="raise").dt.normalize()
    result["code"] = result["code"].astype(str)
    result["weight"] = pd.to_numeric(result["weight"], errors="raise")
    if result[required].isna().any().any():
        raise ValueError("benchmark-weight fields must not be missing")
    if result.duplicated(["decision_at", "code"]).any():
        raise ValueError("benchmark weights must be unique by decision_at/code")
    if (result["available_at"] > result["decision_at"]).any():
        raise ValueError("benchmark weights were not available by decision_at")
    if not np.isfinite(result["weight"]).all() or (result["weight"] < 0).any():
        raise ValueError("benchmark weights must be finite and non-negative")
    sums = result.groupby("decision_at")["weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError("benchmark weights must sum to one on every decision date")
    return result.sort_values(["decision_at", "code"]).reset_index(drop=True)
