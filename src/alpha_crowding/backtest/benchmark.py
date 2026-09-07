"""Point-in-time contract for a tradable benchmark replication basket."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_drifted_benchmark_weights(
    anchor_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    *,
    anchor_sum_tolerance: float = 0.02,
) -> pd.DataFrame:
    """Drift monthly closing-weight anchors to weekly closing decisions."""

    anchor_required = {"weight_date", "available_at", "code", "weight"}
    return_required = {"date", "code", "daily_return"}
    for name, frame, required in (
        ("anchor", anchor_weights, anchor_required),
        ("return", daily_returns, return_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} fields: {sorted(missing)}")
    anchors = anchor_weights[list(anchor_required)].copy()
    anchors["weight_date"] = pd.to_datetime(
        anchors["weight_date"], errors="raise"
    ).dt.normalize()
    anchors["available_at"] = pd.to_datetime(
        anchors["available_at"], errors="raise"
    ).dt.normalize()
    anchors["code"] = anchors["code"].astype(str)
    anchors["weight"] = pd.to_numeric(anchors["weight"], errors="raise")
    if anchors.duplicated(["weight_date", "code"]).any():
        raise ValueError("anchor weights contain duplicate keys")
    if (anchors["available_at"] > anchors["weight_date"]).any():
        raise ValueError("anchor weights were not available by their weight date")
    if not np.isfinite(anchors["weight"]).all() or (anchors["weight"] < 0.0).any():
        raise ValueError("anchor weights must be finite and non-negative")
    sums = anchors.groupby("weight_date")["weight"].sum()
    if (sums.sub(1.0).abs() > anchor_sum_tolerance).any():
        raise ValueError("rounded anchor weights are too far from full investment")
    anchors["weight"] /= anchors.groupby("weight_date")["weight"].transform("sum")

    returns = daily_returns[list(return_required)].copy()
    returns["date"] = pd.to_datetime(returns["date"], errors="raise").dt.normalize()
    returns["code"] = returns["code"].astype(str)
    returns["daily_return"] = pd.to_numeric(
        returns["daily_return"], errors="coerce"
    )
    if returns.duplicated(["date", "code"]).any():
        raise ValueError("daily returns contain duplicate keys")
    lookup = returns.set_index(["date", "code"])["daily_return"]
    dates = pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize()
    dates = dates.drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("decision_dates must not be empty")

    rows = []
    anchor_dates = pd.DatetimeIndex(anchors["weight_date"].unique()).sort_values()
    market_dates = pd.DatetimeIndex(returns["date"].unique()).sort_values()
    for decision_at in dates:
        eligible = anchor_dates[anchor_dates <= decision_at]
        if len(eligible) == 0:
            raise ValueError(f"no benchmark anchor exists by {decision_at.date()}")
        anchor_date = eligible[-1]
        anchor = anchors.loc[anchors["weight_date"].eq(anchor_date)]
        weights = dict(zip(anchor["code"], anchor["weight"]))
        drift_dates = market_dates[(market_dates > anchor_date) & (market_dates <= decision_at)]
        for date in drift_dates:
            keys = pd.MultiIndex.from_tuples([(date, code) for code in weights])
            daily = lookup.reindex(keys)
            if daily.isna().any():
                missing_codes = [
                    code for code, value in zip(weights, daily) if pd.isna(value)
                ]
                raise ValueError(
                    f"benchmark holdings lack returns at {date.date()}: "
                    f"{missing_codes[:10]}"
                )
            notionals = {
                code: weights[code] * (1.0 + float(value))
                for code, value in zip(weights, daily.to_numpy())
            }
            total = sum(notionals.values())
            if total <= 0.0:
                raise ValueError("benchmark replication value became non-positive")
            weights = {code: value / total for code, value in notionals.items()}
        available_at = anchor["available_at"].max()
        rows.extend(
            {
                "decision_at": decision_at,
                "available_at": available_at,
                "anchor_date": anchor_date,
                "code": code,
                "weight": weight,
            }
            for code, weight in weights.items()
        )
    return validate_benchmark_replication_weights(pd.DataFrame(rows))


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
    result = weights.copy()
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
