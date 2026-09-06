"""Stock-level target-weight and transaction-cost accounting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Hashable, Mapping


Asset = Hashable
Weights = Mapping[Asset, float]


def _as_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_long_only_weights(
    weights: Weights,
    *,
    name: str,
    tolerance: float = 1e-10,
) -> Dict[Asset, float]:
    if not weights:
        raise ValueError(f"{name} must not be empty")

    result: Dict[Asset, float] = {}
    for asset, raw_weight in weights.items():
        weight = _as_finite(raw_weight, name=f"{name}[{asset!r}]")
        if weight < -tolerance:
            raise ValueError(f"{name} must be long-only")
        result[asset] = max(0.0, weight)

    total = sum(result.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{name} must sum to 1; got {total!r}")
    # Remove harmless floating point drift while preserving the user's asset set.
    return {asset: weight / total for asset, weight in result.items()}


def combine_stock_target_weights(
    factor_long_weights: Weights,
    benchmark_weights: Weights,
    active_weight: float,
) -> Dict[Asset, float]:
    """Create ``w * FactorLong + (1-w) * Benchmark`` at stock level."""

    active = _as_finite(active_weight, name="active_weight")
    if not 0.0 <= active <= 1.0:
        raise ValueError("active_weight must be in [0, 1]")

    factor = _validated_long_only_weights(
        factor_long_weights, name="factor_long_weights"
    )
    benchmark = _validated_long_only_weights(
        benchmark_weights, name="benchmark_weights"
    )

    assets = list(factor)
    assets.extend(asset for asset in benchmark if asset not in factor)
    combined = {
        asset: active * factor.get(asset, 0.0)
        + (1.0 - active) * benchmark.get(asset, 0.0)
        for asset in assets
    }
    # The inputs and convex coefficient guarantee this, but normalize tiny drift.
    total = sum(combined.values())
    return {asset: weight / total for asset, weight in combined.items()}


def drift_weights(
    post_trade_weights: Weights,
    asset_returns: Mapping[Asset, float],
) -> Dict[Asset, float]:
    """Drift post-trade weights through asset returns to the next pre-trade state."""

    weights = _validated_long_only_weights(
        post_trade_weights, name="post_trade_weights"
    )
    notionals: Dict[Asset, float] = {}
    for asset, weight in weights.items():
        if asset not in asset_returns:
            raise KeyError(f"missing return for held asset {asset!r}")
        asset_return = _as_finite(
            asset_returns[asset], name=f"asset_returns[{asset!r}]"
        )
        if asset_return < -1.0:
            raise ValueError("asset returns must be at least -1")
        notionals[asset] = weight * (1.0 + asset_return)

    portfolio_growth = sum(notionals.values())
    if portfolio_growth <= 0.0:
        raise ValueError("portfolio value must remain positive after drift")
    return {
        asset: notional / portfolio_growth for asset, notional in notionals.items()
    }


@dataclass(frozen=True)
class RebalanceResult:
    """An auditable rebalance measured against pre-cost portfolio value."""

    pre_trade_weights: Mapping[Asset, float]
    target_weights: Mapping[Asset, float]
    signed_trades: Mapping[Asset, float]
    signed_trade_notionals: Mapping[Asset, float]
    transaction_costs_by_asset: Mapping[Asset, float]
    buy_fraction: float
    sell_fraction: float
    gross_traded_fraction: float
    half_l1_turnover: float
    gross_traded_notional: float
    transaction_cost: float
    transaction_cost_fraction: float


def calculate_rebalance(
    pre_trade_weights: Weights,
    target_weights: Weights,
    *,
    portfolio_value: float = 1.0,
    one_way_cost_bps: float = 10.0,
) -> RebalanceResult:
    """Calculate both sides of trading and charge each traded dollar once.

    ``gross_traded_fraction`` is ``sum(abs(delta_weight))``.  It is twice the
    conventional half-L1 turnover for a fully invested rebalance.  A complete
    switch between disjoint portfolios therefore costs ``2 * cost_rate``.
    """

    before = _validated_long_only_weights(
        pre_trade_weights, name="pre_trade_weights"
    )
    target = _validated_long_only_weights(target_weights, name="target_weights")
    value = _as_finite(portfolio_value, name="portfolio_value")
    cost_bps = _as_finite(one_way_cost_bps, name="one_way_cost_bps")
    if value <= 0.0:
        raise ValueError("portfolio_value must be positive")
    if cost_bps < 0.0:
        raise ValueError("one_way_cost_bps must be non-negative")

    assets = list(before)
    assets.extend(asset for asset in target if asset not in before)
    signed_trades = {
        asset: target.get(asset, 0.0) - before.get(asset, 0.0)
        for asset in assets
    }
    signed_trade_notionals = {
        asset: trade * value for asset, trade in signed_trades.items()
    }
    buy_fraction = sum(max(trade, 0.0) for trade in signed_trades.values())
    sell_fraction = sum(max(-trade, 0.0) for trade in signed_trades.values())
    gross_fraction = buy_fraction + sell_fraction
    gross_notional = value * gross_fraction
    cost_rate = cost_bps * 1e-4
    transaction_costs_by_asset = {
        asset: abs(notional) * cost_rate
        for asset, notional in signed_trade_notionals.items()
    }
    transaction_cost = sum(transaction_costs_by_asset.values())

    return RebalanceResult(
        pre_trade_weights=before,
        target_weights=target,
        signed_trades=signed_trades,
        signed_trade_notionals=signed_trade_notionals,
        transaction_costs_by_asset=transaction_costs_by_asset,
        buy_fraction=buy_fraction,
        sell_fraction=sell_fraction,
        gross_traded_fraction=gross_fraction,
        half_l1_turnover=0.5 * gross_fraction,
        gross_traded_notional=gross_notional,
        transaction_cost=transaction_cost,
        transaction_cost_fraction=transaction_cost / value,
    )


def rebalance_after_drift(
    previous_post_trade_weights: Weights,
    asset_returns: Mapping[Asset, float],
    new_target_weights: Weights,
    *,
    portfolio_value: float = 1.0,
    one_way_cost_bps: float = 10.0,
) -> RebalanceResult:
    """Convenience wrapper that prevents turnover from using stale weights."""

    pre_trade = drift_weights(previous_post_trade_weights, asset_returns)
    return calculate_rebalance(
        pre_trade,
        new_target_weights,
        portfolio_value=portfolio_value,
        one_way_cost_bps=one_way_cost_bps,
    )
