"""Stock-level controller targets and constrained execution accounting."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import numpy as np
import pandas as pd

from .accounting import combine_stock_target_weights


class MissingExecutionDataError(RuntimeError):
    """Raised when a held or ordered security lacks required market data."""


def _normalized_dates(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    return result


def _as_bool(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if not series.isin([True, False, 0, 1]).all():
        raise ValueError(f"{name} must be boolean")
    return series.astype(bool)


def validate_execution_constraints(constraints: pd.DataFrame) -> pd.DataFrame:
    """Validate point-in-time, side-specific opening tradeability observations."""

    required = [
        "date",
        "execution_at",
        "available_at",
        "code",
        "can_buy",
        "can_sell",
    ]
    missing = set(required) - set(constraints.columns)
    if missing:
        raise KeyError(f"missing execution-constraint fields: {sorted(missing)}")
    optional = ["reason"] if "reason" in constraints.columns else []
    result = _normalized_dates(constraints[required + optional], ["date"])
    result["execution_at"] = pd.to_datetime(
        result["execution_at"], errors="raise", utc=True
    )
    result["available_at"] = pd.to_datetime(
        result["available_at"], errors="raise", utc=True
    )
    result["code"] = result["code"].astype(str)
    result["can_buy"] = _as_bool(result["can_buy"], name="can_buy")
    result["can_sell"] = _as_bool(result["can_sell"], name="can_sell")
    if result[["date", "execution_at", "available_at", "code"]].isna().any().any():
        raise ValueError("execution-constraint keys must not be missing")
    if result.duplicated(["date", "code"]).any():
        raise ValueError("execution constraints must be unique by date/code")
    execution_dates = result["execution_at"].dt.tz_convert(None).dt.normalize()
    if not execution_dates.eq(result["date"]).all():
        raise ValueError("execution_at must fall on the stated trading date")
    if (result["available_at"] > result["execution_at"]).any():
        raise ValueError("execution constraints were not available by execution time")
    if "reason" not in result:
        result["reason"] = ""
    result["reason"] = result["reason"].fillna("").astype(str)
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def build_open_tradeability(
    daily_open: pd.DataFrame,
    daily_limits: pd.DataFrame,
    *,
    source_available_time: str = "09:00:00",
    execution_time: str = "09:30:00",
    market_timezone: str = "Asia/Shanghai",
    price_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Create conservative side-specific opening constraints from PIT limit prices."""

    open_required = ["date", "code", "open", "tradestatus"]
    limit_required = ["date", "code", "up_limit", "down_limit"]
    for name, frame, required in (
        ("daily open", daily_open, open_required),
        ("daily limit", daily_limits, limit_required),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} fields: {sorted(missing)}")
    market = _normalized_dates(daily_open[open_required], ["date"])
    limits = _normalized_dates(daily_limits[limit_required], ["date"])
    for frame in (market, limits):
        frame["code"] = frame["code"].astype(str)
    if market.duplicated(["date", "code"]).any():
        raise ValueError("daily open rows contain duplicate keys")
    if limits.duplicated(["date", "code"]).any():
        raise ValueError("daily limit rows contain duplicate keys")
    market["open"] = pd.to_numeric(market["open"], errors="coerce")
    market["tradestatus"] = pd.to_numeric(
        market["tradestatus"], errors="raise"
    ).astype(int)
    if not market["tradestatus"].isin([0, 1]).all():
        raise ValueError("tradestatus must be binary")
    for column in ("up_limit", "down_limit"):
        limits[column] = pd.to_numeric(limits[column], errors="coerce")
    joined = market.merge(limits, on=["date", "code"], how="left", validate="one_to_one")
    traded = joined["tradestatus"].eq(1)
    invalid_traded = traded & (
        joined[["open", "up_limit", "down_limit"]].isna().any(axis=1)
        | (joined["open"] <= 0.0)
        | (joined["down_limit"] <= 0.0)
        | (joined["up_limit"] <= joined["down_limit"])
    )
    if invalid_traded.any():
        examples = joined.loc[invalid_traded, ["date", "code"]].head()
        raise ValueError(
            "tradable securities lack valid opening limit data: "
            f"{examples.to_dict('records')}"
        )
    joined["can_buy"] = traded & (
        joined["open"] < joined["up_limit"] - price_tolerance
    )
    joined["can_sell"] = traded & (
        joined["open"] > joined["down_limit"] + price_tolerance
    )
    joined["reason"] = ""
    joined.loc[~traded, "reason"] = "suspended"
    joined.loc[traded & ~joined["can_buy"], "reason"] = "open_at_upper_limit"
    joined.loc[traded & ~joined["can_sell"], "reason"] = "open_at_lower_limit"
    date_text = joined["date"].dt.strftime("%Y-%m-%d")
    joined["available_at"] = pd.to_datetime(
        date_text + " " + source_available_time,
        errors="raise",
    ).dt.tz_localize(market_timezone).dt.tz_convert("UTC")
    joined["execution_at"] = pd.to_datetime(
        date_text + " " + execution_time,
        errors="raise",
    ).dt.tz_localize(market_timezone).dt.tz_convert("UTC")
    return validate_execution_constraints(
        joined[
            [
                "date",
                "execution_at",
                "available_at",
                "code",
                "can_buy",
                "can_sell",
                "reason",
            ]
        ]
    )


def build_controller_stock_targets(
    exposure_schedule: pd.DataFrame,
    factor_memberships: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    *,
    policy_columns: Sequence[str],
) -> pd.DataFrame:
    """Combine factor-long and benchmark weights for every controller policy."""

    policies = tuple(policy_columns)
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policy_columns must be nonempty and unique")
    schedule_required = {"decision_at", "effective_at", "factor", *policies}
    membership_required = {"date", "factor", "leg", "code", "weight"}
    benchmark_required = {"decision_at", "code", "weight"}
    for name, frame, required in (
        ("schedule", exposure_schedule, schedule_required),
        ("memberships", factor_memberships, membership_required),
        ("benchmark", benchmark_weights, benchmark_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} fields: {sorted(missing)}")

    schedule = _normalized_dates(exposure_schedule, ["decision_at", "effective_at"])
    if schedule.duplicated(["decision_at", "factor"]).any():
        raise ValueError("exposure schedule must be unique by decision_at/factor")
    memberships = _normalized_dates(factor_memberships, ["date"])
    memberships = memberships.loc[memberships["leg"].eq("LONG")].copy()
    benchmark = _normalized_dates(benchmark_weights, ["decision_at"])
    for frame in (memberships, benchmark):
        frame["code"] = frame["code"].astype(str)
        frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if memberships.duplicated(["date", "factor", "code"]).any():
        raise ValueError("factor-long weights contain duplicate keys")
    if benchmark.duplicated(["decision_at", "code"]).any():
        raise ValueError("benchmark weights contain duplicate keys")

    factor_groups = {
        (date, factor): dict(zip(rows["code"], rows["weight"]))
        for (date, factor), rows in memberships.groupby(["date", "factor"])
    }
    benchmark_groups = {
        date: dict(zip(rows["code"], rows["weight"]))
        for date, rows in benchmark.groupby("decision_at")
    }
    rows: list[dict[str, object]] = []
    for current in schedule.sort_values(["decision_at", "factor"]).itertuples():
        factor_key = (current.decision_at, current.factor)
        if factor_key not in factor_groups:
            raise ValueError(f"missing factor-long weights for {factor_key}")
        if current.decision_at not in benchmark_groups:
            raise ValueError(
                f"missing benchmark weights for {current.decision_at.date()}"
            )
        for policy in policies:
            active_weight = getattr(current, policy)
            if pd.isna(active_weight):
                continue
            combined = combine_stock_target_weights(
                factor_groups[factor_key],
                benchmark_groups[current.decision_at],
                float(active_weight),
            )
            rows.extend(
                {
                    "decision_at": current.decision_at,
                    "effective_at": current.effective_at,
                    "factor": current.factor,
                    "policy": policy,
                    "active_weight": float(active_weight),
                    "code": code,
                    "target_weight": weight,
                }
                for code, weight in combined.items()
                if weight > 0.0
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    sums = result.groupby(["effective_at", "factor", "policy"])[
        "target_weight"
    ].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=1e-10, atol=1e-10):
        raise AssertionError("controller stock targets do not sum to one")
    if result.duplicated(["effective_at", "factor", "policy", "code"]).any():
        raise ValueError("controller stock targets contain duplicate keys")
    return result.sort_values(
        ["effective_at", "factor", "policy", "code"]
    ).reset_index(drop=True)


def simulate_stock_level_controller(
    targets: pd.DataFrame,
    daily_market: pd.DataFrame,
    execution_constraints: pd.DataFrame,
    *,
    one_way_cost_bps: float,
    initial_value: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exact-notional daily accounting with side-specific rejected trades.

    Targets are attempted only on their effective date. Rejected orders remain
    unfilled and the old holding or cash stays in the portfolio until a later
    scheduled rebalance. Costs are charged only on executed buy and sell
    notionals. Returns are then applied using the project's next-session
    convention.
    """

    target_required = {
        "effective_at",
        "decision_at",
        "factor",
        "policy",
        "code",
        "target_weight",
        "active_weight",
    }
    market_required = {"date", "code", "overnight_return", "intraday_return"}
    for name, frame, required in (
        ("target", targets, target_required),
        ("daily market", daily_market, market_required),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"missing {name} fields: {sorted(missing)}")
    value = float(initial_value)
    cost_bps = float(one_way_cost_bps)
    if not isfinite(value) or value <= 0.0:
        raise ValueError("initial_value must be finite and positive")
    if not isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("one_way_cost_bps must be finite and non-negative")
    cost_rate = cost_bps * 1e-4

    target_frame = _normalized_dates(targets, ["effective_at", "decision_at"])
    target_frame["code"] = target_frame["code"].astype(str)
    target_frame["target_weight"] = pd.to_numeric(
        target_frame["target_weight"], errors="raise"
    )
    target_frame["active_weight"] = pd.to_numeric(
        target_frame["active_weight"], errors="raise"
    )
    if (
        not np.isfinite(target_frame[["target_weight", "active_weight"]]).all().all()
        or (target_frame["target_weight"] < 0.0).any()
        or not target_frame["active_weight"].between(0.0, 1.0).all()
    ):
        raise ValueError("target and active weights must be finite and valid")
    if (target_frame["effective_at"] <= target_frame["decision_at"]).any():
        raise ValueError("controller targets must take effect after their decision")
    target_key = ["effective_at", "factor", "policy", "code"]
    if target_frame.duplicated(target_key).any():
        raise ValueError("targets must be unique by effective_at/factor/policy/code")
    sums = target_frame.groupby(["effective_at", "factor", "policy"])[
        "target_weight"
    ].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("every controller target must sum to one")
    group_key = ["effective_at", "factor", "policy"]
    if (
        target_frame.groupby(group_key)["decision_at"].nunique().gt(1).any()
        or target_frame.groupby(group_key)["active_weight"].nunique().gt(1).any()
    ):
        raise ValueError("target metadata must be constant within each rebalance")

    market = _normalized_dates(daily_market, ["date"])
    market["code"] = market["code"].astype(str)
    for column in ("overnight_return", "intraday_return"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    if market.duplicated(["date", "code"]).any():
        raise ValueError("daily market must be unique by date/code")
    constraints = validate_execution_constraints(execution_constraints)
    market_lookup = market.set_index(["date", "code"])[
        ["overnight_return", "intraday_return"]
    ]
    constraint_lookup = constraints.set_index(["date", "code"])
    sessions = pd.DatetimeIndex(market["date"].drop_duplicates()).sort_values()

    portfolio_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    for (factor, policy), group in target_frame.groupby(["factor", "policy"]):
        schedules: dict[pd.Timestamp, pd.DataFrame] = {}
        for effective_at, date_rows in group.groupby("effective_at"):
            if effective_at in schedules:
                raise ValueError("multiple targets map to one execution date")
            schedules[effective_at] = date_rows
        first = min(schedules)
        holdings: dict[str, float] = {}
        cash = value
        current_active_weight: float | None = None
        current_decision_at: pd.Timestamp | None = None
        for date in sessions[sessions >= first]:
            start_value = cash + sum(holdings.values())
            overnight_pnl = 0.0
            for code, amount in list(holdings.items()):
                key = (date, code)
                if key not in market_lookup.index:
                    raise MissingExecutionDataError(
                        f"missing held overnight return for {code} at {date.date()}"
                    )
                overnight_return = market_lookup.loc[key, "overnight_return"]
                if pd.isna(overnight_return):
                    raise MissingExecutionDataError(
                        f"missing held overnight return for {code} at {date.date()}"
                    )
                overnight_return = float(overnight_return)
                if not isfinite(overnight_return) or overnight_return < -1.0:
                    raise ValueError("overnight returns must be finite and at least -1")
                asset_pnl = amount * overnight_return
                holdings[code] = amount + asset_pnl
                overnight_pnl += asset_pnl
            holdings = {
                code: amount for code, amount in holdings.items() if amount > 1e-15
            }
            pre_trade_value = cash + sum(holdings.values())
            if pre_trade_value <= 0.0:
                raise ValueError("portfolio value became non-positive")
            transaction_cost = 0.0
            gross_traded_notional = 0.0
            rebalance = date in schedules
            if rebalance:
                current_target = schedules[date]
                target = dict(
                    zip(current_target["code"], current_target["target_weight"])
                )
                active_weight = float(current_target["active_weight"].iloc[0])
                decision_at = current_target["decision_at"].iloc[0]
                current_active_weight = active_weight
                current_decision_at = decision_at
                assets = sorted(set(holdings) | set(target))
                pre_trade_holdings = dict(holdings)
                flags: dict[str, tuple[bool, bool, str]] = {}
                for code in assets:
                    key = (date, code)
                    if key not in constraint_lookup.index:
                        raise MissingExecutionDataError(
                            f"missing execution constraint for {code} at {date.date()}"
                        )
                    row = constraint_lookup.loc[key]
                    flags[code] = (
                        bool(row["can_buy"]),
                        bool(row["can_sell"]),
                        str(row["reason"]),
                    )
                estimated_post_trade_value = pre_trade_value
                for _ in range(100):
                    desired = {
                        code: target.get(code, 0.0) * estimated_post_trade_value
                        - holdings.get(code, 0.0)
                        for code in assets
                    }
                    sells = {
                        code: -amount
                        for code, amount in desired.items()
                        if amount < 0.0 and flags[code][1]
                    }
                    sell_total = sum(sells.values())
                    sell_cost = cost_rate * sell_total
                    cash_after_sells = cash + sell_total - sell_cost
                    buy_requests = {
                        code: amount
                        for code, amount in desired.items()
                        if amount > 0.0 and flags[code][0]
                    }
                    buy_total_requested = sum(buy_requests.values())
                    affordable = max(0.0, cash_after_sells) / (1.0 + cost_rate)
                    buy_scale = (
                        min(1.0, affordable / buy_total_requested)
                        if buy_total_requested > 0.0
                        else 0.0
                    )
                    buys = {
                        code: amount * buy_scale
                        for code, amount in buy_requests.items()
                    }
                    buy_total = sum(buys.values())
                    buy_cost = cost_rate * buy_total
                    transaction_cost = sell_cost + buy_cost
                    revised_value = pre_trade_value - transaction_cost
                    if abs(revised_value - estimated_post_trade_value) <= max(
                        1e-14, 1e-13 * pre_trade_value
                    ):
                        break
                    estimated_post_trade_value = revised_value
                else:
                    raise RuntimeError("transaction-cost fixed point did not converge")
                gross_traded_notional = sell_total + buy_total
                for code, amount in sells.items():
                    holdings[code] = holdings.get(code, 0.0) - amount
                for code, amount in buys.items():
                    holdings[code] = holdings.get(code, 0.0) + amount
                holdings = {
                    code: amount for code, amount in holdings.items() if amount > 1e-15
                }
                cash = cash + sell_total - buy_total - transaction_cost
                if cash < -1e-12:
                    raise AssertionError("execution created negative cash")
                cash = max(0.0, cash)
                post_trade_value = cash + sum(holdings.values())
                for code in assets:
                    requested = desired[code]
                    executed = buys.get(code, 0.0) - sells.get(code, 0.0)
                    can_buy, can_sell, source_reason = flags[code]
                    reject_reason = ""
                    if requested > 0.0 and not can_buy:
                        reject_reason = source_reason or "cannot_buy"
                    elif requested < 0.0 and not can_sell:
                        reject_reason = source_reason or "cannot_sell"
                    elif (
                        requested > 0.0
                        and buy_scale < 1.0 - 1e-10
                        and executed + 1e-12 < requested
                    ):
                        reject_reason = "insufficient_cash_after_constraints_and_costs"
                    ledger_rows.append(
                        {
                            "date": date,
                            "decision_at": decision_at,
                            "factor": factor,
                            "policy": policy,
                            "active_weight": active_weight,
                            "code": code,
                            "pre_trade_weight": pre_trade_holdings.get(code, 0.0)
                            / pre_trade_value,
                            "target_weight": target.get(code, 0.0),
                            "requested_trade_notional": requested,
                            "executed_trade_notional": executed,
                            "transaction_cost": abs(executed) * cost_rate,
                            "post_trade_weight": holdings.get(code, 0.0)
                            / post_trade_value,
                            "reject_reason": reject_reason,
                        }
                    )
            post_trade_value = cash + sum(holdings.values())
            intraday_pnl = 0.0
            for code, amount in list(holdings.items()):
                key = (date, code)
                if key not in market_lookup.index or pd.isna(
                    market_lookup.loc[key, "intraday_return"]
                ):
                    raise MissingExecutionDataError(
                        f"missing held intraday return for {code} at {date.date()}"
                    )
                intraday_return = float(market_lookup.loc[key, "intraday_return"])
                if not isfinite(intraday_return) or intraday_return < -1.0:
                    raise ValueError("intraday returns must be finite and at least -1")
                asset_pnl = amount * intraday_return
                holdings[code] = amount + asset_pnl
                intraday_pnl += asset_pnl
            holdings = {
                code: amount for code, amount in holdings.items() if amount > 1e-15
            }
            end_value = cash + sum(holdings.values())
            if end_value <= 0.0:
                raise ValueError("portfolio value became non-positive after returns")
            portfolio_rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "policy": policy,
                    "decision_at": current_decision_at,
                    "active_weight": current_active_weight,
                    "rebalance": rebalance,
                    "pre_trade_value": pre_trade_value,
                    "post_trade_value": post_trade_value,
                    "end_value": end_value,
                    "overnight_pnl": overnight_pnl,
                    "intraday_pnl": intraday_pnl,
                    "gross_return": (
                        overnight_pnl + intraday_pnl
                    ) / start_value,
                    "net_return": end_value / start_value - 1.0,
                    "cash_weight": cash / end_value,
                    "gross_traded_notional": gross_traded_notional,
                    "half_l1_turnover": 0.5
                    * gross_traded_notional
                    / pre_trade_value,
                    "transaction_cost": transaction_cost,
                    "transaction_cost_fraction": transaction_cost
                    / pre_trade_value,
                }
            )
    portfolio = pd.DataFrame(portfolio_rows).sort_values(
        ["date", "factor", "policy"]
    ).reset_index(drop=True)
    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["date", "factor", "policy", "code"]
    ).reset_index(drop=True)
    return portfolio, ledger
