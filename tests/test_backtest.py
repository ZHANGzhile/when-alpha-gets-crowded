import math
import unittest

import pandas as pd

from alpha_crowding.backtest import (
    MissingExecutionDataError,
    active_weights_from_oos_probabilities,
    arithmetic_active_returns,
    build_oos_exposure_schedule,
    build_controller_stock_targets,
    calculate_rebalance,
    combine_stock_target_weights,
    drift_weights,
    evaluate_absolute_performance,
    evaluate_relative_performance,
    historical_risk_percentile,
    probability_to_active_weight,
    rebalance_after_drift,
    relative_nav,
    risk_percentile_to_active_weight,
    simulate_stock_level_controller,
    summarize_controller_performance,
    summarize_crash_episode_losses,
    validate_benchmark_replication_weights,
    validate_execution_constraints,
)


class ControllerPolicyTests(unittest.TestCase):
    def test_frozen_risk_tier_boundaries(self):
        cases = {
            0.00: 1.00,
            0.599999: 1.00,
            0.60: 0.75,
            0.799999: 0.75,
            0.80: 0.50,
            0.899999: 0.50,
            0.90: 0.25,
            1.00: 0.25,
        }
        for percentile, expected_weight in cases.items():
            with self.subTest(percentile=percentile):
                self.assertEqual(
                    risk_percentile_to_active_weight(percentile), expected_weight
                )

    def test_historical_percentile_uses_midranks(self):
        self.assertEqual(historical_risk_percentile(0.2, [0.2, 0.2]), 0.5)
        self.assertEqual(historical_risk_percentile(0.3, [0.1, 0.2]), 1.0)
        self.assertIsNone(
            historical_risk_percentile(0.3, [0.1, 0.2], min_history=3)
        )

    def test_weight_sequence_never_uses_current_or_future_probability(self):
        probabilities = [0.2, 0.2, 0.9, 0.1]
        weights = active_weights_from_oos_probabilities(
            probabilities, min_history=2
        )
        self.assertEqual(weights, (None, None, 0.25, 1.0))
        self.assertEqual(
            active_weights_from_oos_probabilities(
                probabilities + [0.99], min_history=2
            )[:4],
            weights,
        )

    def test_probability_policy_is_only_the_clipped_sensitivity(self):
        self.assertEqual(probability_to_active_weight(0.1), 0.9)
        self.assertEqual(probability_to_active_weight(0.9), 0.25)

    def test_exposure_schedule_uses_only_earlier_same_factor_predictions(self):
        calendar = pd.bdate_range("2024-01-01", periods=16)
        dates = calendar[[1, 6, 11]]
        predictions = pd.DataFrame(
            {
                "decision_at": list(dates) * 2,
                "factor": ["A"] * 3 + ["B"] * 3,
                "probability_M2": [0.1, 0.2, 0.9, 0.9, 0.8, 0.1],
            }
        )
        result = build_oos_exposure_schedule(
            predictions,
            calendar,
            probability_columns=("probability_M2",),
            minimum_history_weeks=2,
        )
        current = result.loc[result["decision_at"].eq(dates[-1])].set_index("factor")
        self.assertEqual(current.loc["A", "active_weight_M2"], 0.25)
        self.assertEqual(current.loc["B", "active_weight_M2"], 1.0)
        self.assertEqual(current.loc["A", "effective_at"], calendar[12])
        warmup = result["decision_at"].isin(dates[:2])
        self.assertTrue(result.loc[warmup, "active_weight_M2"].isna().all())

    def test_benchmark_contract_enforces_pit_and_full_investment(self):
        valid = pd.DataFrame(
            {
                "decision_at": ["2024-01-05", "2024-01-05"],
                "available_at": ["2024-01-04", "2024-01-05"],
                "code": ["a", "b"],
                "weight": [0.4, 0.6],
            }
        )
        checked = validate_benchmark_replication_weights(valid)
        self.assertAlmostEqual(checked["weight"].sum(), 1.0)
        late = valid.copy()
        late.loc[0, "available_at"] = "2024-01-08"
        with self.assertRaisesRegex(ValueError, "not available"):
            validate_benchmark_replication_weights(late)
        incomplete = valid.copy()
        incomplete.loc[0, "weight"] = 0.3
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_benchmark_replication_weights(incomplete)


class StockAccountingTests(unittest.TestCase):
    def test_combines_factor_and_benchmark_at_stock_level(self):
        combined = combine_stock_target_weights(
            {"A": 0.6, "B": 0.4},
            {"B": 0.5, "C": 0.5},
            0.75,
        )
        self.assertAlmostEqual(combined["A"], 0.45)
        self.assertAlmostEqual(combined["B"], 0.425)
        self.assertAlmostEqual(combined["C"], 0.125)
        self.assertAlmostEqual(sum(combined.values()), 1.0)

    def test_weights_drift_before_turnover_is_measured(self):
        drifted = drift_weights({"A": 0.6, "B": 0.4}, {"A": 0.10, "B": -0.05})
        self.assertAlmostEqual(drifted["A"], 0.66 / 1.04)
        self.assertAlmostEqual(drifted["B"], 0.38 / 1.04)

        result = calculate_rebalance(
            drifted,
            {"A": 0.5, "B": 0.5},
            portfolio_value=1_000_000,
            one_way_cost_bps=10,
        )
        expected_half_l1 = abs(drifted["A"] - 0.5)
        self.assertAlmostEqual(result.half_l1_turnover, expected_half_l1)
        self.assertAlmostEqual(
            result.gross_traded_fraction, 2.0 * expected_half_l1
        )
        self.assertAlmostEqual(
            result.transaction_cost,
            result.gross_traded_notional * 10e-4,
        )
        self.assertAlmostEqual(
            sum(abs(value) for value in result.signed_trade_notionals.values()),
            result.gross_traded_notional,
        )
        self.assertAlmostEqual(
            sum(result.transaction_costs_by_asset.values()),
            result.transaction_cost,
        )

    def test_complete_switch_charges_buys_and_sells(self):
        result = calculate_rebalance(
            {"A": 1.0}, {"B": 1.0}, one_way_cost_bps=10
        )
        self.assertEqual(result.buy_fraction, 1.0)
        self.assertEqual(result.sell_fraction, 1.0)
        self.assertEqual(result.gross_traded_fraction, 2.0)
        self.assertEqual(result.half_l1_turnover, 1.0)
        self.assertAlmostEqual(result.transaction_cost_fraction, 0.002)

    def test_rebalance_after_drift_matches_explicit_two_step_accounting(self):
        direct = rebalance_after_drift(
            {"A": 0.6, "B": 0.4},
            {"A": 0.10, "B": -0.05},
            {"A": 0.5, "B": 0.5},
        )
        explicit_pre_trade = drift_weights(
            {"A": 0.6, "B": 0.4}, {"A": 0.10, "B": -0.05}
        )
        explicit = calculate_rebalance(
            explicit_pre_trade, {"A": 0.5, "B": 0.5}
        )
        self.assertAlmostEqual(
            direct.gross_traded_fraction, explicit.gross_traded_fraction
        )

    def test_invalid_or_incomplete_holdings_fail_loudly(self):
        with self.assertRaises(ValueError):
            combine_stock_target_weights({"A": 0.9}, {"A": 1.0}, 0.5)
        with self.assertRaises(KeyError):
            drift_weights({"A": 0.5, "B": 0.5}, {"A": 0.1})

    def test_stock_targets_are_built_for_each_policy(self):
        schedule = pd.DataFrame(
            {
                "decision_at": ["2024-01-05"],
                "effective_at": ["2024-01-08"],
                "factor": ["MOM"],
                "active_weight_M2": [0.5],
                "active_weight_M3": [0.25],
            }
        )
        memberships = pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-05"],
                "factor": ["MOM", "MOM"],
                "leg": ["LONG", "LONG"],
                "code": ["A", "B"],
                "weight": [0.5, 0.5],
            }
        )
        benchmark = pd.DataFrame(
            {
                "decision_at": ["2024-01-05", "2024-01-05"],
                "code": ["B", "C"],
                "weight": [0.4, 0.6],
            }
        )
        result = build_controller_stock_targets(
            schedule,
            memberships,
            benchmark,
            policy_columns=("active_weight_M2", "active_weight_M3"),
        )
        sums = result.groupby("policy")["target_weight"].sum()
        self.assertTrue((sums == 1.0).all())
        m2 = result.loc[result["policy"].eq("active_weight_M2")].set_index("code")
        self.assertAlmostEqual(m2.loc["A", "target_weight"], 0.25)
        self.assertAlmostEqual(m2.loc["B", "target_weight"], 0.45)
        self.assertAlmostEqual(m2.loc["C", "target_weight"], 0.30)

    def test_execution_charges_only_actual_trades_and_keeps_blocked_holding(self):
        targets = pd.DataFrame(
            {
                "effective_at": ["2024-01-02", "2024-01-03"],
                "decision_at": ["2024-01-01", "2024-01-02"],
                "factor": ["MOM", "MOM"],
                "policy": ["M3", "M3"],
                "active_weight": [1.0, 1.0],
                "code": ["A", "B"],
                "target_weight": [1.0, 1.0],
            }
        )
        market = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-03"],
                "code": ["A", "A", "B"],
                "overnight_return": [0.0, 0.0, 0.0],
                "intraday_return": [0.0, 0.0, 0.0],
            }
        )
        constraints = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-03"],
                "execution_at": [
                    "2024-01-02T01:30:00Z",
                    "2024-01-03T01:30:00Z",
                    "2024-01-03T01:30:00Z",
                ],
                "available_at": [
                    "2024-01-02T01:29:00Z",
                    "2024-01-03T01:30:00Z",
                    "2024-01-03T01:30:00Z",
                ],
                "code": ["A", "A", "B"],
                "can_buy": [True, True, True],
                "can_sell": [True, False, True],
                "reason": ["", "suspended", ""],
            }
        )
        portfolio, ledger = simulate_stock_level_controller(
            targets,
            market,
            constraints,
            one_way_cost_bps=10,
        )
        blocked = ledger.loc[
            ledger["date"].eq(pd.Timestamp("2024-01-03"))
            & ledger["code"].eq("A")
        ].iloc[0]
        unfunded = ledger.loc[
            ledger["date"].eq(pd.Timestamp("2024-01-03"))
            & ledger["code"].eq("B")
        ].iloc[0]
        self.assertEqual(blocked["executed_trade_notional"], 0.0)
        self.assertEqual(blocked["reject_reason"], "suspended")
        self.assertEqual(unfunded["executed_trade_notional"], 0.0)
        self.assertIn("insufficient_cash", unfunded["reject_reason"])
        second_day = portfolio.loc[
            portfolio["date"].eq(pd.Timestamp("2024-01-03"))
        ].iloc[0]
        self.assertEqual(second_day["transaction_cost"], 0.0)
        first_trade = ledger.loc[ledger["date"].eq(pd.Timestamp("2024-01-02"))]
        self.assertTrue(first_trade["reject_reason"].eq("").all())
        first_day = portfolio.loc[
            portfolio["date"].eq(pd.Timestamp("2024-01-02"))
        ].iloc[0]
        self.assertAlmostEqual(
            first_day["pre_trade_value"] - first_day["transaction_cost"],
            first_day["post_trade_value"],
        )
        self.assertAlmostEqual(
            first_trade["transaction_cost"].sum(),
            first_day["transaction_cost"],
        )

    def test_execution_constraint_contract_and_missing_held_return_fail(self):
        constraints = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "execution_at": ["2024-01-02T01:30:00Z"],
                "available_at": ["2024-01-02T01:31:00Z"],
                "code": ["A"],
                "can_buy": [True],
                "can_sell": [True],
            }
        )
        with self.assertRaisesRegex(ValueError, "not available"):
            validate_execution_constraints(constraints)

        targets = pd.DataFrame(
            {
                "effective_at": ["2024-01-02"],
                "decision_at": ["2024-01-01"],
                "factor": ["MOM"],
                "policy": ["M3"],
                "active_weight": [1.0],
                "code": ["A"],
                "target_weight": [1.0],
            }
        )
        market = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "code": ["A"],
                "overnight_return": [0.0],
                "intraday_return": [None],
            }
        )
        valid = constraints.assign(available_at="2024-01-02T01:30:00Z")
        with self.assertRaisesRegex(MissingExecutionDataError, "intraday return"):
            simulate_stock_level_controller(
                targets,
                market,
                valid,
                one_way_cost_bps=10,
            )

    def test_open_rebalance_assigns_overnight_return_to_old_holding(self):
        targets = pd.DataFrame(
            {
                "effective_at": ["2024-01-02", "2024-01-03"],
                "decision_at": ["2024-01-01", "2024-01-02"],
                "factor": ["MOM", "MOM"],
                "policy": ["M3", "M3"],
                "active_weight": [1.0, 1.0],
                "code": ["A", "B"],
                "target_weight": [1.0, 1.0],
            }
        )
        market = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-03"],
                "code": ["A", "A", "B"],
                "overnight_return": [0.0, 0.10, 0.0],
                "intraday_return": [0.0, 0.0, 0.20],
            }
        )
        constraints = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-03"],
                "execution_at": [
                    "2024-01-02T01:30:00Z",
                    "2024-01-03T01:30:00Z",
                    "2024-01-03T01:30:00Z",
                ],
                "available_at": [
                    "2024-01-02T01:30:00Z",
                    "2024-01-03T01:30:00Z",
                    "2024-01-03T01:30:00Z",
                ],
                "code": ["A", "A", "B"],
                "can_buy": [True, True, True],
                "can_sell": [True, True, True],
            }
        )
        portfolio, _ = simulate_stock_level_controller(
            targets,
            market,
            constraints,
            one_way_cost_bps=0,
        )
        second = portfolio.loc[
            portfolio["date"].eq(pd.Timestamp("2024-01-03"))
        ].iloc[0]
        self.assertAlmostEqual(second["overnight_pnl"], 0.10)
        self.assertAlmostEqual(second["pre_trade_value"], 1.10)
        self.assertAlmostEqual(second["intraday_pnl"], 0.22)
        self.assertAlmostEqual(second["net_return"], 0.32)


class RelativePerformanceTests(unittest.TestCase):
    def test_absolute_evaluation_reports_required_path_metrics(self):
        returns = [0.10, -0.10, 0.02, -0.03]
        metrics = evaluate_absolute_performance(
            returns,
            periods_per_year=4,
            worst_horizon_sessions=2,
        )
        self.assertEqual(metrics.observations, 4)
        self.assertAlmostEqual(
            metrics.total_return,
            (1.10 * 0.90 * 1.02 * 0.97) - 1.0,
        )
        self.assertAlmostEqual(metrics.worst_horizon_return, (0.90 * 1.02) - 1.0)
        self.assertGreater(metrics.annualized_volatility, 0.0)
        self.assertTrue(math.isfinite(metrics.sharpe_ratio))
        self.assertTrue(math.isfinite(metrics.sortino_ratio))
        self.assertGreater(metrics.max_drawdown, 0.0)

    def test_controller_summary_aligns_benchmark_and_counts_rejections(self):
        dates = pd.bdate_range("2024-01-02", periods=20)
        paths = pd.DataFrame(
            {
                "date": dates,
                "factor": "MOM",
                "policy": "M3",
                "cost_bps": 10.0,
                "net_return": [0.001] * 19 + [-0.01],
                "active_weight": [0.75] * 20,
                "half_l1_turnover": [0.0] * 19 + [0.1],
                "transaction_cost": [0.0] * 19 + [0.001],
            }
        )
        benchmark = pd.DataFrame({"date": dates, "daily_return": [0.0] * 20})
        ledger = pd.DataFrame(
            {
                "factor": ["MOM", "MOM"],
                "policy": ["M3", "M3"],
                "cost_bps": [10.0, 10.0],
                "reject_reason": ["", "suspended"],
            }
        )
        result = summarize_controller_performance(paths, benchmark, ledger).iloc[0]
        self.assertEqual(result["observations"], 20)
        self.assertEqual(result["rejected_ledger_rows"], 1)
        self.assertAlmostEqual(result["mean_active_weight"], 0.75)
        self.assertAlmostEqual(result["total_half_l1_turnover"], 0.1)
        self.assertTrue(math.isfinite(result["information_ratio"]))

        with self.assertRaisesRegex(ValueError, "aligned benchmark"):
            summarize_controller_performance(paths, benchmark.iloc[:-1], ledger)

    def test_active_crash_episode_losses_use_complete_oos_intervals(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        paths = pd.DataFrame(
            {
                "date": dates,
                "factor": "MOM",
                "policy": "M3",
                "cost_bps": 10.0,
                "net_return": [0.0, 0.10, -0.10, 0.0],
            }
        )
        benchmark = pd.DataFrame({"date": dates, "daily_return": [0.0] * 4})
        episodes = pd.DataFrame(
            {
                "factor": ["MOM", "MOM"],
                "episode_id": ["inside", "outside"],
                "interval_start_at": [dates[1], dates[0] - pd.Timedelta(days=1)],
                "interval_end_at": [dates[2], dates[0]],
            }
        )
        losses, summary = summarize_crash_episode_losses(
            paths,
            benchmark,
            episodes,
        )
        self.assertEqual(losses["episode_id"].tolist(), ["inside"])
        self.assertAlmostEqual(losses.loc[0, "portfolio_episode_return"], -0.01)
        self.assertAlmostEqual(losses.loc[0, "active_episode_return"], -0.01)
        self.assertAlmostEqual(losses.loc[0, "portfolio_episode_mdd"], 0.10)
        self.assertEqual(summary.loc[0, "crash_episode_count"], 1)

        incomplete = paths.drop(index=2)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize_crash_episode_losses(incomplete, benchmark, episodes.iloc[:1])

    def test_relative_nav_is_wealth_ratio_not_compounded_active_difference(self):
        factor = [0.10, -0.10]
        benchmark = [0.05, 0.00]
        path = relative_nav(factor, benchmark)
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], 1.0)
        self.assertAlmostEqual(path[1], 1.10 / 1.05)
        self.assertAlmostEqual(path[2], (1.10 * 0.90) / 1.05)
        self.assertEqual(arithmetic_active_returns(factor, benchmark), (0.05, -0.10))
        self.assertNotAlmostEqual(path[-1], (1.0 + 0.05) * (1.0 - 0.10))

    def test_evaluation_reports_relative_drawdown_te_ir_and_cvar(self):
        factor = [0.10, -0.10, 0.02, -0.03]
        benchmark = [0.05, 0.00, 0.01, -0.01]
        metrics = evaluate_relative_performance(
            factor, benchmark, periods_per_year=4, cvar_alpha=0.25
        )
        expected_path = relative_nav(factor, benchmark)
        self.assertEqual(metrics.observations, 4)
        self.assertAlmostEqual(
            metrics.relative_total_return, expected_path[-1] - 1.0
        )
        self.assertAlmostEqual(
            metrics.annualized_relative_return, expected_path[-1] - 1.0
        )
        self.assertGreater(metrics.tracking_error, 0.0)
        self.assertTrue(math.isfinite(metrics.information_ratio))
        self.assertGreater(metrics.active_max_drawdown, 0.0)
        self.assertAlmostEqual(metrics.active_cvar, 0.10)

    def test_mismatched_series_are_rejected(self):
        with self.assertRaises(ValueError):
            relative_nav([0.01], [0.01, 0.02])


if __name__ == "__main__":
    unittest.main()
