import math
import unittest

from alpha_crowding.backtest import (
    active_weights_from_oos_probabilities,
    arithmetic_active_returns,
    calculate_rebalance,
    combine_stock_target_weights,
    drift_weights,
    evaluate_relative_performance,
    historical_risk_percentile,
    probability_to_active_weight,
    rebalance_after_drift,
    relative_nav,
    risk_percentile_to_active_weight,
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


class RelativePerformanceTests(unittest.TestCase):
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
