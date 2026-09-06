import unittest

import numpy as np
import pandas as pd

from alpha_crowding.measurement import factor_return_shock, portfolio_risk_characteristics


class PortfolioRiskCharacteristicTests(unittest.TestCase):
    def test_factor_return_shock_excludes_current_five_sessions(self):
        calendar = pd.bdate_range("2022-01-03", periods=80)
        returns = pd.Series(0.001, index=calendar)
        returns.iloc[-5:] = -0.02
        result = factor_return_shock(
            returns,
            calendar,
            decision_at=calendar[-1],
            recent_sessions=5,
            baseline_sessions=30,
            baseline_minimum=20,
        )
        # A constant baseline has zero MAD and must stay invalid instead of
        # receiving an arbitrary epsilon denominator.
        self.assertTrue(np.isnan(result["factor_return_shock"]))
        self.assertEqual(result["factor_return_baseline_end"], calendar[-6])
        self.assertEqual(result["factor_return_recent_start"], calendar[-5])

    def test_recent_windows_do_not_enter_historical_baselines(self):
        calendar = pd.bdate_range("2023-01-02", periods=90)
        rows = []
        for code, multiplier in (("a", 1.0), ("b", 2.0)):
            turn = np.arange(1.0, 91.0) * multiplier
            illiquidity = np.arange(101.0, 191.0) * multiplier
            for date, turn_value, illiq_value in zip(calendar, turn, illiquidity):
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "turn": turn_value,
                        "daily_illiquidity": illiq_value,
                    }
                )
        result = portfolio_risk_characteristics(
            pd.DataFrame(rows),
            ["a", "b"],
            calendar,
            decision_at=calendar[-1],
            turnover_recent_sessions=5,
            turnover_baseline_sessions=20,
            turnover_baseline_minimum=20,
            liquidity_recent_sessions=10,
            liquidity_baseline_sessions=30,
            liquidity_recent_minimum=10,
            liquidity_baseline_minimum=30,
        )
        portfolio = result["portfolio"]
        self.assertEqual(portfolio["turnover_baseline_end"], calendar[-6])
        self.assertEqual(portfolio["turnover_recent_start"], calendar[-5])
        self.assertEqual(portfolio["liquidity_baseline_end"], calendar[-11])
        self.assertEqual(portfolio["liquidity_recent_start"], calendar[-10])
        self.assertGreater(portfolio["turnover_shock"], 0)
        self.assertGreater(portfolio["liquidity_shock"], 0)
        self.assertEqual(portfolio["turnover_level_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
