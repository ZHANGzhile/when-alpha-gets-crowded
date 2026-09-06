import unittest

import numpy as np
import pandas as pd

from alpha_crowding.outcomes import forward_window_outcome


class ForwardOutcomeTests(unittest.TestCase):
    def test_window_starts_after_decision_and_active_uses_wealth_ratio(self):
        calendar = pd.bdate_range("2024-01-02", periods=8)
        portfolio = pd.Series([0.5, 0.10, -0.10, 0.0, 0.0, 0.0, 0.0, 0.0], index=calendar)
        benchmark = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=calendar)
        result = forward_window_outcome(
            portfolio,
            calendar,
            decision_at=calendar[0],
            horizon_sessions=2,
            benchmark_returns=benchmark,
        )
        self.assertEqual(result["label_start_at"], calendar[1])
        self.assertEqual(result["label_end_at"], calendar[2])
        self.assertAlmostEqual(result["future_return"], -0.01)
        self.assertAlmostEqual(result["future_mdd"], 0.10)

    def test_immature_horizon_stays_missing(self):
        calendar = pd.bdate_range("2024-01-02", periods=4)
        result = forward_window_outcome(
            pd.Series(0.0, index=calendar),
            calendar,
            decision_at=calendar[-2],
            horizon_sessions=2,
        )
        self.assertFalse(result["outcome_mature"])
        self.assertTrue(np.isnan(result["future_return"]))
        self.assertTrue(pd.isna(result["label_end_at"]))


if __name__ == "__main__":
    unittest.main()
