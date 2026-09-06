import unittest

import numpy as np
import pandas as pd

from alpha_crowding.factors import assign_legs, compute_raw_signals


class FactorSignalTests(unittest.TestCase):
    @staticmethod
    def bars(n=300):
        dates = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame(
            {
                "date": dates,
                "code": "A",
                "close": np.arange(1, n + 1, dtype=float),
                "pb_mrq": 2.0,
            }
        )

    def test_factor_formulas(self):
        bars = self.bars()
        result = compute_raw_signals(bars)
        row = result.iloc[260]
        expected_mom = bars.iloc[239].close / bars.iloc[8].close - 1
        expected_rev = -(bars.iloc[260].close / bars.iloc[240].close - 1)
        self.assertAlmostEqual(row.momentum, expected_mom)
        self.assertAlmostEqual(row.reversal, expected_rev)
        self.assertAlmostEqual(row.value, -np.log(2.0))
        self.assertLess(row.low_volatility, 0)

    def test_future_mutation_does_not_change_past_signal(self):
        bars = self.bars()
        before = compute_raw_signals(bars).set_index("date").loc[bars.iloc[260].date]
        bars.loc[bars.index > 260, "close"] = 1_000_000
        after = compute_raw_signals(bars).set_index("date").loc[bars.iloc[260].date]
        self.assertAlmostEqual(before.momentum, after.momentum)
        self.assertAlmostEqual(before.reversal, after.reversal)
        self.assertAlmostEqual(before.low_volatility, after.low_volatility)

    def test_leg_assignment_is_exact_and_deterministic(self):
        frame = pd.DataFrame(
            {
                "date": pd.Timestamp("2024-01-05"),
                "code": [f"S{i:02d}" for i in range(20)],
                "industry": ["I1"] * 10 + ["I2"] * 10,
                "signal": np.arange(20, dtype=float),
            }
        )
        result = assign_legs(frame, signal_col="signal", quantile=0.10)
        self.assertEqual((result.leg == "LONG").sum(), 2)
        self.assertEqual((result.leg == "SHORT").sum(), 2)
        self.assertAlmostEqual(result.loc[result.leg == "LONG", "weight"].sum(), 1.0)
        self.assertAlmostEqual(result.loc[result.leg == "SHORT", "weight"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()

