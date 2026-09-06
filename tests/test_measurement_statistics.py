import unittest

import numpy as np
import pandas as pd

from alpha_crowding.measurement import (
    amihud_illiquidity,
    industry_residual_returns,
    robust_past_shock,
    shrunk_correlation_features,
)


class MeasurementStatisticsTests(unittest.TestCase):
    def test_industry_residuals_are_same_date_and_zero_mean(self):
        panel = pd.DataFrame(
            {
                "date": ["2024-01-02"] * 4,
                "code": ["a", "b", "c", "d"],
                "industry": ["I1", "I1", "I2", "I2"],
                "daily_return": [0.01, 0.03, -0.02, 0.02],
            }
        )
        result = industry_residual_returns(panel)
        means = result.groupby("industry")["industry_residual_return"].mean()
        self.assertTrue(np.allclose(means, 0.0))
        self.assertEqual(result.loc[0, "industry_return"], 0.02)

    def test_shrunk_structure_detects_common_component(self):
        rng = np.random.default_rng(42)
        common = rng.normal(scale=0.02, size=(80, 1))
        matrix = pd.DataFrame(common + rng.normal(scale=0.005, size=(80, 12)))
        result = shrunk_correlation_features(matrix)
        self.assertGreater(result.residual_sync, 0.5)
        self.assertGreater(result.eigen_concentration, 0.5)
        self.assertEqual(result.valid_assets, 12)

    def test_illiquidity_and_past_only_shock_preserve_invalidity(self):
        illiquidity = amihud_illiquidity(
            pd.Series([0.02, 0.01]), pd.Series([100.0, 0.0])
        )
        self.assertAlmostEqual(illiquidity.iloc[0], 0.0002)
        self.assertTrue(pd.isna(illiquidity.iloc[1]))
        history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(
            robust_past_shock(history, 6.0, minimum_history=5), 3.0
        )


if __name__ == "__main__":
    unittest.main()
