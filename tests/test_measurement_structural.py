import unittest

import numpy as np
import pandas as pd

from alpha_crowding.measurement import measure_structural_placebos


class StructuralPlaceboIntegrationTests(unittest.TestCase):
    def test_real_leg_runs_through_matching_and_shrunk_measurement(self):
        rng = np.random.default_rng(17)
        dates = pd.bdate_range("2024-01-02", periods=45)
        codes = [f"s{i:02d}" for i in range(24)]
        industries = ["A"] * 12 + ["B"] * 12
        candidates = pd.DataFrame(
            {
                "code": codes,
                "industry": industries,
                "lagged_liquidity": np.arange(1, 25, dtype=float),
            }
        )
        common = rng.normal(0, 0.01, size=(len(dates), 1))
        values = common + rng.normal(0, 0.005, size=(len(dates), len(codes)))
        daily = pd.DataFrame(
            {
                "date": np.repeat(dates, len(codes)),
                "code": np.tile(codes, len(dates)),
                "daily_return": values.ravel(),
            }
        )
        result = measure_structural_placebos(
            daily,
            candidates,
            codes[2:8] + codes[14:20],
            decision_at=dates[-1],
            factor="MOM",
            leg="LONG",
            draws=10,
            minimum_valid_draws=8,
            minimum_days=40,
            minimum_assets=10,
            maximum_mean_absolute_z_difference=3.0,
        )
        self.assertEqual(set(result.features["feature"]), {"residual_sync", "eigen_concentration"})
        self.assertTrue(result.features["placebo_valid"].all())
        self.assertEqual(len(result.placebo_features), 20)
        self.assertEqual(result.placebo_diagnostics["candidate_count"].iloc[0], 24)
        for _, members in result.placebo_memberships.groupby("draw_id"):
            self.assertEqual(members["code"].nunique(), 12)


if __name__ == "__main__":
    unittest.main()
