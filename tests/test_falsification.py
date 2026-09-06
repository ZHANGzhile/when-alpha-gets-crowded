import unittest

import numpy as np
import pandas as pd

from alpha_crowding.experiments import (
    discriminant_validity_table,
    multivariate_residual_diagnostic,
    time_misalign_features,
)


class TimeMisalignmentTests(unittest.TestCase):
    def test_shift_stays_inside_factor_and_records_source(self):
        dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        frame = pd.DataFrame(
            {
                "decision_at": list(dates) * 2,
                "factor": ["MOM"] * 4 + ["REV"] * 4,
                "crowding": [1, 2, 3, 4, 10, 20, 30, 40],
                "stress": range(8),
            }
        )
        result = time_misalign_features(frame, ["crowding"], lag_weeks=2)
        mom = result[result.factor.eq("MOM")].reset_index(drop=True)
        rev = result[result.factor.eq("REV")].reset_index(drop=True)
        self.assertTrue(pd.isna(mom.loc[0, "crowding"]))
        self.assertEqual(mom.loc[2, "crowding"], 1)
        self.assertEqual(rev.loc[2, "crowding"], 10)
        self.assertEqual(mom.loc[2, "misaligned_source_at"], dates[0])
        self.assertEqual(mom.loc[2, "stress"], 2)


class DiscriminantValidityTests(unittest.TestCase):
    def test_pairwise_table_exposes_relabelled_generic_risk(self):
        x = np.linspace(-2, 2, 80)
        frame = pd.DataFrame({"crowding": 3 * x + 1, "market_vol": x})
        result = discriminant_validity_table(
            frame, ["crowding"], ["market_vol"], minimum_observations=30
        ).iloc[0]
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.pearson, 1.0)
        self.assertAlmostEqual(result.r_squared, 1.0)

    def test_multivariate_diagnostic_reports_remaining_variation(self):
        rng = np.random.default_rng(7)
        x1 = rng.normal(size=120)
        x2 = rng.normal(size=120)
        crowding = 0.8 * x1 - 0.4 * x2 + rng.normal(scale=0.5, size=120)
        frame = pd.DataFrame({"crowding": crowding, "x1": x1, "x2": x2})
        result = multivariate_residual_diagnostic(
            frame, {"crowding": ["x1", "x2"]}, minimum_observations=50
        ).iloc[0]
        self.assertTrue(result.valid)
        self.assertGreater(result.r_squared, 0.5)
        self.assertGreater(result.residual_std, 0.0)


if __name__ == "__main__":
    unittest.main()
