import unittest

import pandas as pd

from alpha_crowding.factors import (
    MissingHeldReturnError,
    combine_long_short_returns,
    simulate_factor_leg_returns,
)


class FactorReturnTests(unittest.TestCase):
    def test_next_session_execution_and_weight_drift(self):
        memberships = pd.DataFrame(
            {
                "date": ["2024-01-05"] * 4,
                "factor": ["MOM"] * 4,
                "leg": ["LONG", "LONG", "SHORT", "SHORT"],
                "code": ["a", "b", "c", "d"],
                "weight": [0.5, 0.5, 0.5, 0.5],
            }
        )
        returns = pd.DataFrame(
            {
                "date": ["2024-01-08"] * 4 + ["2024-01-09"] * 4,
                "code": ["a", "b", "c", "d"] * 2,
                "daily_return": [0.10, 0.0, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0],
            }
        )
        legs = simulate_factor_leg_returns(
            memberships, returns, ["2024-01-05", "2024-01-08", "2024-01-09"]
        )
        long = legs[legs["leg"].eq("LONG")].reset_index(drop=True)
        self.assertEqual(long.loc[0, "date"], pd.Timestamp("2024-01-08"))
        self.assertAlmostEqual(long.loc[0, "daily_return"], 0.05)
        self.assertAlmostEqual(long.loc[1, "daily_return"], 0.5 / 1.05 * 0.10)
        combined = combine_long_short_returns(legs)
        self.assertAlmostEqual(combined.loc[0, "long_short_return"], 0.05)

    def test_missing_held_return_is_a_hard_failure(self):
        memberships = pd.DataFrame(
            {
                "date": ["2024-01-05", "2024-01-05"],
                "factor": ["MOM", "MOM"],
                "leg": ["LONG", "SHORT"],
                "code": ["delisted", "other"],
                "weight": [1.0, 1.0],
            }
        )
        returns = pd.DataFrame(
            {
                "date": ["2024-01-08"],
                "code": ["other"],
                "daily_return": [0.0],
            }
        )
        with self.assertRaisesRegex(MissingHeldReturnError, "delisted"):
            simulate_factor_leg_returns(
                memberships, returns, ["2024-01-05", "2024-01-08"]
            )


if __name__ == "__main__":
    unittest.main()
