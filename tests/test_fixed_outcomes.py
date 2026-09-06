import unittest

import pandas as pd

from alpha_crowding.factors import MissingHeldReturnError
from alpha_crowding.outcomes import fixed_membership_leg_path


class FixedMembershipOutcomeTests(unittest.TestCase):
    def test_next_session_start_and_weight_drift(self):
        calendar = pd.bdate_range("2024-01-02", periods=4)
        members = pd.DataFrame({"code": ["a", "b"], "weight": [0.5, 0.5]})
        returns = pd.DataFrame(
            {
                "date": [calendar[1], calendar[1], calendar[2], calendar[2]],
                "code": ["a", "b", "a", "b"],
                "daily_return": [1.0, 0.0, 0.0, 0.0],
            }
        )
        path = fixed_membership_leg_path(
            members, returns, calendar, decision_at=calendar[0], horizon_sessions=2
        )
        self.assertEqual(path.iloc[0]["date"], calendar[1])
        self.assertAlmostEqual(path.iloc[0]["daily_return"], 0.5)
        self.assertAlmostEqual(path.iloc[1]["daily_return"], 0.0)

    def test_missing_held_return_is_not_filled(self):
        calendar = pd.bdate_range("2024-01-02", periods=3)
        members = pd.DataFrame({"code": ["a", "b"], "weight": [0.5, 0.5]})
        returns = pd.DataFrame(
            {"date": [calendar[1]], "code": ["a"], "daily_return": [0.0]}
        )
        with self.assertRaises(MissingHeldReturnError):
            fixed_membership_leg_path(
                members, returns, calendar, decision_at=calendar[0], horizon_sessions=1
            )


if __name__ == "__main__":
    unittest.main()
