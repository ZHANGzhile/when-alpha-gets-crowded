import unittest

import pandas as pd

from alpha_crowding.data import join_industry_asof


class IndustryAsofTests(unittest.TestCase):
    def test_join_never_uses_future_snapshot_and_tracks_age(self):
        weekly = pd.DataFrame(
            {"date": ["2024-02-02", "2024-03-15"], "code": ["x", "x"]}
        )
        industry = pd.DataFrame(
            {
                "requested_date": ["2024-01-31", "2024-02-29"],
                "code": ["x", "x"], "industry": ["old", "new"],
                "stable_sector": ["a", "b"],
            }
        )
        result = join_industry_asof(weekly, industry, maximum_age_days=10)
        self.assertEqual(result.industry.tolist(), ["old", "new"])
        self.assertEqual(result.industry_age_days.tolist(), [2, 15])
        self.assertEqual(result.industry_is_fresh.tolist(), [True, False])

    def test_duplicate_industry_snapshot_is_rejected(self):
        weekly = pd.DataFrame({"date": ["2024-02-02"], "code": ["x"]})
        industry = pd.DataFrame(
            {
                "requested_date": ["2024-01-31"] * 2, "code": ["x"] * 2,
                "industry": ["a", "b"], "stable_sector": ["a", "b"],
            }
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            join_industry_asof(weekly, industry)


if __name__ == "__main__":
    unittest.main()
