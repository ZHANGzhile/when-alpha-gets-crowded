import unittest

import pandas as pd

from alpha_crowding.factors import build_factor_memberships


class FactorPortfolioTests(unittest.TestCase):
    def test_builds_exact_disjoint_legs_and_signed_weights(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-05"] * 20,
                "code": [f"s{i:02d}" for i in range(20)],
                "industry": ["A"] * 10 + ["B"] * 10,
                "base_eligible": True,
                "industry_is_fresh": True,
                "momentum": range(20),
            }
        )
        result = build_factor_memberships(
            frame, factors={"MOM": "momentum"}, enabled_factors=["MOM"]
        )
        legs = result[result.leg != "MIDDLE"]
        self.assertEqual((legs.leg == "LONG").sum(), 2)
        self.assertEqual((legs.leg == "SHORT").sum(), 2)
        self.assertAlmostEqual(legs[legs.leg == "LONG"].signed_weight.sum(), 1.0)
        self.assertAlmostEqual(legs[legs.leg == "SHORT"].signed_weight.sum(), -1.0)

    def test_stale_industry_and_ineligible_rows_are_removed_before_ranking(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-05"] * 4,
                "code": ["a", "b", "c", "d"],
                "industry": ["A"] * 4,
                "base_eligible": [True, True, False, True],
                "industry_is_fresh": [True, False, True, True],
                "momentum": [1, 2, 3, 4],
            }
        )
        result = build_factor_memberships(
            frame,
            factors={"MOM": "momentum"},
            enabled_factors=["MOM"],
            quantile=0.4,
        )
        self.assertEqual(result.code.tolist(), ["a", "d"])


if __name__ == "__main__":
    unittest.main()
