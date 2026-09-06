import unittest

import pandas as pd

from alpha_crowding.measurement import draw_matched_placebos, stable_group_seed


class MeasurementPlaceboTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame(
            {
                "code": [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)],
                "industry": ["A"] * 8 + ["B"] * 8,
                "lagged_liquidity": list(range(1, 9)) + list(range(11, 19)),
            }
        )

    def test_draws_preserve_industry_counts_and_no_replacement(self):
        result = draw_matched_placebos(
            self.candidates,
            ["a2", "a5", "b2", "b5"],
            draws=10,
            seed=7,
            maximum_mean_absolute_z_difference=2.0,
        )
        self.assertTrue(result.diagnostics["valid"].all())
        for _, draw in result.memberships.groupby("draw_id"):
            self.assertEqual(draw["code"].nunique(), 4)
            self.assertEqual(draw.groupby("industry").size().to_dict(), {"A": 2, "B": 2})

    def test_seed_is_stable_and_exclusion_removes_actual_members(self):
        seed = stable_group_seed(
            42, decision_at="2024-01-05", factor="MOM", leg="LONG"
        )
        first = draw_matched_placebos(
            self.candidates,
            ["a2", "a5", "b2", "b5"],
            draws=3,
            seed=seed,
            exclude_actual=True,
            maximum_mean_absolute_z_difference=3.0,
        )
        second = draw_matched_placebos(
            self.candidates,
            ["a2", "a5", "b2", "b5"],
            draws=3,
            seed=seed,
            exclude_actual=True,
            maximum_mean_absolute_z_difference=3.0,
        )
        self.assertTrue(first.memberships.equals(second.memberships))
        self.assertFalse(first.memberships["code"].isin(["a2", "a5", "b2", "b5"]).any())


if __name__ == "__main__":
    unittest.main()
