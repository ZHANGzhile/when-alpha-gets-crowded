import unittest

import pandas as pd

from alpha_crowding.measurement import (
    measure_strategy_convergence_placebos,
    pairwise_jaccard,
)


class StrategyConvergenceTests(unittest.TestCase):
    def test_pairwise_jaccard_and_joint_draws(self):
        edges = pairwise_jaccard(
            {"MOM": {"a", "b"}, "REV": {"b", "c"}, "LOWVOL": {"a", "b"}}
        )
        self.assertEqual(len(edges), 3)
        self.assertAlmostEqual(edges["jaccard"].max(), 1.0)

        rows = []
        for factor, offset in (("MOM", 0), ("REV", 1), ("LOWVOL", 2)):
            for industry, prefix in (("A", "a"), ("B", "b")):
                for number in range(8):
                    rows.append(
                        {
                            "factor": factor,
                            "code": f"{prefix}{number}",
                            "industry": industry,
                            "lagged_liquidity": float(number + 1 + offset / 10),
                            "leg": "LONG" if number in {2, 5} else "MIDDLE",
                        }
                    )
        result = measure_strategy_convergence_placebos(
            pd.DataFrame(rows),
            decision_at="2024-01-05",
            leg="LONG",
            draws=10,
            minimum_valid_draws=8,
            maximum_mean_absolute_z_difference=3.0,
        )
        self.assertEqual(len(result.features), 3)
        self.assertTrue(result.features["placebo_valid"].all())
        self.assertEqual(len(result.placebo_features), 30)
        for draw_id, draw in result.placebo_memberships.groupby("draw_id"):
            self.assertEqual(draw["factor"].nunique(), 3, draw_id)


if __name__ == "__main__":
    unittest.main()
