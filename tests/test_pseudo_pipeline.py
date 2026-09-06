import unittest

import numpy as np
import pandas as pd

from alpha_crowding.measurement import (
    leave_one_out_placebo_adjustment,
    measure_continuous_pseudo_convergence,
    measure_continuous_pseudo_structure,
)


class LeaveOneOutAdjustmentTests(unittest.TestCase):
    def test_current_strategy_is_excluded_from_its_null(self):
        frame = pd.DataFrame(
            {
                "decision_at": "2024-01-05",
                "original_factor": "MOM",
                "leg": "LONG",
                "feature": "residual_sync",
                "pseudo_strategy_id": ["p0", "p1", "p2"],
                "actual": [1.0, 2.0, 4.0],
            }
        )
        result = leave_one_out_placebo_adjustment(
            frame, minimum_comparators=2
        ).set_index("pseudo_strategy_id")
        self.assertAlmostEqual(result.loc["p0", "placebo_mean"], 3.0)
        self.assertAlmostEqual(result.loc["p0", "excess"], -2.0)
        self.assertAlmostEqual(result.loc["p0", "placebo_std"], np.sqrt(2.0))
        self.assertTrue(result.placebo_valid.all())

    def test_zero_comparator_scale_remains_invalid(self):
        frame = pd.DataFrame(
            {
                "decision_at": "2024-01-05",
                "original_factor": "MOM",
                "leg": "LONG",
                "feature": "residual_sync",
                "pseudo_strategy_id": ["p0", "p1", "p2"],
                "actual": [1.0, 1.0, 1.0],
            }
        )
        result = leave_one_out_placebo_adjustment(frame, minimum_comparators=2)
        self.assertFalse(result.placebo_valid.any())
        self.assertTrue(
            result.invalid_reason.eq("zero_or_undefined_comparator_std").all()
        )


class ContinuousPseudoMeasurementTests(unittest.TestCase):
    def test_structure_and_joint_convergence_keep_pseudo_identity(self):
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2024-01-02", periods=45)
        codes = [f"s{i:02d}" for i in range(30)]
        candidates = pd.DataFrame(
            {
                "factor": np.repeat(["MOM", "REV", "LOWVOL"], 30),
                "code": codes * 3,
                "industry": (["A"] * 15 + ["B"] * 15) * 3,
            }
        )
        common = rng.normal(0, 0.01, size=(len(dates), 1))
        values = common + rng.normal(0, 0.004, size=(len(dates), len(codes)))
        daily = pd.DataFrame(
            {
                "date": np.repeat(dates, len(codes)),
                "code": np.tile(codes, len(dates)),
                "daily_return": values.ravel(),
            }
        )
        rows = []
        for pseudo, offset in (("p0", 0), ("p1", 2), ("p2", 4)):
            for factor_number, factor in enumerate(("MOM", "REV", "LOWVOL")):
                start = (offset + factor_number) % 8
                for leg, selected in (
                    ("LONG", codes[start : start + 10]),
                    ("SHORT", codes[15 + start : 25 + start]),
                ):
                    for code in selected:
                        rows.append(
                            {
                                "original_factor": factor,
                                "pseudo_strategy_id": pseudo,
                                "leg": leg,
                                "code": code,
                            }
                        )
        memberships = pd.DataFrame(rows)
        structural = measure_continuous_pseudo_structure(
            daily,
            candidates,
            memberships,
            decision_at=dates[-1],
            minimum_days=40,
            minimum_assets=8,
        )
        convergence = measure_continuous_pseudo_convergence(
            memberships, decision_at=dates[-1]
        )
        self.assertEqual(set(structural.feature), {"residual_sync", "eigen_concentration"})
        self.assertEqual(structural.pseudo_strategy_id.nunique(), 3)
        self.assertTrue(structural.measurement_valid.all())
        self.assertEqual(set(convergence.feature), {"strategy_convergence"})
        self.assertEqual(convergence.original_factor.nunique(), 3)
        self.assertEqual(convergence.pseudo_strategy_id.nunique(), 3)


if __name__ == "__main__":
    unittest.main()
