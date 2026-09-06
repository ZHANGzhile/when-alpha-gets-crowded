import unittest

import numpy as np
import pandas as pd

from alpha_crowding.experiments import (
    annual_expanding_folds,
    run_annual_walk_forward,
)


FEATURE_SETS = {
    "M0": ["market"],
    "M1": ["market", "state"],
    "M2": ["market", "state", "generic", "stress"],
    "M3": ["market", "state", "generic", "stress", "crowding"],
}


def make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    rows = []
    for date in pd.date_range("2018-01-05", "2022-12-30", freq="W-FRI"):
        for factor_number, factor in enumerate(("MOM", "REV", "LOWVOL", "VALUE")):
            x = rng.normal(size=5)
            logit = -0.4 + 0.7 * x[3] + 0.9 * x[4] + 0.15 * factor_number
            rows.append(
                {
                    "decision_at": date,
                    "label_start_at": date + pd.offsets.BDay(1),
                    "label_end_at": date + pd.offsets.BDay(20),
                    "factor": factor,
                    "market": x[0],
                    "state": x[1],
                    "generic": x[2],
                    "stress": x[3],
                    "crowding": x[4],
                    "target": int(rng.random() < 1 / (1 + np.exp(-logit))),
                }
            )
    return pd.DataFrame(rows)


class AnnualWalkForwardTests(unittest.TestCase):
    def test_folds_use_only_mature_past_labels(self):
        folds = annual_expanding_folds(
            make_panel(),
            evaluation_start="2020-01-01",
            evaluation_end="2021-12-31",
            raw_data_cutoff="2022-02-01",
        )
        self.assertEqual([fold.name for fold in folds], ["2020", "2021"])
        for fold in folds:
            self.assertTrue((fold.training.label_end_at <= fold.fit_cutoff).all())
            self.assertTrue((fold.training.decision_at < fold.evaluation_start).all())
            self.assertTrue(
                (fold.evaluation.decision_at >= fold.evaluation_start).all()
            )

    def test_raw_cutoff_removes_unmature_evaluation_rows(self):
        folds = annual_expanding_folds(
            make_panel(),
            evaluation_start="2022-01-01",
            evaluation_end="2022-12-31",
            raw_data_cutoff="2022-11-30",
        )
        self.assertEqual(len(folds), 1)
        self.assertTrue((folds[0].evaluation.label_end_at <= pd.Timestamp("2022-11-30")).all())
        self.assertLess(folds[0].evaluation.decision_at.max(), pd.Timestamp("2022-11-30"))

    def test_runner_emits_one_oos_prediction_per_key(self):
        result = run_annual_walk_forward(
            make_panel(),
            FEATURE_SETS,
            evaluation_start="2020-01-01",
            evaluation_end="2021-12-31",
            raw_data_cutoff="2022-02-01",
        )
        self.assertEqual(result.manifest["scheme"], "annual_expanding")
        self.assertEqual(len(result.manifest["folds"]), 2)
        self.assertFalse(result.predictions.duplicated(["decision_at", "factor"]).any())
        self.assertEqual(set(result.predictions["fold"]), {"2020", "2021"})
        for model in FEATURE_SETS:
            self.assertTrue(result.predictions[f"probability_{model}"].between(0, 1).all())

    def test_duplicate_panel_key_is_rejected(self):
        frame = make_panel()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            annual_expanding_folds(
                frame,
                evaluation_start="2020-01-01",
                evaluation_end="2020-12-31",
            )

    def test_runner_tunes_regularization_inside_outer_training_only(self):
        result = run_annual_walk_forward(
            make_panel(),
            FEATURE_SETS,
            evaluation_start="2022-01-01",
            evaluation_end="2022-12-31",
            raw_data_cutoff="2023-02-01",
            c_grid=[0.1, 1.0],
            inner_validation_weeks=20,
            inner_splits=2,
            minimum_inner_training_weeks=80,
        )
        fold = result.manifest["folds"][0]
        search = fold["regularization_search"]
        self.assertEqual(search["c_grid"], [0.1, 1.0])
        self.assertEqual(set(search["selected_c_by_model"]), set(FEATURE_SETS))
        self.assertEqual(set(fold["c_values"]), set(FEATURE_SETS))


if __name__ == "__main__":
    unittest.main()
