import unittest

import numpy as np
import pandas as pd

from alpha_crowding.experiments import (
    binary_log_loss,
    common_complete_case_mask,
    fit_nested_logistic_models,
    moving_block_bootstrap_mean,
    temporal_regularization_search,
    weekly_average_loss,
)


FEATURE_SETS = {
    "M0": ["market"],
    "M1": ["market", "factor_state"],
    "M2": ["market", "factor_state", "generic", "stress"],
    "M3": ["market", "factor_state", "generic", "stress", "crowding"],
    "M4": ["market", "factor_state", "generic", "stress", "crowding", "interaction"],
}


def panel(n_weeks=80):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-03", periods=n_weeks, freq="5B")
    rows = []
    for date in dates:
        for factor in ("MOM", "REV", "LOWVOL", "VALUE"):
            values = rng.normal(size=5)
            crowding = values[4]
            stress = values[3]
            logit = -1.5 + 1.3 * crowding + 0.7 * stress + 0.8 * crowding * stress
            target = int(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append(
                {
                    "decision_at": date,
                    "factor": factor,
                    "market": values[0],
                    "factor_state": values[1],
                    "generic": values[2],
                    "stress": stress,
                    "crowding": crowding,
                    "interaction": crowding * stress,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


class ModelingTests(unittest.TestCase):
    def test_common_mask_uses_union_of_all_model_features(self):
        frame = panel(5)
        frame.loc[0, "crowding"] = np.nan
        mask = common_complete_case_mask(
            frame,
            FEATURE_SETS,
            required_columns=["decision_at", "factor", "target"],
        )
        self.assertFalse(mask.iloc[0])
        self.assertEqual(mask.sum(), len(frame) - 1)

    def test_rejects_non_nested_feature_sets(self):
        frame = panel(5)
        with self.assertRaisesRegex(ValueError, "not nested"):
            common_complete_case_mask(
                frame,
                {"M0": ["market"], "M1": ["factor_state"]},
                required_columns=["target"],
            )

    def test_nested_models_share_evaluation_rows(self):
        frame = panel()
        split = frame.decision_at.sort_values().unique()[55]
        result = fit_nested_logistic_models(
            frame[frame.decision_at < split],
            frame[frame.decision_at >= split],
            FEATURE_SETS,
        )
        predictions = result.predictions
        for model in FEATURE_SETS:
            self.assertEqual(predictions[f"probability_{model}"].notna().sum(), len(predictions))
            self.assertTrue(predictions[f"probability_{model}"].between(0, 1).all())
        self.assertEqual(result.manifest["evaluation_rows"], len(predictions))

    def test_weekly_loss_averages_factors_first(self):
        frame = pd.DataFrame(
            {
                "decision_at": ["2024-01-05", "2024-01-05", "2024-01-12"],
                "factor": ["MOM", "REV", "MOM"],
                "target": [1, 0, 1],
                "p": [0.8, 0.2, 0.5],
            }
        )
        weekly = weekly_average_loss(frame, probability_col="p")
        expected = binary_log_loss([1, 0], [0.8, 0.2]).mean()
        self.assertAlmostEqual(weekly.iloc[0], expected)
        self.assertEqual(len(weekly), 2)

    def test_bootstrap_is_deterministic_and_reports_interval(self):
        values = pd.Series(np.linspace(0.01, 0.20, 60))
        first = moving_block_bootstrap_mean(values, block_length=8, repetitions=500, seed=4)
        second = moving_block_bootstrap_mean(values, block_length=8, repetitions=500, seed=4)
        self.assertEqual(first, second)
        self.assertGreater(first.estimate, 0)
        self.assertGreater(first.lower_95, 0)

    def test_temporal_regularization_uses_equal_search_budget(self):
        frame = panel(90)
        frame["label_start_at"] = frame.decision_at + pd.offsets.BDay(1)
        frame["label_end_at"] = frame.decision_at + pd.offsets.BDay(20)
        search = temporal_regularization_search(
            frame,
            FEATURE_SETS,
            c_grid=[0.1, 1.0, 10.0],
            validation_weeks=10,
            n_splits=2,
            minimum_training_weeks=50,
        )
        self.assertEqual(set(search.selected_c_by_model), set(FEATURE_SETS))
        self.assertEqual(len(search.scores), len(FEATURE_SETS) * 3)
        self.assertTrue(search.scores.validation_week_count.eq(20).all())
        self.assertEqual(len(search.manifest["splits"]), 2)

    def test_model_specific_c_mapping_must_match_models(self):
        frame = panel()
        split = frame.decision_at.sort_values().unique()[55]
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            fit_nested_logistic_models(
                frame[frame.decision_at < split],
                frame[frame.decision_at >= split],
                FEATURE_SETS,
                c_value={"M0": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
