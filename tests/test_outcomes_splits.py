from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_crowding.experiments.splits import (  # noqa: E402
    assert_lead_time_information_cutoff,
    assert_no_label_interval_overlap,
    lead_time_cutoffs,
    purge_training_rows,
    purged_training_mask,
)
from alpha_crowding.outcomes import (  # noqa: E402
    MatureTailSpec,
    active_drawdown_series,
    active_mdd,
    build_mature_tail_labels,
    relative_nav,
)


class MatureTailLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = pd.bdate_range("2024-01-02", periods=18)

    def test_only_mature_same_family_history_is_used(self) -> None:
        s = self.sessions
        outcomes = pd.DataFrame(
            {
                "factor": ["MOM", "MOM", "MOM", "MOM", "MOM"],
                "target_family": ["research_ls"] * 5,
                "membership_mode": [
                    "dynamic",
                    "dynamic",
                    "dynamic",
                    "fixed",
                    "fixed",
                ],
                "decision_at": [s[0], s[2], s[8], s[0], s[8]],
                "information_cutoff_at": [s[0], s[2], s[6], s[0], s[6]],
                "label_start_at": [s[1], s[3], s[9], s[1], s[9]],
                "label_end_at": [s[3], s[8], s[12], s[3], s[12]],
                "future_return": [-0.10, -0.50, -0.20, -0.30, -0.20],
            }
        )
        result = build_mature_tail_labels(
            outcomes,
            sessions=s,
            information_cutoff_col="information_cutoff_at",
            spec=MatureTailSpec(
                quantile=0.10,
                lookback_sessions=10,
                min_history=1,
            ),
        )

        dynamic_current = result.iloc[2]
        self.assertEqual(dynamic_current["mature_history_count"], 1)
        self.assertAlmostEqual(dynamic_current["historical_tail_threshold"], -0.10)
        self.assertTrue(bool(dynamic_current["tail_event"]))

        fixed_current = result.iloc[4]
        self.assertEqual(fixed_current["mature_history_count"], 1)
        self.assertAlmostEqual(fixed_current["historical_tail_threshold"], -0.30)
        self.assertFalse(bool(fixed_current["tail_event"]))

        self.assertTrue(pd.isna(result.iloc[0]["tail_event"]))
        self.assertTrue(pd.isna(result.iloc[1]["tail_event"]))

    def test_history_outside_session_lookback_is_excluded(self) -> None:
        s = self.sessions
        outcomes = pd.DataFrame(
            {
                "factor": ["MOM", "MOM"],
                "target_family": ["research_ls", "research_ls"],
                "membership_mode": ["dynamic", "dynamic"],
                "decision_at": [s[0], s[12]],
                "label_start_at": [s[1], s[13]],
                "label_end_at": [s[2], s[16]],
                "future_return": [-0.2, -0.3],
            }
        )
        result = build_mature_tail_labels(
            outcomes,
            sessions=s,
            spec=MatureTailSpec(lookback_sessions=5, min_history=1),
        )
        self.assertEqual(result.iloc[1]["mature_history_count"], 0)
        self.assertTrue(pd.isna(result.iloc[1]["tail_event"]))

    def test_continuous_pseudo_thresholds_do_not_cross_strategy_identity(self) -> None:
        s = self.sessions
        outcomes = pd.DataFrame(
            {
                "original_factor": ["MOM"] * 4,
                "pseudo_strategy_id": ["p0", "p0", "p1", "p1"],
                "target_family": ["research_ls"] * 4,
                "membership_mode": ["continuous_pseudo_dynamic"] * 4,
                "horizon_sessions": [2] * 4,
                "decision_at": [s[0], s[4], s[0], s[4]],
                "label_start_at": [s[1], s[5], s[1], s[5]],
                "label_end_at": [s[2], s[6], s[2], s[6]],
                "future_return": [-0.10, -0.20, -0.80, -0.20],
            }
        )
        result = build_mature_tail_labels(
            outcomes,
            sessions=s,
            group_cols=(
                "original_factor",
                "pseudo_strategy_id",
                "target_family",
                "membership_mode",
                "horizon_sessions",
            ),
            spec=MatureTailSpec(lookback_sessions=10, min_history=1),
        )
        current = result.loc[result["decision_at"].eq(s[4])].set_index(
            "pseudo_strategy_id"
        )
        self.assertAlmostEqual(current.loc["p0", "historical_tail_threshold"], -0.10)
        self.assertAlmostEqual(current.loc["p1", "historical_tail_threshold"], -0.80)
        self.assertTrue(bool(current.loc["p0", "tail_event"]))
        self.assertFalse(bool(current.loc["p1", "tail_event"]))


class ActiveRiskTests(unittest.TestCase):
    def test_relative_nav_is_ratio_of_compounded_wealth(self) -> None:
        index = pd.date_range("2024-01-01", periods=2)
        portfolio = pd.Series([0.02, -0.01], index=index)
        benchmark = pd.Series([0.01, 0.00], index=index)
        actual = relative_nav(portfolio, benchmark)
        expected = np.array([1.02 / 1.01, (1.02 * 0.99) / 1.01])
        np.testing.assert_allclose(actual.to_numpy(), expected)
        self.assertTrue(actual.index.equals(index))

    def test_active_mdd_includes_initial_nav_peak(self) -> None:
        portfolio = np.array([-0.10, 0.10])
        benchmark = np.array([0.0, 0.0])
        drawdown = active_drawdown_series(portfolio, benchmark)
        np.testing.assert_allclose(drawdown, np.array([-0.10, -0.01]))
        self.assertAlmostEqual(active_mdd(portfolio, benchmark), 0.10)

    def test_misaligned_series_are_rejected(self) -> None:
        left = pd.Series([0.0], index=[pd.Timestamp("2024-01-01")])
        right = pd.Series([0.0], index=[pd.Timestamp("2024-01-02")])
        with self.assertRaises(ValueError):
            relative_nav(left, right)


class IntervalPurgeTests(unittest.TestCase):
    def test_purge_uses_closed_real_label_intervals_and_maturity(self) -> None:
        training = pd.DataFrame(
            {
                "label_start_at": pd.to_datetime(
                    ["2024-01-02", "2024-01-04", "2024-01-08"]
                ),
                "label_end_at": pd.to_datetime(
                    ["2024-01-05", "2024-01-09", "2024-01-12"]
                ),
            },
            index=["safe", "boundary_overlap", "immature"],
        )
        protected = pd.DataFrame(
            {
                "label_start_at": pd.to_datetime(["2024-01-09"]),
                "label_end_at": pd.to_datetime(["2024-01-15"]),
            }
        )
        mask = purged_training_mask(
            training, protected, fit_cutoff="2024-01-10"
        )
        self.assertEqual(mask.to_dict(), {
            "safe": True,
            "boundary_overlap": False,
            "immature": False,
        })
        kept = purge_training_rows(
            training, protected, fit_cutoff="2024-01-10"
        )
        self.assertEqual(kept.index.tolist(), ["safe"])
        with self.assertRaises(ValueError):
            assert_no_label_interval_overlap(training, protected)


class LeadTimeCutoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = pd.bdate_range("2024-01-02", periods=20)

    def _valid_frame(self) -> pd.DataFrame:
        s = self.sessions
        return pd.DataFrame(
            {
                "decision_at": [s[10]],
                "information_cutoff_at": [s[5]],
                "feature_available_at": [s[5]],
                "membership_available_at": [s[4]],
                "threshold_available_at": [s[5]],
                "model_fit_at": [s[3]],
                "label_start_at": [s[11]],
            },
            index=["row"],
        )

    def test_exact_five_session_cutoff_and_vintages_pass(self) -> None:
        frame = self._valid_frame()
        assert_lead_time_information_cutoff(
            frame,
            sessions=self.sessions,
            lead_sessions=5,
            available_at_cols=(
                "feature_available_at",
                "membership_available_at",
                "threshold_available_at",
                "model_fit_at",
            ),
        )
        expected = lead_time_cutoffs(
            frame["decision_at"], sessions=self.sessions, lead_sessions=5
        )
        self.assertEqual(expected.iloc[0], self.sessions[5])

    def test_post_cutoff_feature_is_rejected(self) -> None:
        frame = self._valid_frame()
        frame.loc["row", "feature_available_at"] = self.sessions[6]
        with self.assertRaisesRegex(ValueError, "feature_available_at"):
            assert_lead_time_information_cutoff(
                frame,
                sessions=self.sessions,
                lead_sessions=5,
                available_at_cols=("feature_available_at",),
            )

    def test_older_cutoff_requires_non_exact_mode(self) -> None:
        frame = self._valid_frame()
        frame.loc["row", "information_cutoff_at"] = self.sessions[4]
        frame.loc["row", "feature_available_at"] = self.sessions[4]
        frame.loc["row", "threshold_available_at"] = self.sessions[4]
        with self.assertRaises(ValueError):
            assert_lead_time_information_cutoff(
                frame,
                sessions=self.sessions,
                lead_sessions=5,
                available_at_cols=("feature_available_at",),
            )
        assert_lead_time_information_cutoff(
            frame,
            sessions=self.sessions,
            lead_sessions=5,
            available_at_cols=("feature_available_at", "threshold_available_at"),
            require_exact_cutoff=False,
        )


if __name__ == "__main__":
    unittest.main()
