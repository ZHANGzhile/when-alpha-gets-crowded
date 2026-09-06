import unittest

import pandas as pd

from alpha_crowding.experiments import attach_lead_features, build_lead_labels
from alpha_crowding.outcomes import MatureTailSpec


class LeadExperimentPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = pd.bdate_range("2024-01-02", periods=30)
        s = self.sessions
        self.outcomes = pd.DataFrame(
            {
                "factor": ["MOM"] * 3,
                "target_family": ["research_ls"] * 3,
                "membership_mode": ["dynamic"] * 3,
                "horizon_sessions": [2] * 3,
                "decision_at": [s[5], s[10], s[15]],
                "label_start_at": [s[6], s[11], s[16]],
                "label_end_at": [s[7], s[12], s[17]],
                "future_return": [-0.10, -0.50, -0.20],
            }
        )

    def test_threshold_is_rebuilt_at_issue_date(self) -> None:
        labels = build_lead_labels(
            self.outcomes,
            sessions=self.sessions,
            lead_sessions=5,
            spec=MatureTailSpec(quantile=0.10, lookback_sessions=30, min_history=1),
        )
        self.assertEqual(len(labels), 1)
        row = labels.iloc[0]
        self.assertEqual(row.target_anchor_at, self.sessions[15])
        self.assertEqual(row.decision_at, self.sessions[10])
        self.assertEqual(row.mature_history_count, 1)
        self.assertAlmostEqual(row.historical_tail_threshold, -0.10)
        self.assertEqual(row.target, 1)

    def test_latest_nonfuture_feature_vintage_is_attached(self) -> None:
        labels = build_lead_labels(
            self.outcomes,
            sessions=self.sessions,
            lead_sessions=5,
            spec=MatureTailSpec(quantile=0.10, lookback_sessions=30, min_history=1),
        )
        features = pd.DataFrame(
            {
                "decision_at": [self.sessions[5], self.sessions[9], self.sessions[11]],
                "factor": ["MOM"] * 3,
                "crowding": [1.0, 2.0, 999.0],
            }
        )
        panel = attach_lead_features(labels, features, sessions=self.sessions)
        self.assertEqual(panel.iloc[0].feature_source_at, self.sessions[9])
        self.assertEqual(panel.iloc[0].crowding, 2.0)
        self.assertEqual(panel.iloc[0].feature_staleness_sessions, 1)
        self.assertLessEqual(panel.iloc[0].feature_available_at, panel.iloc[0].decision_at)


if __name__ == "__main__":
    unittest.main()
