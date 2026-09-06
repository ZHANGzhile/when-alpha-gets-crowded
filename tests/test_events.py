import unittest

import pandas as pd

from alpha_crowding.outcomes import (
    build_event_study_panel,
    first_threshold_breach_at,
    merge_crash_episodes,
)


def event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": ["MOM"] * 4 + ["REV"],
            "target_family": ["research_ls"] * 5,
            "membership_mode": ["dynamic"] * 5,
            "tail_event": [True] * 5,
            "breach_at": [
                "2024-01-10",
                "2024-01-24",
                "2024-04-03",
                "2024-08-07",
                "2024-01-24",
            ],
            "label_start_at": [
                "2024-01-08",
                "2024-01-22",
                "2024-04-01",
                "2024-08-05",
                "2024-01-22",
            ],
            "label_end_at": [
                "2024-02-02",
                "2024-02-16",
                "2024-04-26",
                "2024-08-30",
                "2024-02-16",
            ],
        }
    )


class CrashEpisodeTests(unittest.TestCase):
    def test_overlapping_intervals_merge_before_windows(self):
        episodes = merge_crash_episodes(event_frame())
        mom = episodes[episodes.factor == "MOM"]
        self.assertEqual(len(mom), 2)
        first = mom.iloc[0]
        self.assertEqual(first.source_event_count, 3)
        self.assertEqual(first.event_at, pd.Timestamp("2024-01-10"))
        self.assertEqual(first.interval_end_at, pd.Timestamp("2024-04-26"))
        self.assertEqual(first.window_end_at, pd.Timestamp("2024-05-01"))

    def test_factors_form_separate_episodes(self):
        episodes = merge_crash_episodes(event_frame())
        self.assertEqual(len(episodes[episodes.factor == "REV"]), 1)
        self.assertNotEqual(episodes.episode_id.iloc[0], episodes.episode_id.iloc[-1])

    def test_breach_must_be_inside_realized_interval(self):
        frame = event_frame().iloc[[0]].copy()
        frame.loc[frame.index[0], "breach_at"] = "2024-03-01"
        with self.assertRaisesRegex(ValueError, "inside"):
            merge_crash_episodes(frame)

    def test_event_panel_aligns_daily_breach_to_next_weekly_anchor(self):
        episodes = merge_crash_episodes(event_frame().iloc[[0]])
        dates = pd.date_range("2023-11-17", "2024-02-16", freq="W-FRI")
        panel = pd.DataFrame(
            {
                "factor": "MOM",
                "target_family": "research_ls",
                "membership_mode": "dynamic",
                "decision_at": dates,
                "crowding": range(len(dates)),
            }
        )
        study = build_event_study_panel(panel, episodes)
        zero = study[study.relative_week == 0].iloc[0]
        self.assertEqual(zero.decision_at, pd.Timestamp("2024-01-12"))
        self.assertEqual(zero.aligned_event_at, pd.Timestamp("2024-01-12"))
        self.assertTrue(study.relative_week.min() >= -8)
        self.assertTrue(study.relative_week.max() <= 4)

    def test_first_threshold_breach_uses_cumulative_path(self):
        sessions = pd.bdate_range("2024-01-02", periods=8)
        returns = pd.Series(
            [0.0, 0.0, -0.04, -0.07, 0.02, 0.01, 0.0, 0.0],
            index=sessions,
        )
        breach = first_threshold_breach_at(
            returns,
            sessions,
            decision_at=sessions[1],
            horizon_sessions=4,
            threshold=-0.10,
        )
        self.assertEqual(breach, sessions[3])

    def test_nonbreaching_path_is_rejected(self):
        sessions = pd.bdate_range("2024-01-02", periods=6)
        returns = pd.Series(0.01, index=sessions)
        with self.assertRaisesRegex(ValueError, "does not breach"):
            first_threshold_breach_at(
                returns,
                sessions,
                decision_at=sessions[0],
                horizon_sessions=3,
                threshold=-0.05,
            )


if __name__ == "__main__":
    unittest.main()
