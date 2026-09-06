from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from alpha_crowding.measurement import (  # noqa: E402
    compute_crowding_state,
    compute_generic_risk,
    compute_stress_trigger,
    historical_zscore,
    historical_zscore_by_group,
    historical_zscore_by_group_trading_window,
    historical_zscore_trading_window,
    summarize_matched_placebo,
    summarize_matched_placebos,
)


class HistoricalZScoreTests(unittest.TestCase):
    def test_exact_trading_session_window_drops_older_weekly_observations(self) -> None:
        calendar = pd.bdate_range("2024-01-02", periods=31)
        decisions = calendar[[0, 5, 10, 15, 20, 25, 30]]
        values = pd.Series(range(1, 8), index=decisions, dtype=float)

        result = historical_zscore_trading_window(
            values,
            calendar,
            lookback_sessions=10,
            min_periods=2,
        )

        # At calendar position 30, only positions 20 and 25 are inside the
        # prior ten sessions.  The value at position 15 has aged out.
        self.assertEqual(result.iloc[-1]["history_n"], 2)
        self.assertAlmostEqual(result.iloc[-1]["historical_center"], 5.5)
        self.assertEqual(result.iloc[-1]["history_window_start"], calendar[20])
        self.assertEqual(result.iloc[-1]["history_window_end"], calendar[29])

    def test_grouped_exact_window_preserves_factor_boundaries(self) -> None:
        calendar = pd.bdate_range("2024-01-02", periods=16)
        dates = calendar[[0, 5, 10, 15]]
        frame = pd.DataFrame(
            {
                "factor": ["MOM"] * 4 + ["REV"] * 4,
                "leg": ["LONG"] * 8,
                "decision_at": list(dates) * 2,
                "excess": [1.0, 2.0, 3.0, 9.0, 10.0, 20.0, 30.0, 90.0],
            }
        )
        result = historical_zscore_by_group_trading_window(
            frame,
            calendar,
            value_col="excess",
            lookback_sessions=10,
            min_periods=2,
        )
        self.assertEqual(result.loc[3, "history_n"], 2)
        self.assertAlmostEqual(result.loc[3, "historical_center"], 2.5)
        self.assertAlmostEqual(result.loc[7, "historical_center"], 25.0)

    def test_current_observation_is_excluded(self) -> None:
        values = pd.Series(
            [1.0, 2.0, 3.0, 100.0],
            index=pd.date_range("2020-01-03", periods=4, freq="W-FRI"),
        )

        result = historical_zscore(values, lookback=3, min_periods=3)

        self.assertEqual(result.iloc[-1]["history_n"], 3)
        self.assertAlmostEqual(result.iloc[-1]["historical_center"], 2.0)
        self.assertAlmostEqual(result.iloc[-1]["historical_scale"], 1.0)
        self.assertAlmostEqual(result.iloc[-1]["historical_z"], 98.0)
        self.assertTrue(result.iloc[:3]["historical_z"].isna().all())

    def test_zero_historical_scale_stays_missing(self) -> None:
        values = pd.Series([1.0, 1.0, 1.0, 2.0])
        result = historical_zscore(values, lookback=3, min_periods=3)

        self.assertEqual(result.iloc[-1]["historical_scale"], 0.0)
        self.assertTrue(math.isnan(result.iloc[-1]["historical_z"]))

    def test_rejects_unsorted_or_infinite_input(self) -> None:
        unsorted = pd.Series(
            [1.0, 2.0], index=pd.to_datetime(["2020-01-10", "2020-01-03"])
        )
        with self.assertRaisesRegex(ValueError, "oldest to newest"):
            historical_zscore(unsorted, lookback=2, min_periods=2)

        with self.assertRaisesRegex(ValueError, "infinite"):
            historical_zscore(
                pd.Series([1.0, np.inf, 2.0]), lookback=2, min_periods=2
            )

    def test_grouped_history_does_not_cross_factor_or_leg(self) -> None:
        frame = pd.DataFrame(
            {
                "factor": ["MOM"] * 4 + ["VALUE"] * 4,
                "leg": ["long"] * 8,
                "decision_at": list(pd.date_range("2020-01-03", periods=4, freq="W-FRI"))
                * 2,
                "excess": [1.0, 2.0, 3.0, 10.0, 100.0, 200.0, 300.0, 1000.0],
            },
            index=[7, 2, 9, 1, 8, 3, 6, 0],
        )

        result = historical_zscore_by_group(
            frame,
            value_col="excess",
            lookback=3,
            min_periods=3,
        )

        self.assertAlmostEqual(result.loc[1, "historical_center"], 2.0)
        self.assertAlmostEqual(result.loc[1, "historical_z"], 8.0)
        self.assertAlmostEqual(result.loc[0, "historical_center"], 200.0)
        self.assertAlmostEqual(result.loc[0, "historical_z"], 8.0)


class MatchedPlaceboTests(unittest.TestCase):
    def test_valid_summary_reports_excess_and_placebo_z(self) -> None:
        draws = np.arange(1.0, 101.0)
        result = summarize_matched_placebo(60.5, draws)

        self.assertTrue(result.placebo_valid)
        self.assertEqual(result.valid_draws, 100)
        self.assertAlmostEqual(result.placebo_mean, 50.5)
        self.assertAlmostEqual(result.excess, 10.0)
        self.assertAlmostEqual(result.placebo_z, 10.0 / np.std(draws, ddof=1))
        self.assertIsNone(result.invalid_reason)

    def test_insufficient_draws_and_zero_scale_are_invalid(self) -> None:
        insufficient = np.concatenate([np.arange(79.0), np.full(21, np.nan)])
        result = summarize_matched_placebo(1.0, insufficient)
        self.assertFalse(result.placebo_valid)
        self.assertEqual(result.invalid_reason, "insufficient_valid_draws")
        self.assertTrue(math.isnan(result.placebo_z))

        constant = summarize_matched_placebo(2.0, np.ones(100))
        self.assertFalse(constant.placebo_valid)
        self.assertEqual(constant.invalid_reason, "zero_or_undefined_placebo_std")
        self.assertTrue(math.isnan(constant.placebo_z))

    def test_frame_summary_requires_exact_alignment(self) -> None:
        actual = pd.Series([1.0, 2.0], index=["a", "b"])
        draws = pd.DataFrame(
            np.tile(np.arange(100.0), (2, 1)), index=["a", "b"]
        )
        result = summarize_matched_placebos(actual, draws)
        self.assertTrue(result["placebo_valid"].all())
        self.assertEqual(result.loc["a", "valid_draws"], 100)

        with self.assertRaisesRegex(ValueError, "match exactly"):
            summarize_matched_placebos(actual, draws.iloc[::-1])

    def test_rejects_infinite_placebo_draw(self) -> None:
        draws = np.arange(100.0)
        draws[0] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            summarize_matched_placebo(1.0, draws)


class TransparentStateTests(unittest.TestCase):
    def test_crowding_is_equal_weight_and_complete_case(self) -> None:
        frame = pd.DataFrame(
            {
                "excess_sync_historical_z": [1.0, 1.0],
                "excess_eigen_historical_z": [2.0, np.nan],
                "excess_overlap_historical_z": [3.0, 3.0],
            },
            index=["complete", "missing"],
        )

        result = compute_crowding_state(frame)

        self.assertAlmostEqual(result.loc["complete", "crowding_state"], 2.0)
        self.assertTrue(result.loc["complete", "crowding_state_valid"])
        self.assertTrue(math.isnan(result.loc["missing", "crowding_state"]))
        self.assertFalse(result.loc["missing", "crowding_state_valid"])
        self.assertEqual(
            result.loc["missing", "crowding_state_component_count"], 2
        )

    def test_crowding_rejects_stress_component(self) -> None:
        frame = pd.DataFrame({"turnover_shock_z": [1.0]})
        with self.assertRaisesRegex(ValueError, "stress-like"):
            compute_crowding_state(frame, components=("turnover_shock_z",))

        disguised = pd.DataFrame({"innocent_storage_name": [1.0]})
        with self.assertRaisesRegex(ValueError, "stress-like"):
            compute_crowding_state(
                disguised,
                components=("turnover_shock_z",),
                column_map={"turnover_shock_z": "innocent_storage_name"},
            )

    def test_component_column_mapping_preserves_semantic_names(self) -> None:
        frame = pd.DataFrame(
            {
                "excess_residual_sync": [1.0],
                "excess_eigen": [2.0],
                "excess_strategy_convergence": [3.0],
            }
        )
        result = compute_crowding_state(
            frame,
            column_map={
                "excess_sync_historical_z": "excess_residual_sync",
                "excess_eigen_historical_z": "excess_eigen",
                "excess_overlap_historical_z": "excess_strategy_convergence",
            },
        )
        self.assertAlmostEqual(result.loc[0, "crowding_state"], 2.0)

    def test_generic_risk_rejects_placebo_adjusted_component(self) -> None:
        frame = pd.DataFrame({"excess_sync_z": [1.0]})
        with self.assertRaisesRegex(ValueError, "placebo-adjusted"):
            compute_generic_risk(frame, components=("excess_sync_z",))

    def test_generic_and_stress_use_explicit_fixed_weights(self) -> None:
        generic_frame = pd.DataFrame(
            {
                "raw_sync_z": [1.0],
                "raw_eigen_z": [2.0],
                "turnover_level_z": [3.0],
                "illiquidity_level_z": [4.0],
            }
        )
        generic = compute_generic_risk(
            generic_frame,
            weights={
                "raw_sync_z": 1.0,
                "raw_eigen_z": 1.0,
                "turnover_level_z": 0.0,
                "illiquidity_level_z": 0.0,
            },
        )
        self.assertAlmostEqual(generic.loc[0, "generic_risk"], 1.5)

        stress_frame = pd.DataFrame(
            {
                "turnover_shock_z": [1.0],
                "turnover_sync_z": [2.0],
                "liquidity_stress_z": [3.0],
                "factor_return_shock_z": [4.0],
            }
        )
        stress = compute_stress_trigger(stress_frame)
        self.assertAlmostEqual(stress.loc[0, "stress_trigger"], 2.5)

    def test_rejects_infinite_component_and_bad_weights(self) -> None:
        frame = pd.DataFrame(
            {
                "raw_sync_z": [np.inf],
                "raw_eigen_z": [2.0],
                "turnover_level_z": [3.0],
                "illiquidity_level_z": [4.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "infinite"):
            compute_generic_risk(frame)

        good = frame.replace(np.inf, 1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            compute_generic_risk(good, weights=[1.0, 1.0, -1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
