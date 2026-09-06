import unittest

import pandas as pd

from alpha_crowding.measurement import causal_rank_ic_state, trailing_return_state


class MarketFactorStateTests(unittest.TestCase):
    def test_trailing_state_never_reads_after_decision(self):
        calendar = pd.bdate_range("2024-01-02", periods=70)
        returns = pd.Series(0.01, index=calendar)
        state = trailing_return_state(
            returns,
            calendar,
            decision_at=calendar[59],
            drawdown_sessions=60,
        )
        self.assertEqual(state["window_end"], calendar[59])
        self.assertAlmostEqual(state["trailing_return"], 1.01**20 - 1.0)
        self.assertAlmostEqual(state["current_drawdown"], 0.0)

    def test_rank_ic_requires_realization_before_decision(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        observations = pd.DataFrame(
            {
                "signal_at": dates[:3],
                "realized_at": dates[2:5],
                "rank_ic": [0.1, 0.2, 100.0],
            }
        )
        state = causal_rank_ic_state(
            observations,
            pd.DatetimeIndex([dates[3]]),
            lookback_observations=3,
            minimum_observations=2,
        )
        self.assertAlmostEqual(state.iloc[0]["rank_ic_mean"], 0.15)
        self.assertEqual(state.iloc[0]["rank_ic_latest_realized_at"], dates[3])


if __name__ == "__main__":
    unittest.main()
