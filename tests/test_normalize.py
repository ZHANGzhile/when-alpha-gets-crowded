import unittest

import pandas as pd

from alpha_crowding.data import (
    add_lagged_matching_characteristics,
    normalize_baostock_daily,
)


def raw_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "code": ["x"] * 3,
            "open": [10, 10, 10.5], "close": [10, 10, 11],
            "preclose": [10 / 1.01, 10, 10], "volume": [100, None, 110],
            "amount": [1000, None, 1210], "turn": [1, None, 1],
            "tradestatus": [1, 0, 1], "pctChg": [1, None, 10],
            "pbMRQ": [2, 2, -1], "isST": [0, 0, 1],
        }
    )


class NormalizeTests(unittest.TestCase):
    def test_suspended_return_is_zero_but_row_is_not_tradable(self):
        result = normalize_baostock_daily(raw_rows())
        self.assertEqual(result.daily_return.tolist(), [0.01, 0.0, 0.10])
        self.assertAlmostEqual(result.return_index.iloc[-1], 1.111)
        self.assertFalse(result.eligible_for_new_position.iloc[1])
        self.assertFalse(result.eligible_for_new_position.iloc[2])
        reconstructed = (
            (1.0 + result.overnight_return)
            * (1.0 + result.intraday_return)
            - 1.0
        )
        self.assertTrue(reconstructed.round(10).eq(result.daily_return.round(10)).all())

    def test_float_shares_only_observed_on_valid_turnover_then_carried_forward(self):
        result = normalize_baostock_daily(raw_rows())
        self.assertEqual(result.float_shares_observed.iloc[0], 10000)
        self.assertTrue(pd.isna(result.float_shares_observed.iloc[1]))
        self.assertEqual(result.float_shares_estimate.iloc[1], 10000)
        self.assertTrue(pd.isna(result.pb_mrq.iloc[2]))

    def test_missing_return_on_tradable_row_is_rejected(self):
        frame = raw_rows()
        frame.loc[0, "pctChg"] = None
        with self.assertRaisesRegex(ValueError, "tradable"):
            normalize_baostock_daily(frame)

    def test_matching_liquidity_excludes_latest_twenty_days(self):
        dates = pd.date_range("2024-01-01", periods=45, freq="D")
        normalized = pd.DataFrame(
            {
                "date": dates,
                "code": ["x"] * 45,
                "daily_return": [0.01] * 44 + [9.0],
                "amount": [100.0] * 45,
                "float_market_cap": list(range(1, 46)),
            }
        )
        result = add_lagged_matching_characteristics(normalized)
        self.assertAlmostEqual(result.loc[44, "lagged_liquidity"], 0.0001)
        self.assertEqual(result.loc[44, "lagged_float_market_cap"], 25)


if __name__ == "__main__":
    unittest.main()
