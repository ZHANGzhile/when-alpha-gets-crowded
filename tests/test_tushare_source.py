import unittest

import pandas as pd

from alpha_crowding.data import (
    TushareQueryError,
    fetch_csi800_weights,
    fetch_daily_stock_limits,
    tushare_code_to_internal,
)


class FakeClient:
    def index_weight(self, **kwargs):
        self.index_kwargs = kwargs
        return pd.DataFrame(
            {
                "index_code": ["000906.SH", "000906.SH"],
                "con_code": ["600000.SH", "000001.SZ"],
                "trade_date": ["20240131", "20240131"],
                "weight": [40.0, 60.0],
            }
        )

    def stk_limit(self, **kwargs):
        self.limit_kwargs = kwargs
        return pd.DataFrame(
            {
                "trade_date": ["20240201"],
                "ts_code": ["600000.SH"],
                "up_limit": [11.0],
                "down_limit": [9.0],
            }
        )


class TushareSourceTests(unittest.TestCase):
    def test_code_mapping_is_explicit(self):
        self.assertEqual(tushare_code_to_internal("600000.SH"), "sh.600000")
        self.assertEqual(tushare_code_to_internal("000001.SZ"), "sz.000001")
        with self.assertRaises(ValueError):
            tushare_code_to_internal("600000")

    def test_index_weights_convert_percent_units(self):
        client = FakeClient()
        result = fetch_csi800_weights(
            client,
            start_date="20240101",
            end_date="20240131",
        )
        self.assertEqual(client.index_kwargs["index_code"], "000906.SH")
        self.assertAlmostEqual(result["weight"].sum(), 1.0)
        self.assertEqual(set(result["code"]), {"sh.600000", "sz.000001"})

    def test_daily_limits_validate_bounds(self):
        client = FakeClient()
        result = fetch_daily_stock_limits(client, trade_date="20240201")
        self.assertEqual(client.limit_kwargs["trade_date"], "20240201")
        self.assertEqual(result.loc[0, "code"], "sh.600000")
        client.stk_limit = lambda **_: pd.DataFrame(
            {
                "trade_date": ["20240201"],
                "ts_code": ["600000.SH"],
                "up_limit": [8.0],
                "down_limit": [9.0],
            }
        )
        with self.assertRaisesRegex(TushareQueryError, "invalid"):
            fetch_daily_stock_limits(client, trade_date="20240201")


if __name__ == "__main__":
    unittest.main()
