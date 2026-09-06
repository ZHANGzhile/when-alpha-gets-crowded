import unittest

import pandas as pd

from alpha_crowding.data import (
    compare_adjustments,
    daily_history_acceptance_failures,
    summarize_daily_history,
)


class DailyHistoryQualityTests(unittest.TestCase):
    def test_quality_summary_exposes_suspension_st_and_pb_gaps(self):
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "code": ["sh.600000"] * 2,
                "open": [10, 10], "high": [11, 10], "low": [9, 10], "close": [10, 10],
                "preclose": [9.5, 10], "volume": [1000, 0], "amount": [10000, 0],
                "turn": [1.0, 0.0], "tradestatus": [1, 0], "pctChg": [5.26, 0],
                "pbMRQ": [1.2, None], "isST": [0, 1],
            }
        )
        result = summarize_daily_history(frame)
        self.assertEqual(result["suspended_rows"], 1)
        self.assertEqual(result["st_rows"], 1)
        self.assertEqual(result["missing_pb_rows"], 1)
        self.assertEqual(result["implied_float_share_rows"], 1)

    def test_adjustment_comparison_separates_price_from_execution_fields(self):
        raw = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"], "code": ["x"] * 2,
                "close": [10, 5], "volume": [100, 200], "amount": [1000, 1000],
                "turn": [1, 2], "pctChg": [0, 0], "pbMRQ": [1, 1],
            }
        )
        adjusted = raw.copy()
        adjusted["close"] = [5, 5]
        result = compare_adjustments(raw, adjusted)
        self.assertEqual(result["adjustment_ratio_change_rows"], 1)
        self.assertTrue(all(result["invariant_fields"].values()))

    def test_acceptance_gate_rejects_integrity_failures(self):
        quality = {
            "status": "DATA",
            "rows": 10,
            "duplicate_dates": 1,
            "date_order_violations": 0,
            "zero_volume_traded_rows": 2,
            "bad_ohlc_order_rows": 0,
        }
        self.assertEqual(
            daily_history_acceptance_failures(quality),
            ["duplicate_dates=1", "zero_volume_traded_rows=2"],
        )


if __name__ == "__main__":
    unittest.main()
