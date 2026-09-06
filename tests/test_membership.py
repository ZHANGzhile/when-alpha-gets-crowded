import unittest

import pandas as pd

from alpha_crowding.data import (
    classify_missing_membership_market_rows,
    repair_merger_membership_gaps,
    validate_constituent_snapshot,
    weekly_last_sessions,
)


class MembershipTests(unittest.TestCase):
    def test_weekly_schedule_uses_last_actual_session(self):
        calendar = pd.DataFrame(
            {
                "calendar_date": pd.date_range("2024-01-01", "2024-01-12"),
                "is_trading_day": [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
            }
        )
        result = weekly_last_sessions(calendar)
        self.assertEqual(result.tolist(), [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-11")])

    def test_snapshot_rejects_future_update(self):
        frame = pd.DataFrame(
            {
                "index": ["CSI300"] * 300,
                "requested_date": ["2024-01-05"] * 300,
                "updateDate": ["2024-01-02"] * 299 + ["2024-01-08"],
                "code": [f"code{i:03d}" for i in range(300)],
            }
        )
        with self.assertRaisesRegex(ValueError, "future"):
            validate_constituent_snapshot(frame, index="CSI300", requested_date="2024-01-05")

    def test_snapshot_requires_exact_index_size(self):
        frame = pd.DataFrame(
            {
                "index": ["CSI500"], "requested_date": ["2024-01-05"],
                "updateDate": ["2024-01-02"], "code": ["sh.600000"],
            }
        )
        with self.assertRaisesRegex(ValueError, "expected 500"):
            validate_constituent_snapshot(frame, index="CSI500", requested_date="2024-01-05")

    def test_official_merger_repair_uses_predecessor_then_successor(self):
        rows = []
        for date in ("2019-01-11", "2019-01-18"):
            rows.extend(
                {
                    "index": "CSI500", "requested_date": date,
                    "updateDate": "2019-01-07", "code": f"code{i:03d}",
                    "code_name": f"name{i:03d}",
                }
                for i in range(499)
            )
        rows.extend(
            [
                {"index": "CSI500", "requested_date": "2019-01-04", "updateDate": "2018-12-24", "code": "old", "code_name": "old name"},
                {"index": "CSI500", "requested_date": "2019-01-25", "updateDate": "2019-01-21", "code": "new", "code_name": "new name"},
            ]
        )
        event = {
            "index": "CSI500", "provider_gap_start": "2019-01-07",
            "provider_successor_visible": "2019-01-21",
            "official_successor_effective": "2019-01-18",
            "announcement_available": "2018-12-17",
            "predecessor_code": "old", "successor_code": "new",
            "source_url": "https://example.test/official",
        }
        repaired, audit = repair_merger_membership_gaps(pd.DataFrame(rows), [event])
        jan11 = repaired[repaired["requested_date"].eq(pd.Timestamp("2019-01-11"))]
        jan18 = repaired[repaired["requested_date"].eq(pd.Timestamp("2019-01-18"))]
        self.assertIn("old", set(jan11["code"]))
        self.assertIn("new", set(jan18["code"]))
        self.assertEqual(len(audit), 2)
        self.assertTrue(
            repaired.loc[repaired["code"].isin(["old", "new"]), "membership_source"]
            .eq("official_index_announcement")
            .any()
        )

    def test_only_pre_effective_official_row_may_lack_market_record(self):
        merged = pd.DataFrame(
            {
                "date": ["2021-09-24", "2021-09-30", "2021-09-24"],
                "membership_source": [
                    "official_index_announcement",
                    "official_index_announcement",
                    "baostock",
                ],
                "effectiveDate": ["2021-09-28", "2021-09-28", "2021-09-13"],
                "_merge": ["left_only", "left_only", "left_only"],
            }
        )
        documented, unexplained = classify_missing_membership_market_rows(merged)
        self.assertEqual(documented.tolist(), [True, False, False])
        self.assertEqual(unexplained.tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()
