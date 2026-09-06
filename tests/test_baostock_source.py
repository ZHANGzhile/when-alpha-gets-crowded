import unittest

from alpha_crowding.data.baostock_source import (
    BaoStockQueryError,
    fetch_index_bars,
    result_to_frame,
)


class FakeResult:
    def __init__(self, fields, rows, error_code="0", error_msg=""):
        self.fields = fields
        self.rows = rows
        self.error_code = error_code
        self.error_msg = error_msg
        self.position = -1

    def next(self):
        self.position += 1
        return self.position < len(self.rows)

    def get_row_data(self):
        return self.rows[self.position]


class BaoStockAdapterTests(unittest.TestCase):
    def test_materializes_result(self):
        result = FakeResult(["date", "code"], [["2024-01-01", "sh.600000"]])
        frame = result_to_frame(result, query_name="test")
        self.assertEqual(frame.to_dict("records"), [{"date": "2024-01-01", "code": "sh.600000"}])

    def test_rejects_protocol_error(self):
        result = FakeResult([], [], error_code="100", error_msg="network")
        with self.assertRaisesRegex(BaoStockQueryError, "network"):
            result_to_frame(result, query_name="test")

    def test_rejects_schema_row_mismatch(self):
        result = FakeResult(["date", "code"], [["2024-01-01"]])
        with self.assertRaisesRegex(BaoStockQueryError, "values"):
            result_to_frame(result, query_name="test")

    def test_index_query_uses_daily_unadjusted_fields(self):
        class FakeBaoStock:
            def __init__(self):
                self.call = None

            def query_history_k_data_plus(self, *args, **kwargs):
                self.call = (args, kwargs)
                return FakeResult(["date", "code"], [["2024-01-02", "sh.000906"]])

        source = FakeBaoStock()
        frame = fetch_index_bars(
            source,
            code="sh.000906",
            start_date="2012-01-01",
            end_date="2026-08-31",
        )
        self.assertEqual(frame.loc[0, "code"], "sh.000906")
        self.assertEqual(source.call[1]["frequency"], "d")
        self.assertEqual(source.call[1]["adjustflag"], "3")


if __name__ == "__main__":
    unittest.main()
