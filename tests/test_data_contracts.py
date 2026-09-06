import unittest

import pandas as pd

from alpha_crowding.data import (
    assert_information_available,
    assert_unique_key,
    validate_interval,
)


class DataContractTests(unittest.TestCase):
    def test_rejects_future_availability(self):
        frame = pd.DataFrame(
            {
                "available_at": ["2024-01-08", "2024-01-10"],
                "decision_at": ["2024-01-08", "2024-01-09"],
            }
        )
        with self.assertRaisesRegex(ValueError, "point-in-time violations"):
            assert_information_available(frame)

    def test_accepts_information_available_at_decision(self):
        frame = pd.DataFrame(
            {"available_at": ["2024-01-08"], "decision_at": ["2024-01-08"]}
        )
        assert_information_available(frame)

    def test_unique_key(self):
        frame = pd.DataFrame({"date": [1, 1], "code": ["A", "A"]})
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            assert_unique_key(frame, ["date", "code"])

    def test_interval_maturity(self):
        frame = pd.DataFrame(
            {
                "label_start_at": ["2024-01-02"],
                "label_end_at": ["2024-01-31"],
                "cutoff": ["2024-01-30"],
            }
        )
        with self.assertRaisesRegex(ValueError, "immature"):
            validate_interval(frame, cutoff_col="cutoff")


if __name__ == "__main__":
    unittest.main()

