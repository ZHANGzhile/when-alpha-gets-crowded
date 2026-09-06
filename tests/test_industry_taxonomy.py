import unittest

import pandas as pd

from alpha_crowding.data import (
    classify_industry_snapshot,
    industry_snapshot_summary,
    stable_industry_sector,
)


class IndustryTaxonomyTests(unittest.TestCase):
    def test_old_and_new_bank_labels_share_stable_sector(self):
        self.assertEqual(stable_industry_sector("金融保险业-银行业"), "finance")
        self.assertEqual(stable_industry_sector("J66货币金融服务"), "finance")
        self.assertEqual(stable_industry_sector("B06煤炭开采和洗选业"), "extractive")
        self.assertEqual(stable_industry_sector("M73研究和试验发展"), "broad_services")

    def test_code_prefix_detection_and_mapping(self):
        frame = pd.DataFrame(
            {"industry": ["制造业-医药制造业", "C27医药制造业", "未知门类"]}
        )
        result = classify_industry_snapshot(frame)
        self.assertEqual(result.taxonomy_has_code_prefix.tolist(), [False, True, False])
        self.assertEqual(result.stable_sector.iloc[:2].tolist(), ["manufacturing"] * 2)
        self.assertTrue(pd.isna(result.stable_sector.iloc[2]))

    def test_snapshot_summary_checks_asof_and_coverage(self):
        frame = pd.DataFrame(
            {
                "requested_date": ["2014-06-30", "2014-06-30"],
                "code": ["sh.600000", "sz.000001"],
                "updateDate": ["2014-06-27", "2014-07-01"],
                "industry": ["J66货币金融服务", "未知门类"],
            }
        )
        summary = industry_snapshot_summary(frame)
        self.assertEqual(summary["future_update_rows"], 1)
        self.assertEqual(summary["stable_mapping_coverage"], 0.5)
        self.assertEqual(summary["unmapped_labels"], ["未知门类"])

    def test_duplicate_security_is_rejected(self):
        frame = pd.DataFrame(
            {
                "requested_date": ["2014-06-30"] * 2,
                "code": ["sh.600000"] * 2,
                "updateDate": ["2014-06-27"] * 2,
                "industry": ["J66货币金融服务"] * 2,
            }
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            industry_snapshot_summary(frame)


if __name__ == "__main__":
    unittest.main()
