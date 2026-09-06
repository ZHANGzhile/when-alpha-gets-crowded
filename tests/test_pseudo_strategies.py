import unittest

import pandas as pd

from alpha_crowding.experiments import (
    build_continuous_permuted_memberships,
    stable_pseudo_seed,
)


def candidate_panel() -> pd.DataFrame:
    rows = []
    for date_number, date in enumerate(("2024-01-05", "2024-01-12")):
        for code_number in range(20):
            rows.append(
                {
                    "date": date,
                    "factor": "MOM",
                    "code": f"s{code_number:02d}",
                    "industry": "A" if code_number < 10 else "B",
                    "raw_signal": code_number + 100 * date_number,
                }
            )
    return pd.DataFrame(rows)


class ContinuousPseudoStrategyTests(unittest.TestCase):
    def test_strategy_identity_is_stable_across_weeks(self):
        result = build_continuous_permuted_memberships(
            candidate_panel(), strategy_count=3, base_seed=11
        )
        audit = result.diagnostics
        self.assertEqual(set(audit.pseudo_strategy_id), {"pseudo_000", "pseudo_001", "pseudo_002"})
        counts = audit.groupby("pseudo_strategy_id")["decision_at"].nunique()
        self.assertTrue(counts.eq(2).all())
        self.assertTrue(audit.valid.all())
        self.assertTrue(audit.long_short_overlap.eq(0).all())

    def test_memberships_are_reproducible_and_legs_are_valid(self):
        first = build_continuous_permuted_memberships(
            candidate_panel(), strategy_count=2, base_seed=19
        )
        second = build_continuous_permuted_memberships(
            candidate_panel(), strategy_count=2, base_seed=19
        )
        self.assertTrue(first.memberships.equals(second.memberships))
        sums = first.memberships.groupby(["date", "strategy_key", "leg"])["weight"].sum()
        self.assertTrue(sums.round(12).eq(1.0).all())
        overlap = first.memberships.groupby(["date", "strategy_key", "code"])["leg"].nunique()
        self.assertTrue(overlap.le(1).all())

    def test_seed_changes_with_strategy_date_and_factor(self):
        base = stable_pseudo_seed(
            1, pseudo_strategy_id="pseudo_000", decision_at="2024-01-05", factor="MOM"
        )
        alternatives = {
            stable_pseudo_seed(
                1, pseudo_strategy_id="pseudo_001", decision_at="2024-01-05", factor="MOM"
            ),
            stable_pseudo_seed(
                1, pseudo_strategy_id="pseudo_000", decision_at="2024-01-12", factor="MOM"
            ),
            stable_pseudo_seed(
                1, pseudo_strategy_id="pseudo_000", decision_at="2024-01-05", factor="REV"
            ),
        }
        self.assertNotIn(base, alternatives)
        self.assertEqual(len(alternatives), 3)


if __name__ == "__main__":
    unittest.main()
