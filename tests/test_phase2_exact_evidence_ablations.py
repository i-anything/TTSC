from __future__ import annotations

import unittest

from scripts.run_phase2_exact_evidence_ablations import (
    assign_disjoint_folds,
    build_fold_report,
    select_smoke_indices,
    validate_publication,
)


def _sample(index: int, scenario: str, target: str) -> dict:
    return {
        "sample_id": f"case-{index}",
        "scenario_type": scenario,
        "ground_truth": {"parent_asin": target},
        "user_profile": {},
    }


def _outcome(sample: dict, *, rank: int | None, turn: int | None) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": rank is not None,
        "first_hit_turn": turn,
        "best_rank": rank,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
    }


class Phase2ExactEvidenceAblationTests(unittest.TestCase):
    def test_smoke_selection_is_deterministic_and_balanced(self) -> None:
        scenarios = ("boundary", "browsing", "buying", "intent_override")
        samples = [
            _sample(index, scenario, f"P{index:09d}")
            for index, scenario in enumerate(scenarios * 12)
        ]

        first = select_smoke_indices(samples)
        second = select_smoke_indices(samples)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        counts = {
            scenario: sum(samples[index]["scenario_type"] == scenario for index in first)
            for scenario in scenarios
        }
        self.assertEqual(set(counts.values()), {10})

    def test_folds_keep_repeated_products_together(self) -> None:
        samples = [
            _sample(index, "buying", f"P{index // 2:09d}")
            for index in range(30)
        ]

        folds = assign_disjoint_folds(samples)

        self.assertEqual(len(folds), 5)
        memberships = {
            index: fold
            for fold, indices in enumerate(folds)
            for index in indices
        }
        for index in range(0, len(samples), 2):
            self.assertEqual(memberships[index], memberships[index + 1])
        self.assertLessEqual(max(map(len, folds)) - min(map(len, folds)), 2)

    def test_fold_report_contains_only_aggregate_metrics(self) -> None:
        scenarios = ("boundary", "browsing", "buying", "intent_override")
        samples = [
            _sample(index, scenarios[index % 4], f"P{index:09d}")
            for index in range(40)
        ]
        baseline = [_outcome(sample, rank=2, turn=2) for sample in samples]
        candidate = [_outcome(sample, rank=1, turn=2) for sample in samples]

        report = build_fold_report(samples, baseline, candidate)

        self.assertEqual(report["assignment"]["sample_counts"], [8] * 5)
        self.assertEqual(report["summary"]["mrr_positive_fold_count"], 5)
        self.assertGreater(report["summary"]["median_mrr_delta"], 0)

    def test_publication_rejects_raw_identifier_fields(self) -> None:
        with self.assertRaises(ValueError):
            validate_publication(
                {
                    "sessions": [],
                    "privacy": {
                        "aggregate_only": True,
                    },
                }
            )

    def test_publication_accepts_aggregate_privacy_contract(self) -> None:
        validate_publication(
            {
                "metric": 0.5,
                "privacy": {
                    "aggregate_only": True,
                    "labels_used_only_after_agent_replay": True,
                    "runtime_received_evaluation_labels": False,
                    "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
                },
            }
        )


if __name__ == "__main__":
    unittest.main()
