from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from conversational_search.ranking import (
    ProfileRankingResult,
    ProfileResidualStatus,
    RankingResult,
    RankingTrace,
)
from scripts import verify_phase9_ranking_oracle as oracle


class Phase9RankingOracleTest(unittest.TestCase):
    def test_frozen_oracle_verifies_all_synthetic_cases(self) -> None:
        with patch.object(
            oracle,
            "rerank_stage_a_with_profile",
            wraps=oracle.rerank_stage_a_with_profile,
        ) as rank:
            aggregate = oracle.verify_oracle()

        self.assertEqual(rank.call_count, oracle.ORACLE_CASES)
        self.assertGreaterEqual(rank.call_count, 1_000)
        self.assertEqual(
            aggregate,
            {
                "cases": oracle.ORACLE_CASES,
                "digest": oracle.EXPECTED_SHA256,
                "status": "ok",
            },
        )

    def test_generator_is_fixed_seed_bounded_synthetic_and_spans_contract(self) -> None:
        cases = tuple(oracle._synthetic_cases())

        self.assertEqual(len(cases), oracle.ORACLE_CASES)
        self.assertEqual(
            {case.intent_kind for case in cases},
            {"empty", "category", "weak", "strong", "mixed"},
        )
        self.assertEqual(
            {case.route_kind for case in cases},
            {"bm25_only", "dense_only", "overlap", "partial"},
        )
        self.assertEqual(
            {case.profile_kind for case in cases},
            {
                "disabled",
                "neutral",
                "comfort",
                "durability",
                "comfort_durability",
                "weather_breathability_sustainability",
            },
        )
        self.assertEqual(
            {case.route_weights for case in cases},
            set(oracle._ROUTE_WEIGHTS),
        )
        for case in cases:
            self.assertLessEqual(
                len(case.documents),
                oracle.MAX_CANDIDATES_PER_CASE,
            )
            self.assertLessEqual(
                len(case.state.requirements),
                oracle.MAX_REQUIREMENTS_PER_CASE,
            )
            self.assertEqual(
                tuple(document.parent_asin for document in case.documents),
                case.fused_ids,
            )
            self.assertEqual(
                set(case.bm25_ids) | set(case.dense_ids),
                set(case.fused_ids),
            )
            self.assertTrue(
                all(value.startswith("P9O") for value in case.fused_ids)
            )
            self.assertTrue(
                all(
                    "fully synthetic phase nine item" in document.text
                    for document in case.documents
                )
            )

        self.assertEqual(cases, tuple(oracle._synthetic_cases()))

    def test_wrong_digest_raises_with_aggregate_drift_evidence(self) -> None:
        with self.assertRaises(oracle.OracleDriftError) as raised:
            oracle.verify_oracle(expected_sha256="f" * 64)

        self.assertEqual(raised.exception.cases, oracle.ORACLE_CASES)
        self.assertEqual(raised.exception.actual, oracle.EXPECTED_SHA256)
        self.assertEqual(raised.exception.expected, "f" * 64)

    def test_malformed_expected_digest_is_rejected_before_generation(self) -> None:
        with patch.object(oracle, "_compute_oracle_digest") as compute:
            with self.assertRaisesRegex(ValueError, "64 lowercase"):
                oracle.verify_oracle(expected_sha256="NOT-A-DIGEST")

        compute.assert_not_called()

    def test_canonical_output_contains_complete_frozen_output_contract(self) -> None:
        result = ProfileRankingResult(
            ranking=RankingResult(
                ranked_ids=("P9O0001C01", "P9O0001C00"),
                trace=RankingTrace(
                    input_ids=("P9O0001C00", "P9O0001C01"),
                    output_ids=("P9O0001C01", "P9O0001C00"),
                    beta=0.45,
                    observable_clause_count=2,
                ),
            ),
            status=ProfileResidualStatus.APPLIED,
            requested_theme_count=2,
            represented_theme_count=1,
        )

        canonical = oracle._canonical_output(result)
        self.assertEqual(
            json.loads(canonical),
            {
                "profile": {
                    "represented_theme_count": 1,
                    "requested_theme_count": 2,
                    "status": "applied",
                },
                "ranked_ids": ["P9O0001C01", "P9O0001C00"],
                "trace": {
                    "beta_hex": (0.45).hex(),
                    "input_ids": ["P9O0001C00", "P9O0001C01"],
                    "observable_clause_count": 2,
                    "output_ids": ["P9O0001C01", "P9O0001C00"],
                },
            },
        )
        self.assertNotIn(b'"beta":', canonical)

    def test_cli_prints_one_compact_json_aggregate_line(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = oracle.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertEqual(
            output.getvalue(),
            json.dumps(
                {
                    "cases": oracle.ORACLE_CASES,
                    "digest": oracle.EXPECTED_SHA256,
                    "status": "ok",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )


if __name__ == "__main__":
    unittest.main()
