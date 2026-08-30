from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from conversational_search.ranking import RankingResult, RankingTrace
from scripts import verify_phase7_stage_a_oracle as oracle


class Phase7StageAOracleTest(unittest.TestCase):
    def test_frozen_oracle_verifies_exactly_one_thousand_cases(self) -> None:
        with patch.object(
            oracle,
            "rerank_stage_a",
            wraps=oracle.rerank_stage_a,
        ) as rank:
            aggregate = oracle.verify_oracle()

        self.assertEqual(rank.call_count, 1_000)
        self.assertEqual(
            aggregate,
            {
                "cases": 1_000,
                "digest": oracle.EXPECTED_SHA256,
                "status": "ok",
            },
        )

    def test_generator_is_fixed_seed_bounded_and_fully_synthetic(self) -> None:
        cases = tuple(oracle._synthetic_cases())

        self.assertEqual(len(cases), oracle.ORACLE_CASES)
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
                all(parent_asin.startswith("S") for parent_asin in case.fused_ids)
            )
            self.assertTrue(
                all("synthetic item" in document.text for document in case.documents)
            )

        first = tuple(oracle._synthetic_cases())
        self.assertEqual(cases, first)

    def test_wrong_digest_raises_with_aggregate_drift_evidence(self) -> None:
        with self.assertRaises(oracle.OracleDriftError) as raised:
            oracle.verify_oracle(expected_sha256="0" * 64)

        self.assertEqual(raised.exception.cases, 1_000)
        self.assertEqual(raised.exception.actual, oracle.EXPECTED_SHA256)
        self.assertEqual(raised.exception.expected, "0" * 64)

    def test_malformed_expected_digest_is_rejected_before_generation(self) -> None:
        with patch.object(oracle, "_compute_oracle_digest") as compute:
            with self.assertRaisesRegex(ValueError, "64 lowercase"):
                oracle.verify_oracle(expected_sha256="NOT-A-DIGEST")

        compute.assert_not_called()

    def test_canonical_output_contains_only_the_frozen_output_contract(self) -> None:
        result = RankingResult(
            ranked_ids=("S0001P01", "S0001P00"),
            trace=RankingTrace(
                input_ids=("S0001P00", "S0001P01"),
                output_ids=("S0001P01", "S0001P00"),
                beta=0.45,
                observable_clause_count=2,
            ),
        )

        canonical = oracle._canonical_output(result)
        self.assertEqual(
            json.loads(canonical),
            {
                "ranked_ids": ["S0001P01", "S0001P00"],
                "trace": {
                    "beta_hex": (0.45).hex(),
                    "input_ids": ["S0001P00", "S0001P01"],
                    "observable_clause_count": 2,
                    "output_ids": ["S0001P01", "S0001P00"],
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
                    "cases": 1_000,
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
