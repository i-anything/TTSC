from __future__ import annotations

import hashlib
import io
import json
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest import mock

from conversational_search.ranking import (
    STAGE_A_RANKING_POLICY,
    Bm25RescueRankingResult,
    Bm25RescueStatus,
    ProfileResidualStatus,
    RankingResult,
    RankingTrace,
)
from scripts import verify_phase10_phase9_exact_oracle as oracle
from tests.test_service_bm25_rescue import (
    _BROWSING_MESSAGE,
    _NO_FEATURE_PREFERENCE,
    _PROFILE,
    _agent,
    _cache_snapshot,
    _retriever,
)


def _without_policy(health: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in health.items() if key != "policy"}


def _normalized_slate(state: object) -> tuple[object, tuple[str, ...]]:
    signature = state.signature  # type: ignore[attr-defined]
    if signature is not None:
        values = list(signature)
        values[6] = "ranking-policy"
        signature = tuple(values)
    return signature, state.shown_ids  # type: ignore[attr-defined]


def _independent_canonical(
    case: oracle._ExactCase,
    phase9: object,
    phase10: Bm25RescueRankingResult,
) -> bytes:
    baseline = phase9
    ranking = baseline.ranking  # type: ignore[attr-defined]
    trace = ranking.trace
    payload = {
        "mode": case.mode,
        "phase9": {
            "profile": {
                "represented_theme_count": (
                    baseline.represented_theme_count  # type: ignore[attr-defined]
                ),
                "requested_theme_count": (
                    baseline.requested_theme_count  # type: ignore[attr-defined]
                ),
                "status": baseline.status.value,  # type: ignore[attr-defined]
            },
            "ranked_ids": list(ranking.ranked_ids),
            "trace": {
                "beta_hex": trace.beta.hex(),
                "input_ids": list(trace.input_ids),
                "observable_clause_count": trace.observable_clause_count,
                "output_ids": list(trace.output_ids),
            },
        },
        "rescue_status": phase10.status.value,
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


class Phase10Phase9ExactRankingOracleTests(unittest.TestCase):
    def test_frozen_oracle_verifies_all_exact_synthetic_cases(self) -> None:
        with (
            mock.patch.object(
                oracle,
                "rerank_stage_a_with_profile",
                wraps=oracle.rerank_stage_a_with_profile,
            ) as phase9,
            mock.patch.object(
                oracle,
                "rerank_stage_a_with_profile_and_bm25_rescue",
                wraps=oracle.rerank_stage_a_with_profile_and_bm25_rescue,
            ) as phase10,
        ):
            aggregate = oracle.verify_oracle()

        self.assertEqual(phase9.call_count, oracle.ORACLE_CASES)
        self.assertEqual(phase10.call_count, oracle.ORACLE_CASES)
        self.assertGreaterEqual(oracle.ORACLE_CASES, 1_200)
        self.assertEqual(
            aggregate,
            {
                "cases": oracle.ORACLE_CASES,
                "digest": oracle.EXPECTED_SHA256,
                "status": "ok",
            },
        )

    def test_generator_is_deterministic_bounded_and_spans_fallback_modes(self) -> None:
        cases = tuple(oracle._synthetic_cases())

        self.assertEqual(len(cases), oracle.ORACLE_CASES)
        self.assertEqual(cases, tuple(oracle._synthetic_cases()))
        self.assertEqual(
            Counter(case.mode for case in cases),
            Counter({mode: 200 for mode in oracle._MODES}),
        )
        self.assertEqual(
            Counter(case.injected_fault for case in cases),
            Counter(
                {
                    None: 800,
                    oracle._RESCUE_FAULT: 200,
                    oracle._PROFILE_SCORE_FAULT: 100,
                    oracle._INVALID_PROFILE: 100,
                }
            ),
        )
        for case in cases:
            self.assertLessEqual(len(case.documents), 12)
            self.assertLessEqual(len(case.state.requirements), 4)
            self.assertEqual(
                tuple(document.parent_asin for document in case.documents),
                case.fused_ids,
            )
            self.assertEqual(
                set(case.bm25_ids) | set(case.dense_ids),
                set(case.fused_ids),
            )
            self.assertTrue(
                all(
                    parent_asin.startswith(("P9O", "P10E"))
                    for parent_asin in case.fused_ids
                )
            )
            self.assertTrue(
                all("synthetic" in document.text for document in case.documents)
            )

    def test_every_mode_is_exact_and_has_its_declared_rescue_status(self) -> None:
        statuses: Counter[Bm25RescueStatus] = Counter()
        for case in oracle._synthetic_cases():
            phase9, phase10 = oracle._evaluate_case(case)
            statuses[phase10.status] += 1
            self.assertEqual(phase10.ranking, phase9.ranking)
            self.assertEqual(phase10.profile_status, phase9.status)
            self.assertEqual(
                phase10.requested_theme_count,
                phase9.requested_theme_count,
            )
            self.assertEqual(
                phase10.represented_theme_count,
                phase9.represented_theme_count,
            )
            self.assertIs(phase10.status, oracle._EXPECTED_STATUSES[case.mode])

        self.assertEqual(
            statuses,
            Counter(
                {
                    Bm25RescueStatus.ZERO_COMPLETENESS: 200,
                    Bm25RescueStatus.EMPTY_BM25: 200,
                    Bm25RescueStatus.NO_POSITIVE_UPLIFT: 200,
                    Bm25RescueStatus.UNCHANGED_ORDER: 200,
                    Bm25RescueStatus.SCORING_FALLBACK: 400,
                }
            ),
        )

    def test_independent_stream_serialization_matches_frozen_digest(self) -> None:
        digest = hashlib.sha256()
        digest.update(b"[")
        cases = 0
        for case in oracle._synthetic_cases():
            if cases:
                digest.update(b",")
            phase9, phase10 = oracle._evaluate_case(case)
            digest.update(_independent_canonical(case, phase9, phase10))
            cases += 1
        digest.update(b"]")

        self.assertEqual(cases, oracle.ORACLE_CASES)
        self.assertEqual(digest.hexdigest(), oracle.EXPECTED_SHA256)

    def test_wrong_and_malformed_expected_digests_fail_safely(self) -> None:
        with self.assertRaises(oracle.OracleDriftError) as raised:
            oracle.verify_oracle(expected_sha256="f" * 64)
        self.assertEqual(raised.exception.cases, oracle.ORACLE_CASES)
        self.assertEqual(raised.exception.actual, oracle.EXPECTED_SHA256)

        with mock.patch.object(oracle, "_compute_oracle_digest") as compute:
            with self.assertRaisesRegex(ValueError, "64 lowercase"):
                oracle.verify_oracle(expected_sha256="NOT-A-DIGEST")
        compute.assert_not_called()

    def test_mismatch_raises_without_exposing_synthetic_rows(self) -> None:
        case = next(oracle._synthetic_cases())
        malformed = Bm25RescueRankingResult(
            ranking=RankingResult(
                ranked_ids=("SYNTHETIC-MISMATCH",),
                trace=RankingTrace(
                    input_ids=("SYNTHETIC-MISMATCH",),
                    output_ids=("SYNTHETIC-MISMATCH",),
                    beta=0.2,
                    observable_clause_count=0,
                ),
            ),
            status=Bm25RescueStatus.ZERO_COMPLETENESS,
            profile_status=ProfileResidualStatus.NEUTRAL,
            requested_theme_count=0,
            represented_theme_count=0,
        )
        with mock.patch.object(oracle, "_rank_phase10", return_value=malformed):
            with self.assertRaises(oracle.OracleExactnessError) as raised:
                oracle._evaluate_case(case)

        self.assertEqual(raised.exception.cases, 1)
        self.assertNotIn("SYNTHETIC-MISMATCH", str(raised.exception))

    def test_canonical_output_is_complete_but_cli_is_aggregate_only(self) -> None:
        case = next(oracle._synthetic_cases())
        phase9, phase10 = oracle._evaluate_case(case)
        canonical = json.loads(oracle._canonical_output(case, phase9, phase10))
        self.assertEqual(
            set(canonical),
            {"mode", "phase9", "rescue_status"},
        )
        self.assertEqual(
            set(canonical["phase9"]),
            {"profile", "ranked_ids", "trace"},
        )

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = oracle.main()
        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "cases": oracle.ORACLE_CASES,
                "digest": oracle.EXPECTED_SHA256,
                "status": "ok",
            },
        )
        for forbidden in ("ranked_ids", "profile", "P9O", "P10E"):
            self.assertNotIn(forbidden, output.getvalue())


class Phase10Phase9ExactServiceOracleTests(unittest.TestCase):
    def test_neutral_service_path_is_exact_end_to_end(self) -> None:
        candidate_retriever = _retriever()
        baseline_retriever = _retriever()
        candidate = _agent(candidate_retriever)
        baseline = _agent(
            baseline_retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        for agent in (candidate, baseline):
            agent.reset("exact-session", _PROFILE)

        candidate_payloads = (
            candidate.respond("exact-session", _BROWSING_MESSAGE, 1, 2),
            candidate.respond("exact-session", _NO_FEATURE_PREFERENCE, 2, 2),
        )
        baseline_payloads = (
            baseline.respond("exact-session", _BROWSING_MESSAGE, 1, 2),
            baseline.respond("exact-session", _NO_FEATURE_PREFERENCE, 2, 2),
        )

        self.assertEqual(candidate_payloads, baseline_payloads)
        self.assertEqual(
            candidate.session_state("exact-session"),
            baseline.session_state("exact-session"),
        )
        self.assertEqual(
            _normalized_slate(candidate.slate_state("exact-session")),
            _normalized_slate(baseline.slate_state("exact-session")),
        )
        self.assertEqual(candidate_retriever.calls, baseline_retriever.calls)
        self.assertEqual(
            candidate_retriever.document_calls,
            baseline_retriever.document_calls,
        )
        self.assertEqual(
            _without_policy(candidate.ranking_health),
            _without_policy(baseline.ranking_health),
        )
        self.assertEqual(candidate.profile_health, baseline.profile_health)
        self.assertEqual(candidate.slate_health, baseline.slate_health)
        self.assertEqual(
            candidate.orchestration_health,
            baseline.orchestration_health,
        )
        candidate_cache = _cache_snapshot(candidate)
        baseline_cache = _cache_snapshot(baseline)
        self.assertEqual(candidate_cache[0][2:], baseline_cache[0][2:])
        self.assertNotEqual(candidate_cache[0][1], baseline_cache[0][1])

        rescue = candidate.rescue_health
        self.assertEqual(rescue["attempts"], 1)
        self.assertEqual(rescue["zero_completeness_neutral"], 1)
        self.assertEqual(
            sum(
                value
                for key, value in rescue.items()
                if key not in {"policy", "attempts"}
            ),
            rescue["attempts"],
        )

    def test_service_fault_is_exact_for_the_turn_and_cannot_enter_cache(self) -> None:
        candidate_retriever = _retriever()
        baseline_retriever = _retriever()
        candidate = _agent(candidate_retriever)
        baseline = _agent(
            baseline_retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        candidate.reset("fault-session", _PROFILE)
        baseline.reset("fault-session", _PROFILE)
        baseline_payload = baseline.respond(
            "fault-session",
            _BROWSING_MESSAGE,
            1,
            2,
        )

        with mock.patch(
            "conversational_search.service."
            "rerank_stage_a_with_profile_and_bm25_rescue",
            side_effect=RuntimeError("synthetic composite fault"),
        ):
            candidate_payload = candidate.respond(
                "fault-session",
                _BROWSING_MESSAGE,
                1,
                2,
            )

        self.assertEqual(candidate_payload, baseline_payload)
        self.assertEqual(
            candidate.session_state("fault-session"),
            baseline.session_state("fault-session"),
        )
        self.assertEqual(
            _normalized_slate(candidate.slate_state("fault-session")),
            _normalized_slate(baseline.slate_state("fault-session")),
        )
        self.assertEqual(candidate_retriever.calls, baseline_retriever.calls)
        self.assertEqual(
            candidate_retriever.document_calls,
            baseline_retriever.document_calls,
        )
        self.assertEqual(candidate.profile_health, baseline.profile_health)
        self.assertEqual(candidate.orchestration_health["stores"], 0)
        self.assertEqual(len(_cache_snapshot(candidate)), 0)
        self.assertEqual(
            candidate.rescue_health["validation_or_scoring_fallbacks"],
            1,
        )
        self.assertEqual(candidate.ranking_health["attempts"], 1)
        self.assertEqual(candidate.ranking_health["successes"], 1)


if __name__ == "__main__":
    unittest.main()
