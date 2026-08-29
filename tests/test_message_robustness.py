from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from conversational_search.intent import (
    CANONICAL_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
    IntentState,
    apply_user_message,
    record_question,
)
from scripts.run_message_robustness import (
    EXPECTED_COORDINATE_PLAN_SUITE_SHA256,
    EXPECTED_DECISION_PLAN_SUITE_SHA256,
    EXPECTED_PERTURBATION_BANK_SHA256,
    EXPECTED_PERTURBATION_SPEC_SHA256,
    REPLICATES,
    SeededMessageAgent,
    _paraphrase_variants,
    _surface_variants,
    coordinate_plan_sha256,
    coordinate_plan_suite_sha256,
    decision_plan_suite_sha256,
    parse_canonical_message,
    perturb_message,
    perturbation_bank_sha256,
    perturbation_spec_sha256,
)
from scripts import run_message_robustness as robustness


CANONICAL_MESSAGES = (
    "I'm looking for Shoes. A key requirement is: leather.",
    "I'm looking for Shoes, but I'm still exploring.",
    "I'm looking for Shoes. leather",
    "For that, what matters is: leather.",
    "Actually, ignore my earlier preference. What I need is: cotton.",
    "I don't have a preference for material; please use your judgment.",
    "I don't have an additional preference for material.",
    "Those options are not quite right yet. Ask me about one specific attribute.",
)


def _starting_state(kind: str) -> IntentState:
    if kind in {"buying", "browsing", "tentative"}:
        return IntentState()
    initial = apply_user_message(
        IntentState(),
        "I'm looking for Shoes. A key requirement is: leather.",
        1,
        policy=CANONICAL_INTENT_POLICY,
    )
    if kind in {
        "answer",
        "boundary_no_preference",
        "no_additional_preference",
        "request_question",
    }:
        return record_question(initial, "material")
    return initial


class MessagePerturbationTest(unittest.TestCase):
    def test_bank_is_slot_preserving_and_robustly_state_equivalent(self) -> None:
        for canonical in CANONICAL_MESSAGES:
            parsed = parse_canonical_message(canonical)
            self.assertIsNotNone(parsed)
            assert parsed is not None
            state = _starting_state(parsed.kind)
            turn = 1 if state.last_turn == 0 else 2
            expected = apply_user_message(
                state,
                canonical,
                turn,
                policy=CANONICAL_INTENT_POLICY,
            )
            for variant in (*_surface_variants(parsed), *_paraphrase_variants(parsed)):
                with self.subTest(kind=parsed.kind, variant=variant):
                    for _, slot in parsed.slots:
                        self.assertIn(slot, variant)
                    observed = apply_user_message(
                        state,
                        variant,
                        turn,
                        policy=ROBUST_INTENT_POLICY,
                    )
                    self.assertEqual(observed, expected)

    def test_coordinate_sampler_is_reproducible_and_call_order_independent(self) -> None:
        canonical = CANONICAL_MESSAGES[0]
        first = perturb_message(canonical, replicate=1, ordinal=17, turn=4)
        for replicate in REPLICATES:
            perturb_message(canonical, replicate=replicate, ordinal=1, turn=1)
        second = perturb_message(canonical, replicate=1, ordinal=17, turn=4)

        self.assertEqual(first, second)
        self.assertEqual(
            coordinate_plan_sha256(1, 200),
            coordinate_plan_sha256(1, 200),
        )
        self.assertNotEqual(
            coordinate_plan_sha256(1, 200),
            coordinate_plan_sha256(2, 200),
        )
        self.assertEqual(
            coordinate_plan_suite_sha256(200),
            EXPECTED_COORDINATE_PLAN_SUITE_SHA256,
        )
        self.assertEqual(
            coordinate_plan_suite_sha256(200),
            "7e8987bfd92c113849d89888238b87a6495d7f1c3864e9dc7511953a0340289b",
        )
        self.assertEqual(
            perturbation_bank_sha256(),
            EXPECTED_PERTURBATION_BANK_SHA256,
        )
        self.assertEqual(
            perturbation_bank_sha256(),
            "be52c101df1194b517c7be136da1d67b616fa878d5e3c221d6026a6774db7d4a",
        )
        self.assertEqual(
            decision_plan_suite_sha256(200),
            EXPECTED_DECISION_PLAN_SUITE_SHA256,
        )
        self.assertEqual(
            decision_plan_suite_sha256(200),
            "f0879f8fc57a170a158cdbcf0f708e5b4cc215ac7d9d4603a2a1f30c0c7e223f",
        )
        self.assertEqual(
            perturbation_spec_sha256(200),
            EXPECTED_PERTURBATION_SPEC_SHA256,
        )
        self.assertEqual(
            perturbation_spec_sha256(200),
            "a965e2da01c15bc4cfd768146169b12073bb13f51a27c56cb3464223f720175c",
        )
        root = Path(robustness.__file__).resolve().parents[1]
        self.assertEqual(
            robustness._validate_protocol_goldens(root, 200),
            {
                "perturbation_bank_sha256": EXPECTED_PERTURBATION_BANK_SHA256,
                "coordinate_plan_suite_sha256": (
                    EXPECTED_COORDINATE_PLAN_SUITE_SHA256
                ),
                "decision_plan_suite_sha256": (
                    EXPECTED_DECISION_PLAN_SUITE_SHA256
                ),
                "perturbation_spec_sha256": EXPECTED_PERTURBATION_SPEC_SHA256,
            },
        )
        with mock.patch.object(robustness, "UNCHANGED_CUTOFF", 0.16):
            self.assertNotEqual(
                decision_plan_suite_sha256(200),
                EXPECTED_DECISION_PLAN_SUITE_SHA256,
            )
            self.assertNotEqual(
                perturbation_spec_sha256(200),
                EXPECTED_PERTURBATION_SPEC_SHA256,
            )

    def test_unknown_messages_pass_through(self) -> None:
        message = "A free-form message outside the simulator grammar."
        result = perturb_message(message, replicate=1, ordinal=1, turn=1)

        self.assertEqual(result.message, message)
        self.assertEqual(result.kind, "unknown")
        self.assertEqual(result.mode, "unchanged")

    def test_payload_terminal_period_is_not_stripped_twice(self) -> None:
        state = record_question(
            apply_user_message(
                IntentState(),
                "I'm looking for Shoes, but I'm still exploring.",
                1,
            ),
            "feature",
        )
        canonical = "For that, what matters is: Packs flat for travel.."
        natural = "What matters to me there is: Packs flat for travel.."

        expected = apply_user_message(
            state,
            canonical,
            2,
            policy=CANONICAL_INTENT_POLICY,
        )
        observed = apply_user_message(
            state,
            natural,
            2,
            policy=ROBUST_INTENT_POLICY,
        )

        self.assertEqual(observed, expected)
        self.assertEqual(observed.requirements[-1].value, "Packs flat for travel.")

    def test_seeded_wrapper_reports_equivalence_without_retaining_text(self) -> None:
        class Delegate:
            def reset(self, session_id: str, user_profile: dict) -> None:
                pass

            def respond(
                self,
                session_id: str,
                user_message: str,
                turn: int,
                top_k: int,
            ) -> dict:
                return {
                    "message": "ok",
                    "ask_attribute": "material",
                    "recommendations": [],
                }

        wrapper = SeededMessageAgent(  # type: ignore[arg-type]
            Delegate(),
            replicate=1,
            policy=ROBUST_INTENT_POLICY,
        )
        wrapper.reset("session", {})
        wrapper.respond("session", CANONICAL_MESSAGES[1], 1, 10)
        summary = wrapper.summary()

        self.assertEqual(summary["state_matches"], 1)
        self.assertEqual(summary["query_matches"], 1)
        serialized = repr(summary)
        self.assertNotIn("Shoes", serialized)
        self.assertNotIn("session", serialized)

    def test_health_validation_is_fail_closed(self) -> None:
        route = {"bm25": {"ok": 1}, "dense": {"ok": 1}, "fallback_turns": 0}
        ranking = {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
            "unavailable_skips": 0,
        }
        slate = {"attempts": 1, "successes": 1, "failures": 0}
        audit = {
            "response_exceptions": 0,
            "invalid_responses": 0,
            "input_processing_exceptions": 0,
            "slot_corruptions": 0,
            "known_messages": 1,
            "unknown_messages": 0,
            "state_checks": 1,
            "integration_checks": 1,
            "integration_matches": 1,
        }

        robustness._validate_run_health(1, route, ranking, slate, audit)
        with self.assertRaisesRegex(RuntimeError, "reranker health"):
            robustness._validate_run_health(
                1,
                route,
                {**ranking, "unavailable_skips": 1},
                slate,
                audit,
            )
        with self.assertRaisesRegex(RuntimeError, "message adapter health"):
            robustness._validate_run_health(
                1,
                route,
                ranking,
                slate,
                {**audit, "slot_corruptions": 1},
            )

    def test_frozen_inputs_scenarios_privacy_and_publication_paths_fail_closed(self) -> None:
        drifted_control = {
            **robustness.PHASE5_METRICS,
            "scenario_metrics": {},
        }
        with self.assertRaisesRegex(RuntimeError, "control metrics drifted"):
            robustness._validate_phase5_control(drifted_control)

        with mock.patch.object(robustness, "_sha256", return_value="bad"):
            with self.assertRaisesRegex(RuntimeError, "frozen Phase 6 inputs drifted"):
                robustness._validate_frozen_inputs(
                    Path("/repository"),
                    Path("/catalog"),
                    Path("/dataset"),
                )

        root = Path(robustness.__file__).resolve().parents[1]
        for output in (
            root / "docs" / "phase6_results.json",
            root / "benchmarks" / "diagnostics" / "phase6.json",
        ):
            with self.assertRaisesRegex(ValueError, "append-only"):
                robustness._validate_output(
                    output,
                    root / "data" / "catalog.jsonl",
                    root / "data" / "public_set.jsonl",
                )

        with self.assertRaisesRegex(RuntimeError, "top-level schema"):
            robustness._validate_aggregate_privacy({})

    def test_end_to_end_orchestration_order_backend_reuse_and_privacy(self) -> None:
        def control_result() -> dict:
            return {
                **robustness.PHASE5_METRICS,
                "reported_token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "scenario_metrics": robustness.PHASE5_SCENARIO_METRICS,
                "sessions": [],
            }

        def perturbed_result(candidate: bool) -> dict:
            hit = candidate
            return {
                "sample_count": 1,
                "hit_rate_at_10": 0.99 if candidate else 0.8,
                "mrr": 0.52 if candidate else 0.4,
                "mttc": 3.0 if candidate else 4.0,
                "efficiency": 0.8 if candidate else 0.7,
                "recommended_technical_score": 0.81 if candidate else 0.7,
                "reported_token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "scenario_metrics": {
                    "buying": {
                        "sample_count": 1,
                        "hit_rate_at_10": float(hit),
                        "mrr": 0.52 if hit else 0.0,
                        "mttc": 3.0 if hit else 11.0,
                    }
                },
                "sessions": [
                    {
                        "sample_id": "SECRET_SAMPLE",
                        "scenario_type": "buying",
                        "hit": hit,
                        "first_hit_turn": 3 if hit else None,
                        "best_rank": 2 if hit else None,
                    }
                ],
            }

        def diagnostic(candidate: bool, replicate: int | None) -> dict:
            matched = int(candidate or replicate is None)
            return {
                "route_health": {
                    "bm25": {"ok": 1},
                    "dense": {"ok": 1},
                    "fallback_turns": 0,
                },
                "ranking_health": {
                    "policy": "stage_a",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "unavailable_skips": 0,
                },
                "slate_health": {
                    "policy": "stagnation_aware",
                    "attempts": 1,
                    "successes": 1,
                    "failures": 0,
                    "initializations": 1,
                    "ranking_resets": 0,
                    "stagnant_turns": 0,
                    "unseen_selected_on_stagnant": 0,
                    "repeat_backfills": 0,
                },
                "message_audit": {
                    "known_messages": 1,
                    "unknown_messages": 0,
                    "transformed_messages": int(replicate is not None),
                    "family_counts": {"buying": 1},
                    "mode_counts": {"paraphrase": 1},
                    "state_checks": 1,
                    "state_matches": matched,
                    "query_matches": matched,
                    "critical_checks": 1,
                    "critical_matches": matched,
                    "integration_checks": 1,
                    "integration_matches": 1,
                    "response_exceptions": 0,
                    "invalid_responses": 0,
                    "input_processing_exceptions": 0,
                    "slot_corruptions": 0,
                    "state_mismatch_families": {},
                    "query_mismatch_families": {},
                    "state_mismatch_components": {},
                    "choice_trace_sha256": f"trace-{candidate}-{replicate}",
                },
                "evaluation_wall_seconds": 0.1,
                "respond_latency_ms": {"warm_p95": 10.0},
            }

        backend = SimpleNamespace(dense_available=True, bm25_available=True)
        runtime = SimpleNamespace(retrieval_backend=backend)

        def fake_run(*args: object, **kwargs: object) -> tuple[dict, dict]:
            policy = kwargs["policy"]
            replicate = kwargs["replicate"]
            candidate = policy is ROBUST_INTENT_POLICY
            if replicate is None:
                return control_result(), diagnostic(candidate, replicate)
            return perturbed_result(candidate), diagnostic(candidate, replicate)

        with (
            mock.patch.object(robustness, "load_jsonl", return_value=[{}]),
            mock.patch.object(
                robustness,
                "catalog_index",
                return_value=({"A"}, {"A": ["Shoes"]}, {"A": {}}),
            ),
            mock.patch.object(
                robustness,
                "ConversationalSearchAgent",
                return_value=runtime,
            ),
            mock.patch.object(robustness, "_run_variant", side_effect=fake_run) as run,
            mock.patch.object(
                robustness,
                "_validate_frozen_inputs",
                return_value=dict(robustness.EXPECTED_INPUT_SHA256),
            ),
            mock.patch.object(
                robustness,
                "_validate_protocol_goldens",
                return_value={
                    "perturbation_bank_sha256": "0" * 64,
                    "coordinate_plan_suite_sha256": "1" * 64,
                    "decision_plan_suite_sha256": "2" * 64,
                    "perturbation_spec_sha256": "3" * 64,
                },
            ),
            mock.patch.object(robustness, "_sha256", return_value="0" * 64),
        ):
            payload = robustness.run_message_robustness(
                "catalog.jsonl",
                "dataset.jsonl",
            )

        order = [
            (call.kwargs["policy"], call.kwargs["replicate"])
            for call in run.call_args_list
        ]
        expected = [
            (CANONICAL_INTENT_POLICY, None),
            (ROBUST_INTENT_POLICY, None),
            *[
                item
                for replicate in REPLICATES
                for item in (
                    (CANONICAL_INTENT_POLICY, replicate),
                    (ROBUST_INTENT_POLICY, replicate),
                )
            ],
            (ROBUST_INTENT_POLICY, REPLICATES[0]),
        ]
        self.assertEqual(order, expected)
        self.assertTrue(all(call.args[5] is backend for call in run.call_args_list))
        serialized = json.dumps(payload)
        self.assertNotIn("SECRET_SAMPLE", serialized)
        self.assertNotIn('"sessions"', serialized)
        self.assertTrue(payload["decision_gate"]["adopt"])

        dictionary_row_leak = json.loads(serialized)
        dictionary_row_leak["control"]["observations"] = {
            "SECRET_SAMPLE": {
                "utterance": "private user words",
                "selection": "private target",
            }
        }
        with self.assertRaisesRegex(RuntimeError, "schema drifted"):
            robustness._validate_aggregate_privacy(dictionary_row_leak)


if __name__ == "__main__":
    unittest.main()
