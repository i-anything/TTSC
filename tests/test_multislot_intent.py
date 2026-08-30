from __future__ import annotations

import io
import json
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest import mock

from conversational_search.intent import (
    LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
    IntentReduction,
    IntentReductionStatus,
    IntentState,
    Requirement,
    apply_user_message,
    apply_user_message_with_trace,
    record_question,
    render_dense_query,
    render_lexical_query,
)
from scripts import verify_phase11_intent_oracle as phase11_oracle


def _reduce(state: IntentState, message: str, turn: int = 1):
    return apply_user_message_with_trace(
        state,
        message,
        turn,
        policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    )


def _baseline(state: IntentState, message: str, turn: int = 1) -> IntentState:
    return apply_user_message(
        state,
        message,
        turn,
        policy=ROBUST_INTENT_POLICY,
    )


class LosslessMultiSlotIntentTest(unittest.TestCase):
    def test_two_unambiguous_positive_slots_are_atomic(self) -> None:
        reduction = _reduce(IntentState(), "I want red and leather.")

        self.assertEqual(reduction.status, IntentReductionStatus.APPLIED)
        self.assertEqual(
            [(item.value, item.attribute, item.source) for item in reduction.state.requirements],
            [
                ("red", "color", "free_text"),
                ("leather", "material", "free_text"),
            ],
        )
        self.assertEqual(reduction.positive_atoms, 2)
        self.assertEqual(reduction.residual_atoms, 0)

    def test_initial_buying_and_tentative_envelopes_preserve_provenance(self) -> None:
        buying = _reduce(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: red and leather.",
        )
        tentative = _reduce(
            IntentState(),
            "Looking for Shoes, maybe red and leather.",
        )

        self.assertEqual(buying.state.category, "Shoes")
        self.assertEqual(tentative.state.category, "Shoes")
        self.assertEqual(
            [item.source for item in buying.state.requirements],
            ["initial_explicit", "initial_explicit"],
        )
        self.assertEqual(
            [item.source for item in tentative.state.requirements],
            ["initial_tentative", "initial_tentative"],
        )

    def test_compound_answer_preserves_answer_provenance(self) -> None:
        state = record_question(
            apply_user_message(
                IntentState(),
                "I'm looking for Shoes, but I'm still exploring.",
                1,
            ),
            "material",
        )
        reduction = _reduce(
            state,
            "For that, what matters is: leather and red.",
            2,
        )

        self.assertEqual(
            [(item.value, item.attribute, item.source) for item in reduction.state.requirements],
            [
                ("leather", "material", "answer"),
                ("red", "color", "answer"),
            ],
        )

    def test_typed_and_residual_constraints_are_both_preserved(self) -> None:
        reduction = _reduce(
            IntentState(),
            "I need it under $100 and waterproof.",
        )

        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("under $100", "budget"), ("waterproof", None)],
        )
        self.assertEqual(reduction.residual_atoms, 1)
        self.assertIn("under $100", render_lexical_query(reduction.state))
        self.assertIn("waterproof", render_lexical_query(reduction.state))

    def test_explicit_label_protects_conjunction_inside_brand(self) -> None:
        reduction = _reduce(
            IntentState(),
            "Brand: Calvin and Klein; material: leather.",
        )

        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("Calvin and Klein", "brand"), ("leather", "material")],
        )

    def test_explicit_feature_protects_same_attribute_phrase(self) -> None:
        reduction = _reduce(
            IntentState(),
            "feature: black and white pattern; material: leather.",
        )

        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("black and white pattern", "feature"), ("leather", "material")],
        )

    def test_same_attribute_alternative_is_exact_baseline(self) -> None:
        state = IntentState()
        message = "black and white"
        reduction = _reduce(state, message)

        self.assertEqual(reduction.status, IntentReductionStatus.SINGLE_SLOT)
        self.assertEqual(reduction.state, _baseline(state, message))

    def test_ambiguous_unsplit_span_is_exact_baseline(self) -> None:
        state = IntentState()
        message = "red leather"
        reduction = _reduce(state, message)

        self.assertEqual(reduction.status, IntentReductionStatus.AMBIGUOUS)
        self.assertEqual(reduction.state, _baseline(state, message))

    def test_exclusion_is_not_rendered_as_positive_query_evidence(self) -> None:
        reduction = _reduce(IntentState(), "not leather, breathable")

        self.assertEqual(reduction.state.excluded, ("leather",))
        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("breathable", None)],
        )
        self.assertNotIn("leather", render_dense_query(reduction.state).casefold())
        self.assertNotIn("leather", render_lexical_query(reduction.state).casefold())
        self.assertEqual(reduction.exclusion_atoms, 1)
        self.assertEqual(reduction.state.intent_version, 1)

    def test_source_order_resolves_positive_exclusion_conflicts(self) -> None:
        negative_last = _reduce(IntentState(), "leather and not leather")
        positive_last = _reduce(IntentState(), "not leather and leather")

        self.assertEqual(negative_last.state.requirements, ())
        self.assertEqual(negative_last.state.excluded, ("leather",))
        self.assertEqual(
            [(item.value, item.attribute) for item in positive_last.state.requirements],
            [("leather", "material")],
        )
        self.assertEqual(positive_last.state.excluded, ())

    def test_explicit_label_and_bare_value_share_one_conflict_identity(self) -> None:
        labeled = IntentState(
            requirements=(
                Requirement("color: red", "answer", 1, "color"),
            ),
            excluded=("material: wool",),
            last_turn=1,
        )

        excluded = _reduce(labeled, "not red; material: leather", 2)
        restored = _reduce(labeled, "color: blue; wool", 2)

        self.assertEqual(
            [(item.value, item.attribute) for item in excluded.state.requirements],
            [("leather", "material")],
        )
        self.assertEqual(excluded.state.excluded, ("material: wool", "red"))
        self.assertEqual(restored.state.excluded, ())
        self.assertEqual(
            [(item.value, item.attribute) for item in restored.state.requirements],
            [("color: red", "color"), ("blue", "color"), ("wool", "material")],
        )

    def test_no_preference_and_positive_constraint_compose(self) -> None:
        initial = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: red.",
            1,
        )
        reduction = _reduce(initial, "Any color is fine; leather", 2)

        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("leather", "material")],
        )
        self.assertEqual(reduction.state.no_preference, frozenset({"color"}))
        self.assertEqual(reduction.clear_atoms, 1)
        self.assertEqual(reduction.state.intent_version, 1)

    def test_clear_removes_typed_exclusion_for_same_attribute(self) -> None:
        excluded = _reduce(IntentState(), "not red; leather").state
        cleared = _reduce(excluded, "Any color is fine; wool", 2).state

        self.assertEqual(cleared.excluded, ())
        self.assertIn("color", cleared.no_preference)
        self.assertEqual(
            [(item.value, item.attribute) for item in cleared.requirements],
            [("leather", "material"), ("wool", "material")],
        )

    def test_same_slot_replacement_is_destructive_once(self) -> None:
        initial = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: red.",
            1,
        )
        reduction = _reduce(initial, "no longer red; now blue", 2)

        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("blue", "color")],
        )
        self.assertEqual(reduction.state.excluded, ("red",))
        self.assertEqual(reduction.replacement_atoms, 1)
        self.assertEqual(reduction.state.intent_version, 1)

    def test_replace_old_with_new_is_one_atomic_correction(self) -> None:
        reduction = _reduce(IntentState(), "replace red with blue")

        self.assertEqual(reduction.status, IntentReductionStatus.APPLIED)
        self.assertEqual(reduction.state.excluded, ("red",))
        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("blue", "color")],
        )

    def test_full_override_preserves_answer_and_replaces_initial_sources(self) -> None:
        initial = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: red.",
            1,
        )
        answered = apply_user_message(
            record_question(initial, "brand"),
            "For that, what matters is: Alpine Works.",
            2,
        )
        reduction = _reduce(
            answered,
            "Actually, ignore my earlier preference. What I need is: leather and blue.",
            3,
        )

        self.assertEqual(
            [(item.value, item.attribute, item.source) for item in reduction.state.requirements],
            [
                ("Alpine Works", "brand", "answer"),
                ("leather", "material", "override"),
                ("blue", "color", "override"),
            ],
        )
        self.assertEqual(reduction.state.intent_version, 1)

    def test_pronoun_scaffold_is_consumed_without_losing_values(self) -> None:
        reduction = _reduce(IntentState(), "make it red and leather")

        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("red", "color"), ("leather", "material")],
        )

    def test_unsafe_negation_scope_is_exact_baseline(self) -> None:
        for message in (
            "not only leather but red",
            "not sure about leather and red",
            "no less than size 10 and leather",
            "without a doubt leather and red",
        ):
            with self.subTest(message=message):
                reduction = _reduce(IntentState(), message)
                self.assertNotEqual(reduction.status, IntentReductionStatus.APPLIED)
                self.assertEqual(reduction.state, _baseline(IntentState(), message))

    def test_numeric_comma_is_not_a_constraint_boundary(self) -> None:
        reduction = _reduce(IntentState(), "under $1,000 and red")

        self.assertEqual(
            [(item.value, item.attribute) for item in reduction.state.requirements],
            [("under $1,000", "budget"), ("red", "color")],
        )

    def test_candidate_message_bound_falls_back_exactly(self) -> None:
        state = IntentState()
        message = "x" * 2049
        reduction = _reduce(state, message)

        self.assertEqual(reduction.status, IntentReductionStatus.BOUNDS)
        self.assertEqual(reduction.state, _baseline(state, message))

    def test_atom_count_bound_falls_back_exactly(self) -> None:
        state = IntentState()
        message = "; ".join(f"feature: value{index}" for index in range(9))
        reduction = _reduce(state, message)

        self.assertEqual(reduction.status, IntentReductionStatus.BOUNDS)
        self.assertEqual(reduction.state, _baseline(state, message))

    def test_atom_value_bound_falls_back_exactly(self) -> None:
        state = IntentState()
        message = f"feature: {'x' * 257}; material: leather"
        reduction = _reduce(state, message)

        self.assertEqual(reduction.status, IntentReductionStatus.BOUNDS)
        self.assertEqual(reduction.state, _baseline(state, message))

    def test_state_requirement_bound_falls_back_exactly(self) -> None:
        state = IntentState(
            requirements=tuple(
                Requirement(f"value-{index}", "free_text", 1)
                for index in range(24)
            ),
            last_turn=1,
        )
        message = "red and leather"
        reduction = _reduce(state, message, 2)

        self.assertEqual(reduction.status, IntentReductionStatus.BOUNDS)
        self.assertEqual(reduction.state, _baseline(state, message, 2))

    def test_state_exclusion_bound_falls_back_exactly(self) -> None:
        state = IntentState(
            excluded=tuple(f"excluded-{index}" for index in range(16)),
            last_turn=1,
        )
        message = "not leather and red"
        reduction = _reduce(state, message, 2)

        self.assertEqual(reduction.status, IntentReductionStatus.BOUNDS)
        self.assertEqual(reduction.state, _baseline(state, message, 2))

    def test_reduction_is_immutable_and_deterministic(self) -> None:
        state = IntentState()
        first = _reduce(state, "red and leather")
        second = _reduce(state, "red and leather")

        self.assertEqual(first, second)
        self.assertEqual(state, IntentState())
        self.assertIsNot(first.state, state)

    def test_non_candidate_policy_trace_is_exact_baseline(self) -> None:
        state = IntentState()
        message = "red and leather"
        reduction = apply_user_message_with_trace(
            state,
            message,
            1,
            policy=ROBUST_INTENT_POLICY,
        )

        self.assertEqual(reduction.status, IntentReductionStatus.BASELINE_POLICY)
        self.assertEqual(reduction.state, _baseline(state, message))

    def test_invalid_turn_and_policy_still_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            _reduce(IntentState(), "red and leather", 0)
        with self.assertRaises(TypeError):
            apply_user_message_with_trace(  # type: ignore[arg-type]
                IntentState(),
                "red and leather",
                1,
                policy="invalid",
            )

    def test_dense_renderer_matches_product_side_sections(self) -> None:
        reduction = _reduce(
            IntentState(),
            "brand: Alpine Works; material: leather; color: red; under $100",
        )

        self.assertEqual(
            render_dense_query(reduction.state).splitlines(),
            [
                "Brand: Alpine Works",
                "Attributes: Material: leather | Color: red",
                "Price: under $100",
            ],
        )


class Phase11IntentOracleTest(unittest.TestCase):
    def test_frozen_independent_oracle_passes_all_cases(self) -> None:
        self.assertGreaterEqual(phase11_oracle.VALID_ORACLE_CASES, 20_000)
        self.assertGreaterEqual(
            phase11_oracle.BASELINE_EQUIVALENCE_CASES,
            10_000,
        )
        self.assertEqual(
            phase11_oracle.verify_oracle(),
            {
                "cases": phase11_oracle.ORACLE_CASES,
                "valid_cases": phase11_oracle.VALID_ORACLE_CASES,
                "baseline_equivalence_cases": (
                    phase11_oracle.BASELINE_EQUIVALENCE_CASES
                ),
                "digest": phase11_oracle.EXPECTED_SHA256,
                "status": "ok",
            },
        )

    def test_generator_is_deterministic_and_spans_all_fallback_classes(self) -> None:
        first = tuple(phase11_oracle.synthetic_cases())
        second = tuple(phase11_oracle.synthetic_cases())

        self.assertEqual(first, second)
        self.assertEqual(len(first), phase11_oracle.ORACLE_CASES)
        fallback_modes = Counter(
            case.mode for case in first if case.family == "fallback"
        )
        self.assertEqual(
            set(fallback_modes),
            {
                "single_residual",
                "same_slot",
                "ambiguous_slot",
                "unsafe_negation",
                "message_bound",
                "atom_count_bound",
                "value_bound",
            },
        )
        self.assertTrue(all(count >= 1_400 for count in fallback_modes.values()))

    def test_oracle_mismatch_is_aggregate_only(self) -> None:
        case = next(phase11_oracle.synthetic_cases())
        wrong = IntentReduction(
            state=case.state,
            status=IntentReductionStatus.SINGLE_SLOT,
        )
        with mock.patch.object(
            phase11_oracle,
            "apply_user_message_with_trace",
            return_value=wrong,
        ):
            with self.assertRaises(phase11_oracle.OracleExactnessError) as raised:
                phase11_oracle.evaluate_case(case)

        self.assertEqual(raised.exception.cases, 1)
        self.assertNotIn(case.message, str(raised.exception))

    def test_cli_output_is_one_aggregate_json_line(self) -> None:
        aggregate = {
            "cases": phase11_oracle.ORACLE_CASES,
            "valid_cases": phase11_oracle.VALID_ORACLE_CASES,
            "baseline_equivalence_cases": (
                phase11_oracle.BASELINE_EQUIVALENCE_CASES
            ),
            "digest": phase11_oracle.EXPECTED_SHA256,
            "status": "ok",
        }
        output = io.StringIO()
        with (
            mock.patch.object(
                phase11_oracle,
                "verify_oracle",
                return_value=aggregate,
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(phase11_oracle.main(), 0)

        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertEqual(json.loads(output.getvalue()), aggregate)
        for forbidden in ("message", "requirements", "dense_query", "lexical_query"):
            self.assertNotIn(forbidden, output.getvalue())


if __name__ == "__main__":
    unittest.main()
