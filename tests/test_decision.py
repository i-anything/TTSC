from __future__ import annotations

import unittest

from conversational_search.decision import (
    Bm25OnlyConditions,
    ProtocolDecisionStatus,
    ProtocolObservation,
    _question_utility_upper_bound,
    derive_bm25_only_conditions,
    exact_query_constraints,
    intent_override_is_locked,
    parse_protocol_event,
    plan_expected_utility_decision,
    plan_protocol_decision,
    protocol_events_are_structured_for_routing,
    protocol_observation_attribute,
    protocol_route_dependency_digest,
    protocol_state_is_consistent,
    recognize_protocol_observation,
)
from conversational_search.exact_evidence import rank_exact_evidence
from conversational_search.intent import IntentState, Requirement
from conversational_search.protocol import (
    CandidateReplyStatus,
    DisclosureCard,
    ProductProtocolEvidence,
    QuestionReplyModel,
    ReplyPartition,
)
from conversational_search.protocol import ObservedProtocolEvent, ProtocolEventKind
from conversational_search.utility_planner import (
    ExpectedUtilityCandidate,
    RetrievalChoice,
    SimulatedQuestion,
    plan_expected_utility,
)
from conversational_search.slates import SlateState


def _evidence(parent_asin: str, color: str) -> ProductProtocolEvidence:
    return ProductProtocolEvidence(
        parent_asin=parent_asin,
        coarse_category="Shoes",
        card=DisclosureCard(
            f"{color.title()} shoe",
            (f"color: {color}",),
            ("style: classic accent",),
        ),
        text=f"{color} shoe",
    )


class ProtocolObservationTest(unittest.TestCase):
    def test_only_published_shapes_enter_protocol_mode(self) -> None:
        cases = {
            (
                "I'm looking for Shoes, but I'm still exploring."
            ): ProtocolObservation.INITIAL,
            "For that, what matters is: color: blue.": ProtocolObservation.DISCLOSURE,
            (
                "Actually, ignore my earlier preference. "
                "What I need is: color: blue."
            ): ProtocolObservation.OVERRIDE,
            (
                "I don't have a preference for other; "
                "please use your judgment."
            ): ProtocolObservation.BOUNDARY_DECLINE,
            "I need comfy shoes, maybe blue": ProtocolObservation.UNSUPPORTED,
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertIs(
                    recognize_protocol_observation(
                        message,
                        1 if expected is ProtocolObservation.INITIAL else 2,
                    ),
                    expected,
                )

        message = "I don't have an additional preference for material."
        observation = recognize_protocol_observation(message, 2)
        self.assertEqual(
            protocol_observation_attribute(message, observation),
            "material",
        )

        disclosure = "For that, what matters is: cotton; color: blue."
        event = parse_protocol_event(
            disclosure,
            recognize_protocol_observation(disclosure, 2),
            2,
            asked_attribute="other",
        )
        self.assertIs(event.kind, ProtocolEventKind.DISCLOSURE)
        self.assertEqual(event.attribute, "other")
        self.assertEqual(event.values, ())
        self.assertEqual(event.reply_payload, "cotton; color: blue")
        self.assertEqual(event.serialized_reply_values, "cotton; color: blue")

    def test_tentative_parser_preserves_multiple_sentences_after_category(self) -> None:
        message = (
            "I'm looking for Shoes. "
            "First preference sentence. Second preference sentence"
        )
        observation = recognize_protocol_observation(message, 1)

        event = parse_protocol_event(
            message,
            observation,
            1,
            asked_attribute=None,
        )

        self.assertIs(event.kind, ProtocolEventKind.INITIAL_TENTATIVE)
        self.assertEqual(
            event.values,
            ("First preference sentence. Second preference sentence",),
        )

    def test_event_log_is_authoritative_for_semicolon_card_values(self) -> None:
        value = "waterproof; machine washable"
        state = IntentState(
            requirements=(
                Requirement(value, "initial_explicit", 1, "feature"),
            )
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=(value,),
            ),
        )

        self.assertEqual(exact_query_constraints(state, events), (value,))

    def test_exact_query_uses_only_strong_atoms_and_lock_uses_tentative_source(
        self,
    ) -> None:
        state = IntentState(
            requirements=(
                Requirement("old preference", "initial_tentative", 1, "feature"),
                Requirement("cotton; color: blue", "answer", 2, "other"),
            )
        )

        self.assertEqual(exact_query_constraints(state), ("cotton", "color: blue"))
        self.assertTrue(intent_override_is_locked(state))
        self.assertFalse(
            intent_override_is_locked(
                IntentState(
                    requirements=(
                        *state.requirements,
                        Requirement("cotton", "override", 3, "material"),
                    )
                )
            )
        )

    def test_override_exact_query_uses_current_state_not_stale_initial_event(
        self,
    ) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("leather sole", "override", 2, "feature"),
            ),
            intent_version=1,
            last_turn=2,
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("rubber sole",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.OVERRIDE,
                values=("leather sole",),
            ),
        )

        self.assertTrue(protocol_state_is_consistent(state, events, 2))
        self.assertEqual(exact_query_constraints(state, events), ("leather sole",))

    def test_bm25_only_conditions_require_every_boolean_and_bm25_support(
        self,
    ) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("rubber sole", "initial_explicit", 1, "feature"),
            ),
            last_turn=1,
        )
        inputs = {
            "message_is_exact_protocol": True,
            "session_state_is_consistent": True,
            "category_is_exactly_recognized": True,
            "exact_product_constraints": 1,
            "session_forces_hybrid": False,
            "protocol_values_are_structured": True,
        }

        conditions = derive_bm25_only_conditions(state, **inputs)

        self.assertTrue(conditions.pre_bm25_eligible)
        self.assertFalse(conditions.bm25_only)
        self.assertTrue(conditions.with_bm25_support(True).bm25_only)
        for key in (
            "message_is_exact_protocol",
            "session_state_is_consistent",
            "category_is_exactly_recognized",
            "protocol_values_are_structured",
        ):
            with self.subTest(condition=key):
                changed = dict(inputs)
                changed[key] = False
                self.assertFalse(
                    derive_bm25_only_conditions(
                        state,
                        **changed,
                    ).pre_bm25_eligible
                )
        self.assertFalse(
            derive_bm25_only_conditions(
                state,
                **{**inputs, "exact_product_constraints": 0},
            ).pre_bm25_eligible
        )
        self.assertFalse(
            derive_bm25_only_conditions(
                state,
                **{**inputs, "session_forces_hybrid": True},
            ).pre_bm25_eligible
        )

        free_text_state = IntentState(
            category="Shoes",
            requirements=(
                *state.requirements,
                Requirement("something unusual", "free_text", 1),
            ),
            last_turn=1,
        )
        self.assertFalse(
            derive_bm25_only_conditions(
                free_text_state,
                **inputs,
            ).no_unparsed_or_free_text_requirement
        )

    def test_protocol_state_and_disclosure_shape_gate_routing(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("rubber sole", "initial_explicit", 1, "feature"),
            ),
            last_turn=1,
        )
        initial = ObservedProtocolEvent(
            1,
            ProtocolEventKind.INITIAL_EXPLICIT,
            values=("rubber sole",),
        )
        self.assertTrue(protocol_state_is_consistent(state, (initial,), 1))
        self.assertFalse(
            protocol_state_is_consistent(
                IntentState(requirements=state.requirements, last_turn=1),
                (initial,),
                1,
            )
        )
        self.assertFalse(
            protocol_state_is_consistent(
                state,
                (
                    ObservedProtocolEvent(
                        1,
                        ProtocolEventKind.INITIAL_EXPLICIT,
                        values=("truncated requirement",),
                    ),
                ),
                1,
            )
        )

        material = ObservedProtocolEvent(
            2,
            ProtocolEventKind.DISCLOSURE,
            "material",
            reply_payload="cotton",
        )
        opaque = ObservedProtocolEvent(
            2,
            ProtocolEventKind.DISCLOSURE,
            "other",
            reply_payload="cotton; color: blue",
        )
        self.assertTrue(
            protocol_events_are_structured_for_routing((initial, material))
        )
        self.assertFalse(
            protocol_events_are_structured_for_routing((initial, opaque))
        )
        self.assertEqual(
            exact_query_constraints(state, (initial, material)),
            ("rubber sole", "cotton"),
        )

        conflicting_slot_state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("rubber sole", "initial_explicit", 1, "feature"),
                Requirement("leather sole", "answer", 2, "feature"),
            ),
            asked_attributes=("feature",),
            last_asked_attribute="feature",
            last_turn=2,
        )
        conflicting_slot_events = (
            initial,
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "feature",
                reply_payload="leather sole",
            ),
        )
        self.assertFalse(
            protocol_state_is_consistent(
                conflicting_slot_state,
                conflicting_slot_events,
                2,
            )
        )

    def test_route_dependency_hash_covers_events_conditions_and_support(self) -> None:
        conditions = Bm25OnlyConditions(True, True, True, 1, True, True)
        initial = ObservedProtocolEvent(
            1,
            ProtocolEventKind.INITIAL_EXPLICIT,
            values=("rubber sole",),
        )
        first = protocol_route_dependency_digest(
            conditions,
            (initial,),
            ("A",),
        )

        self.assertEqual(len(first), 64)
        self.assertNotEqual(
            first,
            protocol_route_dependency_digest(
                conditions.with_bm25_support(True),
                (initial,),
                ("A",),
            ),
        )
        self.assertNotEqual(
            first,
            protocol_route_dependency_digest(
                conditions,
                (initial,),
                ("B",),
            ),
        )
        changed_event = ObservedProtocolEvent(
            1,
            ProtocolEventKind.INITIAL_EXPLICIT,
            values=("leather sole",),
        )
        self.assertNotEqual(
            first,
            protocol_route_dependency_digest(
                conditions,
                (changed_event,),
                ("A",),
            ),
        )

    def test_overlong_exact_event_is_rejected_instead_of_truncated(self) -> None:
        message = (
            "I'm looking for Shoes. A key requirement is: "
            + "x" * 1_025
            + "."
        )
        observation = recognize_protocol_observation(message, 1)

        with self.assertRaisesRegex(ValueError, "character bound"):
            parse_protocol_event(
                message,
                observation,
                1,
                asked_attribute=None,
            )

class ProtocolDecisionTest(unittest.TestCase):
    def test_expected_planner_projects_large_world_into_partition_budget(self) -> None:
        evidence = tuple(
            ProductProtocolEvidence(
                parent_asin=f"ITEM{index:03d}",
                coarse_category="Shoes",
                card=DisclosureCard(
                    f"Item {index}",
                    (f"feature code {index}",),
                    (),
                ),
                text=f"feature code {index}",
            )
            for index in range(100)
        )
        candidate_ids = tuple(item.parent_asin for item in evidence)
        state = IntentState(category="Shoes", last_turn=1)
        events = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
        )
        exact = rank_exact_evidence(
            candidate_ids,
            evidence,
            state,
            protocol_events=events,
        )

        decision = plan_expected_utility_decision(
            state,
            exact,
            evidence,
            slate_state=SlateState(),
            ranking_signature=(0, "synthetic-large-world"),
            protocol_events=events,
            current_turn=1,
            requested_top_k=10,
        )

        self.assertIsNot(
            decision.status,
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
        )
        self.assertLess(decision.trace.support_count, len(evidence))
        self.assertLessEqual(decision.trace.simulated_partition_count, 128)

    def test_question_upper_bound_dominates_a_realizable_rerank(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.6),
            ExpectedUtilityCandidate("B", 2, 0.3),
        )
        question = QuestionReplyModel(
            "color",
            (
                ReplyPartition(
                    "For that, what matters is: blue.",
                    CandidateReplyStatus.DISCLOSURE,
                    ("A",),
                    0.6,
                ),
                ReplyPartition(
                    "For that, what matters is: red.",
                    CandidateReplyStatus.DISCLOSURE,
                    ("B",),
                    0.3,
                ),
            ),
            unknown_probability=0.1,
        )
        simulated = SimulatedQuestion(
            "color",
            (("A", 1), ("B", 1)),
            ordinary_post_ranks_by_width=(
                (0, (("A", 1), ("B", 1))),
                (1, (("A", None), ("B", 1))),
                (2, (("A", None), ("B", None))),
            ),
        )
        plan = plan_expected_utility(
            candidates,
            (simulated,),
            current_turn=1,
            top_k=2,
            widths=(0, 1, 2),
            retrieval_choices=(RetrievalChoice.REUSE,),
            out_of_pool_probability=0.1,
            protocol_confidence=1.0,
            allow_zero_width=True,
            no_question_post_ranks_by_width=(
                (0, (("A", 1), ("B", 2))),
                (1, (("A", 1), ("B", 2))),
                (2, (("A", 1), ("B", 2))),
            ),
        )

        upper_bound = _question_utility_upper_bound(
            candidates,
            question,
            widths=(0, 1, 2),
            current_turn=1,
            top_k=2,
            protocol_locked=False,
            computation_cost=0.0,
        )

        question_values = [
            action.value
            for action in (plan.selected, plan.runner_up)
            if action is not None and action.question == "color"
        ]
        self.assertTrue(question_values)
        self.assertLessEqual(max(question_values), upper_bound)

    def test_joint_planner_selects_an_informative_question_and_dynamic_width(
        self,
    ) -> None:
        evidence = (
            _evidence("BLUE", "blue"),
            _evidence("RED", "red"),
            _evidence("GREEN", "green"),
        )
        state = IntentState(category="Shoes")
        exact = rank_exact_evidence(
            ("BLUE", "RED", "GREEN"),
            evidence,
            state,
        )

        decision = plan_protocol_decision(
            state,
            exact,
            evidence,
            current_turn=1,
            requested_top_k=2,
        )

        self.assertIs(decision.status, ProtocolDecisionStatus.APPLIED)
        self.assertEqual(decision.width, 1)
        # ``other`` and ``color`` induce the same partition here.  Preserve the
        # declared protocol-question order so the wildcard wins only that tie.
        self.assertEqual(decision.question, "other")
        self.assertEqual(decision.trace.support_count, 3)

    def test_equal_question_utilities_prefer_the_other_wildcard(self) -> None:
        evidence = tuple(
            ProductProtocolEvidence(
                parent_asin=parent_asin,
                coarse_category="Shoes",
                card=DisclosureCard(
                    f"{parent_asin} shoe",
                    ("shared feature",),
                    (),
                ),
            )
            for parent_asin in ("FIRST", "SECOND")
        )
        state = IntentState(category="Shoes")
        exact = rank_exact_evidence(
            tuple(item.parent_asin for item in evidence),
            evidence,
            state,
        )

        decision = plan_protocol_decision(
            state,
            exact,
            evidence,
            current_turn=2,
            requested_top_k=1,
        )

        self.assertIs(decision.status, ProtocolDecisionStatus.APPLIED)
        self.assertEqual(decision.question, "other")

    def test_lock_allows_zero_width_and_final_turn_forces_full_width(self) -> None:
        evidence = (_evidence("BLUE", "blue"), _evidence("RED", "red"))
        locked_state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("classic", "initial_tentative", 1, "feature"),
            ),
        )
        locked_exact = rank_exact_evidence(
            ("BLUE", "RED"), evidence, locked_state
        )
        locked = plan_protocol_decision(
            locked_state,
            locked_exact,
            evidence,
            current_turn=1,
            requested_top_k=2,
        )

        final_state = IntentState(category="Shoes")
        final_exact = rank_exact_evidence(("BLUE", "RED"), evidence, final_state)
        final = plan_protocol_decision(
            final_state,
            final_exact,
            evidence,
            current_turn=10,
            requested_top_k=2,
        )

        self.assertEqual(locked.width, 0)
        self.assertTrue(locked.trace.protocol_locked)
        self.assertEqual((final.question, final.width), (None, 2))

    def test_shown_best_tier_is_conditioned_out_and_zero_support_fails_open(
        self,
    ) -> None:
        evidence = (_evidence("BLUE", "blue"), _evidence("RED", "red"))
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("color: blue", "answer", 2, "color"),
            ),
        )
        exact = rank_exact_evidence(("BLUE", "RED"), evidence, state)

        exhausted = plan_protocol_decision(
            state,
            exact,
            evidence,
            shown_ids=("BLUE",),
            current_turn=3,
            requested_top_k=2,
        )

        self.assertIs(
            exhausted.status,
            ProtocolDecisionStatus.FAIL_OPEN_NO_SUPPORT,
        )
        self.assertEqual(exhausted.width, 2)

    def test_conditioning_out_the_best_tier_uses_residual_consistent_support(
        self,
    ) -> None:
        evidence = (
            ProductProtocolEvidence(
                parent_asin="BEST",
                coarse_category="Shoes",
                card=DisclosureCard(
                    "Blue shoe",
                    ("color: blue",),
                    ("waterproof",),
                ),
                text="blue waterproof shoe",
            ),
            ProductProtocolEvidence(
                parent_asin="RESIDUAL",
                coarse_category="Shoes",
                card=DisclosureCard(
                    "Blue shoe",
                    ("color: blue",),
                    ("breathable",),
                ),
                text="blue breathable shoe",
            ),
        )
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("color: blue", "initial_explicit", 1, "color"),
                Requirement("waterproof", "initial_tentative", 1, "feature"),
            ),
        )
        exact = rank_exact_evidence(("BEST", "RESIDUAL"), evidence, state)
        self.assertEqual(
            tuple(item.parent_asin for item in exact.beliefs),
            ("BEST",),
        )

        decision = plan_protocol_decision(
            state,
            exact,
            evidence,
            shown_ids=("BEST",),
            current_turn=2,
            requested_top_k=2,
        )

        self.assertIs(decision.status, ProtocolDecisionStatus.APPLIED)
        self.assertEqual(decision.ordered_ids, ("RESIDUAL",))
        self.assertEqual(decision.trace.support_count, 1)


if __name__ == "__main__":
    unittest.main()
