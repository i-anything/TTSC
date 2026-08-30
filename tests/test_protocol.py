from __future__ import annotations

import dataclasses
import unittest

from conversational_search.protocol import (
    ALLOWED_ATTRIBUTES,
    MAX_CONSTRAINT_CHARACTERS,
    MAX_EVIDENCE_TEXT_CHARACTERS,
    PROTOCOL_ACTIONS,
    CandidateReplyStatus,
    CandidateReplayStatus,
    DisclosureCard,
    ObservedProtocolEvent,
    ProductProtocolEvidence,
    ProtocolEventKind,
    ProtocolMode,
    ReplyMatchStatus,
    build_disclosure_card,
    build_product_protocol_evidence,
    build_protocol_world_model,
    classify_constraint,
    coarse_category,
    eligible_protocol_actions,
    match_protocol_reply,
    project_protocol_world_model,
    remaining_reply,
    replay_protocol_transcript,
)
from evaluator.local_evaluator import ALLOWED_ATTRIBUTES as EVALUATOR_ALLOWED_ATTRIBUTES
from evaluator.local_evaluator import (
    classify_constraint as evaluator_classify_constraint,
)
from evaluator.local_evaluator import coarse_category as evaluator_coarse_category
from evaluator.local_evaluator import customer_reply, intent_card


def _card_dict(card: DisclosureCard) -> dict[str, object]:
    return {
        "target_category": card.target_category,
        "hard_constraints": list(card.hard_constraints),
        "soft_preferences": list(card.soft_preferences),
    }


class DisclosureCardTests(unittest.TestCase):
    def test_reconstruction_exactly_matches_evaluator_order_and_deduplication(
        self,
    ) -> None:
        product = {
            "parent_asin": "SYNTH00001",
            "title": "Black cotton running shoe",
            "features": ["  Flexible sole.  ", "cotton", None, ""],
            "details": {
                "Department": "Women",
                "empty": [],
                "blank": "",
            },
            "description": ["Synthetic fixture description"],
            "categories": ["Clothing", "Women", "Running Shoes"],
            "store": "Fixture Store",
            "price": 49.0,
        }

        actual = build_disclosure_card(product)

        self.assertEqual(_card_dict(actual), intent_card(product))
        self.assertEqual(actual.hard_constraints, ("cotton", "color: black"))
        self.assertEqual(
            actual.soft_preferences,
            ("Flexible sole", "Department: Women"),
        )

    def test_reconstruction_matches_title_fallback_and_constraint_truncation(
        self,
    ) -> None:
        products = [
            {
                "parent_asin": "SYNTH00002",
                "title": "Minimal fixture item",
                "features": [],
                "details": {},
                "description": [],
                "categories": ["Clothing"],
                "store": "",
                "price": None,
            },
            {
                "parent_asin": "SYNTH00003",
                "title": "Trim fixture item",
                "features": ["  " + "x" * 220 + "...  "],
                "details": {},
                "description": [],
                "categories": [],
                "store": "",
                "price": None,
            },
        ]

        for product in products:
            with self.subTest(parent_asin=product["parent_asin"]):
                actual = build_disclosure_card(product)
                self.assertEqual(_card_dict(actual), intent_card(product))
                self.assertLessEqual(
                    max(map(len, (*actual.hard_constraints, *actual.soft_preferences))),
                    MAX_CONSTRAINT_CHARACTERS,
                )

    def test_card_is_immutable_and_rejects_unbounded_external_values(self) -> None:
        card = DisclosureCard("Fixture", ("cotton",), ("color: blue",))

        with self.assertRaises(dataclasses.FrozenInstanceError):
            card.target_category = "changed"  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "at most two"):
            DisclosureCard("Fixture", ("a", "b", "c"), ())
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            DisclosureCard("x" * (MAX_CONSTRAINT_CHARACTERS + 1), (), ())


class ProtocolClassificationTests(unittest.TestCase):
    def test_coarse_category_exactly_matches_evaluator(self) -> None:
        fixtures = [
            [],
            ["Clothing", "Clothing Shoes & Jewelry"],
            [
                "Clothing, Shoes & Jewelry",
                "Women, Athletic",
                "Running Shoes",
            ],
            ["Accessories", "  Jewelry  ", "Bracelets, Charm Bracelets"],
        ]

        for values in fixtures:
            with self.subTest(values=values):
                self.assertEqual(
                    coarse_category(values),
                    evaluator_coarse_category(values),
                )

    def test_constraint_classifier_exactly_matches_every_evaluator_bucket(
        self,
    ) -> None:
        fixtures = [
            "budget around $49.0",
            "under 80",
            "cotton blend",
            "rayon-like fabric",
            "color: brown",
            "bright blue",
            "wide width",
            "department: womens",
            "winter work use",
            "brown",
            "brand: Fixture",
            "breathable cushioning",
        ]

        for value in fixtures:
            with self.subTest(value=value):
                self.assertEqual(
                    classify_constraint(value),
                    evaluator_classify_constraint(value),
                )


class CandidateReplyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.card = DisclosureCard(
            "Synthetic running shoe",
            ("cotton", "color: blue"),
            ("department: womens", "budget around $49.0"),
        )

    def assertEvaluatorEquivalent(
        self,
        ask_attribute: object,
        disclosed_values: tuple[str, ...] = (),
        *,
        boundary_pending: bool = False,
    ) -> None:
        signature = remaining_reply(
            self.card,
            ask_attribute,
            disclosed_values,
            boundary_pending=boundary_pending,
        )
        disclosed = set(disclosed_values)
        sample = {
            "scenario_type": "boundary" if boundary_pending else "browsing",
            "intent_card": _card_dict(self.card),
        }

        expected_text, boundary_used = customer_reply(
            sample,
            ask_attribute,
            disclosed,
            False,
        )

        self.assertEqual(signature.reply_text, expected_text)
        self.assertEqual(signature.boundary_consumed, boundary_used)
        self.assertEqual(
            disclosed,
            set(disclosed_values).union(signature.values),
        )

    def test_reply_signatures_match_evaluator_for_all_response_classes(self) -> None:
        cases = [
            ("other", ("cotton",), False),
            ("material", (), False),
            ("budget", (), False),
            ("brand", (), False),
            ("unsupported", (), False),
            (None, (), False),
            (42, (), False),
            ("", (), False),
            (
                "other",
                (
                    "cotton",
                    "color: blue",
                    "department: womens",
                    "budget around $49.0",
                ),
                False,
            ),
            ("unsupported", (), True),
        ]

        for ask_attribute, disclosed, boundary_pending in cases:
            with self.subTest(
                ask_attribute=ask_attribute,
                disclosed=disclosed,
                boundary_pending=boundary_pending,
            ):
                self.assertEvaluatorEquivalent(
                    ask_attribute,
                    disclosed,
                    boundary_pending=boundary_pending,
                )

    def test_other_discloses_at_most_two_remaining_values_in_card_order(self) -> None:
        signature = remaining_reply(self.card, "other", ("cotton",))

        self.assertIs(signature.status, CandidateReplyStatus.DISCLOSURE)
        self.assertEqual(
            signature.values,
            ("color: blue", "department: womens"),
        )

    def test_invalid_attribute_normalizes_to_other_after_boundary_check(self) -> None:
        ordinary = remaining_reply(self.card, "unsupported", ())
        boundary = remaining_reply(
            self.card,
            "unsupported",
            (),
            boundary_pending=True,
        )

        self.assertEqual(ordinary.attribute, "other")
        self.assertIs(ordinary.status, CandidateReplyStatus.DISCLOSURE)
        self.assertEqual(boundary.attribute, "unsupported")
        self.assertIs(boundary.status, CandidateReplyStatus.BOUNDARY_DECLINE)

    def test_all_ten_official_actions_match_the_evaluator(self) -> None:
        self.assertEqual(set(PROTOCOL_ACTIONS), ALLOWED_ATTRIBUTES)
        self.assertEqual(ALLOWED_ATTRIBUTES, EVALUATOR_ALLOWED_ATTRIBUTES)

        for action in PROTOCOL_ACTIONS:
            with self.subTest(action=action):
                self.assertEvaluatorEquivalent(action)

    def test_fallback_card_duplicate_is_not_silently_deduplicated(self) -> None:
        card = DisclosureCard(
            "Minimal fixture",
            ("Minimal fixture",),
            ("Minimal fixture",),
        )

        signature = remaining_reply(card, "other", ())

        self.assertEqual(
            signature.reply_text,
            "For that, what matters is: Minimal fixture; Minimal fixture.",
        )


def _world_evidence(
    parent_asin: str,
    hard: tuple[str, ...],
    soft: tuple[str, ...] = (),
) -> ProductProtocolEvidence:
    return ProductProtocolEvidence(
        parent_asin=parent_asin,
        coarse_category="Shoes",
        card=DisclosureCard(f"{parent_asin} shoe", hard, soft),
    )


class ProtocolWorldModelTests(unittest.TestCase):
    def test_projection_preserves_retained_mass_and_moves_tail_to_unknown(self) -> None:
        evidence = (
            ProductProtocolEvidence(
                parent_asin="A",
                coarse_category="Shoes",
                card=DisclosureCard("A", ("color: blue",), ()),
            ),
            ProductProtocolEvidence(
                parent_asin="B",
                coarse_category="Shoes",
                card=DisclosureCard("B", ("color: red",), ()),
            ),
            ProductProtocolEvidence(
                parent_asin="C",
                coarse_category="Shoes",
                card=DisclosureCard("C", ("color: green",), ()),
            ),
        )
        world = build_protocol_world_model(
            evidence,
            observed_turn_count=0,
            candidate_weights={"A": 0.6, "B": 0.3, "C": 0.1},
            out_of_pool_probability=0.2,
            boundary_pending=True,
            actions=("color",),
        )

        projected = project_protocol_world_model(world, ("A", "B"))

        original = {
            item.parent_asin: item.probability
            for item in world.candidate_probabilities
        }
        self.assertEqual(
            tuple(item.parent_asin for item in projected.candidate_probabilities),
            ("A", "B"),
        )
        self.assertEqual(
            tuple(item.probability for item in projected.candidate_probabilities),
            (original["A"], original["B"]),
        )
        self.assertAlmostEqual(
            projected.out_of_pool_probability,
            world.out_of_pool_probability + original["C"],
        )
        question = projected.question("color")
        assert question is not None
        self.assertEqual(question.shared_reply_probability, 0.5)
        self.assertAlmostEqual(
            sum(partition.probability for partition in question.partitions)
            + question.unknown_probability
            + question.shared_reply_probability,
            1.0,
        )

    def test_transcript_replay_tracks_accumulated_disclosures(self) -> None:
        card = DisclosureCard(
            "Blue shoe",
            ("cotton", "color: blue"),
            ("waterproof", "budget around $40"),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("cotton",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                attribute="other",
                reply_payload="color: blue; waterproof",
            ),
        )

        replay = replay_protocol_transcript(card, events)

        self.assertIs(replay.status, CandidateReplayStatus.CONSISTENT)
        self.assertEqual(
            replay.disclosed_values,
            ("cotton", "color: blue", "waterproof"),
        )
        self.assertEqual(replay.matched_event_count, 2)

        world = build_protocol_world_model(
            (_world_evidence("A", card.hard_constraints, card.soft_preferences),),
            protocol_events=events,
            observed_turn_count=2,
            out_of_pool_probability=0.1,
        )
        other = world.question("other")
        self.assertIsNotNone(other)
        assert other is not None
        self.assertEqual(
            other.partitions[0].observable_reply,
            "For that, what matters is: budget around $40.",
        )

    def test_partitions_are_deterministic_disjoint_and_observable(self) -> None:
        evidence = (
            _world_evidence("ONE_VALUE", ("x; y",)),
            _world_evidence("TWO_VALUES", ("x", "y")),
        )
        initial = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
        )

        world = build_protocol_world_model(
            evidence,
            protocol_events=initial,
            observed_turn_count=1,
            out_of_pool_probability=0.2,
            boundary_pending=False,
        )

        self.assertIs(world.assessment.mode, ProtocolMode.EXACT)
        self.assertEqual(
            tuple(item.action for item in world.questions),
            PROTOCOL_ACTIONS,
        )
        self.assertAlmostEqual(
            sum(item.probability for item in world.candidate_probabilities)
            + world.out_of_pool_probability,
            1.0,
        )
        other = world.question("other")
        self.assertIsNotNone(other)
        assert other is not None
        self.assertEqual(len(other.partitions), 1)
        self.assertEqual(
            other.partitions[0].candidate_ids,
            ("ONE_VALUE", "TWO_VALUES"),
        )
        self.assertAlmostEqual(other.partitions[0].probability, 0.8)
        self.assertAlmostEqual(other.unknown_probability, 0.2)

        repeated = build_protocol_world_model(
            evidence,
            protocol_events=initial,
            observed_turn_count=1,
            out_of_pool_probability=0.2,
            boundary_pending=False,
        )
        self.assertEqual(world, repeated)

    def test_sparse_browsing_has_explicit_boundary_world_and_unknown_mass(self) -> None:
        world = build_protocol_world_model(
            (
                _world_evidence("BLUE", ("color: blue",)),
                _world_evidence("RED", ("color: red",)),
            ),
            protocol_events=(
                ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
            ),
            observed_turn_count=1,
        )

        self.assertIs(world.assessment.mode, ProtocolMode.AMBIGUOUS)
        self.assertEqual(world.assessment.confidence, 0.5)
        self.assertGreaterEqual(world.out_of_pool_probability, 0.5)
        color = world.question("color")
        self.assertIsNotNone(color)
        assert color is not None
        self.assertIsNotNone(color.shared_reply)
        assert color.shared_reply is not None
        self.assertIs(
            color.shared_reply.status,
            CandidateReplyStatus.BOUNDARY_DECLINE,
        )
        self.assertEqual(color.shared_reply_probability, 0.5)
        self.assertAlmostEqual(
            sum(item.probability for item in color.partitions)
            + color.unknown_probability
            + color.shared_reply_probability,
            1.0,
        )

        boundary_reply = match_protocol_reply(
            world,
            "color",
            color.shared_reply.reply_text,
        )
        self.assertIs(boundary_reply.status, ReplyMatchStatus.SHARED)
        self.assertIs(boundary_reply.next_mode, ProtocolMode.EXACT)

        known_reply = match_protocol_reply(
            world,
            "color",
            color.partitions[0].observable_reply,
        )
        self.assertIs(known_reply.status, ReplyMatchStatus.KNOWN)
        self.assertFalse(known_reply.requires_broad_retrieval)

    def test_boundary_decline_is_consumed_only_once(self) -> None:
        evidence = (_world_evidence("BLUE", ("color: blue",)),)
        events = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.BOUNDARY_DECLINE,
                attribute="color",
            ),
        )

        replay = replay_protocol_transcript(evidence[0].card, events)
        world = build_protocol_world_model(
            evidence,
            protocol_events=events,
            observed_turn_count=2,
            out_of_pool_probability=0.1,
        )

        self.assertTrue(replay.boundary_consumed)
        self.assertIs(world.assessment.mode, ProtocolMode.EXACT)
        color = world.question("color")
        self.assertIsNotNone(color)
        assert color is not None
        self.assertIsNone(color.shared_reply)
        self.assertEqual(
            color.partitions[0].observable_reply,
            "For that, what matters is: color: blue.",
        )

    def test_simulator_impossible_transcript_order_is_rejected(self) -> None:
        invalid_histories = (
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_EXPLICIT,
                    values=("cotton",),
                ),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.BOUNDARY_DECLINE,
                    attribute="color",
                ),
            ),
            (
                ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.BOUNDARY_DECLINE,
                    attribute="color",
                ),
                ObservedProtocolEvent(
                    3,
                    ProtocolEventKind.BOUNDARY_DECLINE,
                    attribute="material",
                ),
            ),
            (
                ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.DISCLOSURE,
                    attribute="color",
                    reply_payload="color: blue",
                ),
                ObservedProtocolEvent(
                    3,
                    ProtocolEventKind.BOUNDARY_DECLINE,
                    attribute="material",
                ),
            ),
            (
                ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
                ObservedProtocolEvent(
                    3,
                    ProtocolEventKind.OVERRIDE,
                    values=("cotton",),
                ),
            ),
        )

        for events in invalid_histories:
            with self.subTest(events=events):
                with self.assertRaises(ValueError):
                    replay_protocol_transcript(
                        DisclosureCard("shoe", ("cotton",), ("classic",)),
                        events,
                    )

    def test_unrecognized_turn_and_unseen_reply_require_broad_recovery(self) -> None:
        evidence = (_world_evidence("BLUE", ("color: blue",)),)
        initial = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
        )
        free_form = build_protocol_world_model(
            evidence,
            protocol_events=initial,
            observed_turn_count=2,
        )

        self.assertIs(free_form.assessment.mode, ProtocolMode.FREE_FORM)
        self.assertEqual(free_form.out_of_pool_probability, 1.0)
        self.assertEqual(free_form.candidate_probabilities, ())
        self.assertEqual(eligible_protocol_actions(free_form), ())

        exact = build_protocol_world_model(
            evidence,
            protocol_events=initial,
            observed_turn_count=1,
            out_of_pool_probability=0.25,
            boundary_pending=False,
        )
        unseen = match_protocol_reply(exact, "color", "I prefer teal mesh.")
        self.assertIs(unseen.status, ReplyMatchStatus.UNSEEN)
        self.assertTrue(unseen.requires_broad_retrieval)
        self.assertIs(unseen.next_mode, ProtocolMode.RECOVERY)
        self.assertEqual(unseen.probability, 0.25)

    def test_impossible_canonical_reply_moves_all_mass_out_of_pool(self) -> None:
        evidence = (
            _world_evidence("BLUE", ("color: blue",)),
            _world_evidence("RED", ("color: red",)),
        )
        events = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                attribute="material",
                reply_payload="cotton",
            ),
        )

        world = build_protocol_world_model(
            evidence,
            protocol_events=events,
            observed_turn_count=2,
        )

        self.assertIs(world.assessment.mode, ProtocolMode.RECOVERY)
        self.assertEqual(world.assessment.consistent_candidate_count, 0)
        self.assertEqual(world.out_of_pool_probability, 1.0)
        self.assertEqual(world.candidate_probabilities, ())

    def test_no_additional_reply_keeps_only_candidates_that_predict_it(self) -> None:
        evidence = (
            _world_evidence("COTTON", ("cotton",)),
            _world_evidence("FEATURE", ("waterproof",)),
        )
        events = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.NO_ADDITIONAL,
                attribute="material",
            ),
        )

        world = build_protocol_world_model(
            evidence,
            protocol_events=events,
            observed_turn_count=2,
            out_of_pool_probability=0.2,
        )

        self.assertIs(world.assessment.mode, ProtocolMode.EXACT)
        self.assertEqual(
            tuple(item.parent_asin for item in world.candidate_probabilities),
            ("FEATURE",),
        )
        self.assertAlmostEqual(world.candidate_probabilities[0].probability, 0.8)

    def test_official_override_replays_but_natural_correction_fails_open(self) -> None:
        evidence = (_world_evidence("COTTON", ("cotton",), ("classic",)),)
        official = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_TENTATIVE,
                values=("classic",),
            ),
            ObservedProtocolEvent(2, ProtocolEventKind.NEED_ATTRIBUTE),
            ObservedProtocolEvent(
                3,
                ProtocolEventKind.OVERRIDE,
                values=("cotton",),
            ),
        )
        exact = build_protocol_world_model(
            evidence,
            protocol_events=official,
            observed_turn_count=3,
            out_of_pool_probability=0.1,
        )
        natural_correction = build_protocol_world_model(
            evidence,
            protocol_events=official[:1],
            observed_turn_count=2,
        )

        self.assertIs(exact.assessment.mode, ProtocolMode.EXACT)
        self.assertEqual(
            replay_protocol_transcript(
                evidence[0].card,
                official,
            ).disclosed_values,
            ("cotton",),
        )
        self.assertIs(natural_correction.assessment.mode, ProtocolMode.FREE_FORM)
        self.assertEqual(natural_correction.out_of_pool_probability, 1.0)

        wrong_tentative = build_protocol_world_model(
            evidence,
            protocol_events=(
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_TENTATIVE,
                    values=("not on the card",),
                ),
            ),
            observed_turn_count=1,
        )
        self.assertIs(wrong_tentative.assessment.mode, ProtocolMode.RECOVERY)
        self.assertEqual(wrong_tentative.out_of_pool_probability, 1.0)

    def test_eligibility_removes_repeated_resolved_and_invariant_actions(self) -> None:
        world = build_protocol_world_model(
            (
                _world_evidence("BLUE", ("color: blue", "cotton")),
                _world_evidence("RED", ("color: red", "leather")),
            ),
            protocol_events=(
                ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
            ),
            observed_turn_count=1,
            out_of_pool_probability=0.1,
            boundary_pending=False,
        )

        eligible = eligible_protocol_actions(
            world,
            asked_actions=("other",),
            resolved_actions=("color",),
        )

        self.assertNotIn("other", eligible)
        self.assertNotIn("color", eligible)
        self.assertNotIn("category", eligible)
        self.assertNotIn("brand", eligible)
        self.assertIn("material", eligible)

    def test_default_residual_mass_is_monotone_and_weights_need_not_be_probabilities(
        self,
    ) -> None:
        initial = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_EXPLICIT, values=("x",)),
        )
        one = build_protocol_world_model(
            (_world_evidence("A", ("x",)),),
            protocol_events=initial,
            observed_turn_count=1,
            candidate_weights={"A": 10.0},
        )
        three = build_protocol_world_model(
            tuple(_world_evidence(parent_asin, ("x",)) for parent_asin in ("A", "B", "C")),
            protocol_events=initial,
            observed_turn_count=1,
            candidate_weights={"A": 10.0, "B": 5.0, "C": 1.0},
        )
        unsupported = build_protocol_world_model(
            (_world_evidence("NO_MATCH", ("y",)),),
            protocol_events=initial,
            observed_turn_count=1,
        )

        self.assertGreater(one.out_of_pool_probability, three.out_of_pool_probability)
        self.assertEqual(unsupported.out_of_pool_probability, 1.0)
        self.assertAlmostEqual(
            sum(item.probability for item in three.candidate_probabilities)
            + three.out_of_pool_probability,
            1.0,
        )

    def test_degenerate_pools_return_typed_recovery(self) -> None:
        initial = (
            ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
        )
        all_unknown = build_protocol_world_model(
            (_world_evidence("A", ("x",)),),
            protocol_events=initial,
            observed_turn_count=1,
            out_of_pool_probability=1.0,
            boundary_pending=False,
        )
        empty = build_protocol_world_model(
            (),
            protocol_events=initial,
            observed_turn_count=1,
            out_of_pool_probability=0.2,
            boundary_pending=False,
        )

        for world in (all_unknown, empty):
            with self.subTest(reason=world.assessment.fallback_reason):
                self.assertIs(world.assessment.mode, ProtocolMode.RECOVERY)
                self.assertEqual(world.candidate_probabilities, ())
                self.assertEqual(world.out_of_pool_probability, 1.0)


class ProductProtocolEvidenceTests(unittest.TestCase):
    def test_builder_keeps_card_exact_while_bounding_transient_text(self) -> None:
        product = {
            "parent_asin": "SYNTH00004",
            "title": "Minimal fixture",
            "features": [],
            "details": {},
            "description": ["x" * 40_000 + " cotton"],
            "categories": ["Clothing", "Women", "Casual Shoes"],
            "store": "Fixture Store",
            "price": 19.99,
            "rating_number": 12,
        }

        evidence = build_product_protocol_evidence(product)

        self.assertEqual(_card_dict(evidence.card), intent_card(product))
        self.assertEqual(
            evidence.coarse_category,
            evaluator_coarse_category(product["categories"]),
        )
        self.assertEqual(len(evidence.text), MAX_EVIDENCE_TEXT_CHARACTERS)
        self.assertEqual(evidence.price, "19.99")
        self.assertEqual(evidence.popularity, 12)


if __name__ == "__main__":
    unittest.main()
