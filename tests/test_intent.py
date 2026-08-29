from __future__ import annotations

import unittest

from conversational_search.intent import (
    CANONICAL_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
    IntentState,
    Requirement,
    apply_user_message,
    record_question,
    render_dense_query,
    render_lexical_query,
)


class IntentStateTest(unittest.TestCase):
    @staticmethod
    def _answer(
        state: IntentState,
        attribute: str,
        value: str,
        *,
        turn: int = 2,
    ) -> IntentState:
        questioned = record_question(state, attribute)
        return apply_user_message(
            questioned,
            f"For that, what matters is: {value}.",
            turn,
        )

    def test_public_types_and_empty_state(self) -> None:
        self.assertIsInstance(IntentState, type)
        self.assertIsInstance(Requirement, type)
        state = IntentState()
        self.assertEqual(render_dense_query(state), "")
        self.assertEqual(render_lexical_query(state), "")

    def test_buying_message_extracts_category_and_explicit_requirement(self) -> None:
        empty = IntentState()
        state = apply_user_message(
            empty,
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
        )

        self.assertEqual(
            render_dense_query(state),
            "Category: Shoes\nAttributes: Material: leather",
        )
        lexical = render_lexical_query(state)
        self.assertIn("Shoes", lexical)
        self.assertIn("leather", lexical)
        self.assertNotIn("key requirement", lexical.casefold())
        self.assertEqual(render_dense_query(empty), "")

    def test_browsing_message_keeps_only_category(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )

        self.assertEqual(render_dense_query(state), "Category: Shoes")
        self.assertEqual(render_lexical_query(state), "Shoes")

    def test_generated_positive_answer_preserves_internal_punctuation(self) -> None:
        browsing = apply_user_message(
            IntentState(),
            "I'm looking for Bags, but I'm still exploring.",
            1,
        )
        value = (
            "Weatherproof shell; two interior pockets. Packs flat for travel; "
            "still breathable in humid weather"
        )
        questioned = record_question(browsing, "feature")
        updated = apply_user_message(
            questioned,
            f"For that, what matters is: {value}.",
            2,
        )

        self.assertEqual(
            render_dense_query(updated),
            f"Category: Bags\nSearch Clues: {value}",
        )
        self.assertIn(value, render_lexical_query(updated))
        self.assertEqual(render_dense_query(questioned), "Category: Bags")
        self.assertEqual(render_dense_query(browsing), "Category: Bags")

    def test_override_removes_initial_source_but_preserves_answer_source(self) -> None:
        initial_messages = (
            "I'm looking for Accessories Belts. Buckle closure",
            "I'm looking for Accessories Belts. A key requirement is: Buckle closure.",
        )
        for initial_message in initial_messages:
            for override_turn in (3, 4):
                with self.subTest(initial=initial_message, turn=override_turn):
                    initial = apply_user_message(
                        IntentState(),
                        initial_message,
                        1,
                    )
                    disclosed = self._answer(initial, "brand", "Alpine Works", turn=2)
                    before_override = render_dense_query(disclosed)

                    updated = apply_user_message(
                        disclosed,
                        "Actually, ignore my earlier preference. What I need is: leather.",
                        override_turn,
                    )

                    dense = render_dense_query(updated)
                    lexical = render_lexical_query(updated)
                    self.assertEqual(
                        dense,
                        "Category: Accessories Belts\n"
                        "Brand: Alpine Works\n"
                        "Attributes: Material: leather",
                    )
                    self.assertNotIn("Buckle closure", dense)
                    self.assertNotIn("Buckle closure", lexical)
                    self.assertIn("Alpine Works", lexical)
                    self.assertIn("leather", lexical)
                    self.assertEqual(render_dense_query(disclosed), before_override)

    def test_no_preference_and_no_additional_clear_the_asked_attribute(self) -> None:
        material = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
        )
        material_question = record_question(material, "material")
        without_material = apply_user_message(
            material_question,
            "I don't have a preference for material; please use your judgment.",
            2,
        )

        self.assertEqual(render_dense_query(without_material), "Category: Shoes")
        material_lexical = render_lexical_query(without_material)
        self.assertNotIn("leather", material_lexical)
        self.assertNotIn("preference", material_lexical.casefold())
        self.assertNotIn("judgment", material_lexical.casefold())

        brand = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        brand = self._answer(brand, "brand", "Alpine Works", turn=2)
        brand_question = record_question(brand, "brand")
        without_brand = apply_user_message(
            brand_question,
            "I don't have an additional preference for brand.",
            3,
        )

        self.assertEqual(render_dense_query(without_brand), "Category: Shoes")
        brand_lexical = render_lexical_query(without_brand)
        self.assertNotIn("Alpine Works", brand_lexical)
        self.assertNotIn("additional preference", brand_lexical.casefold())

    def test_unknown_useful_message_becomes_free_text(self) -> None:
        message = "I need something packable for monsoon commutes."
        state = apply_user_message(IntentState(), message, 1)

        dense = render_dense_query(state)
        lexical = render_lexical_query(state)
        self.assertEqual(dense.splitlines()[0].split(":", 1)[0], "Search Clues")
        self.assertIn("packable for monsoon commutes", dense.casefold())
        self.assertIn("packable for monsoon commutes", lexical.casefold())

    def test_robust_policy_handles_held_out_natural_phrasings(self) -> None:
        cases = (
            (
                "I'm looking for Shoes. A key requirement is: leather.",
                "Please help me find Shoes. It must have leather.",
            ),
            (
                "I'm looking for Shoes, but I'm still exploring.",
                "Show me some Shoes; I'm open to options.",
            ),
            (
                "I'm looking for Shoes. leather",
                "Looking for Shoes, maybe leather.",
            ),
        )
        for canonical, natural in cases:
            with self.subTest(natural=natural):
                expected = apply_user_message(
                    IntentState(),
                    canonical,
                    1,
                    policy=CANONICAL_INTENT_POLICY,
                )
                observed = apply_user_message(
                    IntentState(),
                    natural,
                    1,
                    policy=ROBUST_INTENT_POLICY,
                )
                self.assertEqual(observed, expected)

        browsing = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        questioned = record_question(browsing, "material")
        expected_answer = apply_user_message(
            questioned,
            "For that, what matters is: leather.",
            2,
            policy=CANONICAL_INTENT_POLICY,
        )
        natural_answer = apply_user_message(
            questioned,
            "My preference there is leather.",
            2,
            policy=ROBUST_INTENT_POLICY,
        )
        self.assertEqual(natural_answer, expected_answer)

        expected_no_preference = apply_user_message(
            record_question(expected_answer, "material"),
            "I don't have an additional preference for material.",
            3,
            policy=CANONICAL_INTENT_POLICY,
        )
        natural_no_preference = apply_user_message(
            record_question(natural_answer, "material"),
            "Any material is fine.",
            3,
            policy=ROBUST_INTENT_POLICY,
        )
        self.assertEqual(natural_no_preference, expected_no_preference)

        natural_override = apply_user_message(
            natural_answer,
            "Scratch that. What I really need is cotton.",
            3,
            policy=ROBUST_INTENT_POLICY,
        )
        self.assertEqual(
            [item.value for item in natural_override.requirements],
            ["cotton"],
        )
        self.assertEqual(natural_override.intent_version, 1)

    def test_general_cues_do_not_become_harmful_free_text(self) -> None:
        buying = apply_user_message(
            IntentState(),
            "I need Shoes, with this requirement: leather.",
            1,
        )
        self.assertEqual(buying.category, "Shoes")
        self.assertEqual(buying.requirements[-1].value, "leather")

        browsing = apply_user_message(
            IntentState(),
            "Show me Shoes; I'm keeping my options open.",
            1,
        )
        questioned = record_question(browsing, "material")
        answer = apply_user_message(
            questioned,
            "For that attribute, I care about: leather.",
            2,
        )
        self.assertEqual(answer.requirements[-1].attribute, "material")

        cleared = apply_user_message(
            record_question(answer, "material"),
            "No preference on material; you decide.",
            3,
        )
        self.assertEqual(cleared.requirements, ())
        self.assertIn("material", cleared.no_preference)

        no_additional = apply_user_message(
            record_question(cleared, "color"),
            "I have nothing else to add for color.",
            4,
        )
        self.assertIn("color", no_additional.no_preference)

        ignored = apply_user_message(
            record_question(no_additional, "feature"),
            "No luck so far—could you ask one precise follow-up?",
            5,
        )
        self.assertEqual(ignored.requirements, ())

    def test_successive_strong_overrides_remove_contradictions(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
        )
        state = apply_user_message(
            state,
            "I've changed my mind—replace my earlier preference with cotton.",
            2,
        )
        state = apply_user_message(
            state,
            "Scratch that. What I really need is wool.",
            3,
        )

        self.assertEqual([item.value for item in state.requirements], ["wool"])
        self.assertEqual(state.intent_version, 2)

    def test_replace_override_preserves_unrelated_answers_and_strips_scaffold(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
        )
        state = self._answer(state, "brand", "Alpine Works", turn=2)
        state = apply_user_message(
            state,
            "I've changed my mind—replace my earlier preference with: cotton.",
            3,
        )

        self.assertEqual(
            [item.value for item in state.requirements],
            ["Alpine Works", "cotton"],
        )

    def test_replace_override_supersedes_unknown_value_in_the_same_slot(self) -> None:
        cases = (
            (
                "I'm looking for Shoes. A key requirement is: red.",
                "I've changed my mind—replace my earlier preference with navy.",
                "color",
            ),
            (
                "I'm looking for Shoes. A key requirement is: Brand: Acme.",
                "I've changed my mind—replace my earlier preference with Nike.",
                "brand",
            ),
        )
        for initial_message, override_message, expected_attribute in cases:
            with self.subTest(attribute=expected_attribute):
                state = apply_user_message(IntentState(), initial_message, 1)
                state = apply_user_message(state, override_message, 2)

                self.assertEqual(len(state.requirements), 1)
                self.assertEqual(state.requirements[0].source, "override")
                self.assertEqual(state.requirements[0].attribute, expected_attribute)
                self.assertNotIn("red", render_lexical_query(state).casefold())
                self.assertNotIn("acme", render_lexical_query(state).casefold())

        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
        )
        state = self._answer(state, "brand", "Brand: Acme", turn=2)
        state = apply_user_message(
            state,
            "I've changed my mind—replace my earlier preference with Nike.",
            3,
        )
        self.assertEqual(
            [(item.value, item.attribute) for item in state.requirements],
            [("leather", "material"), ("Nike", "brand")],
        )

        state = apply_user_message(
            state,
            "I've changed my mind—replace my earlier preference with cotton.",
            4,
        )
        self.assertEqual(
            [(item.value, item.attribute) for item in state.requirements],
            [("Nike", "brand"), ("cotton", "material")],
        )

        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: red.",
            1,
        )
        state = self._answer(state, "brand", "Brand: Acme", turn=2)
        state = apply_user_message(
            state,
            "I've changed my mind—replace my earlier preference with charcoal.",
            3,
        )
        self.assertEqual(
            [(item.value, item.attribute) for item in state.requirements],
            [("Brand: Acme", "brand"), ("charcoal", "color")],
        )

        ambiguous = apply_user_message(
            self._answer(
                apply_user_message(
                    IntentState(),
                    "I'm looking for Shoes. A key requirement is: red.",
                    1,
                ),
                "brand",
                "Brand: Acme",
                turn=2,
            ),
            "I've changed my mind—replace my earlier preference with zaffre.",
            3,
        )
        self.assertEqual(
            [(item.value, item.attribute) for item in ambiguous.requirements],
            [
                ("red", "color"),
                ("Brand: Acme", "brand"),
                ("zaffre", "feature"),
            ],
        )

    def test_unmatched_follow_up_stays_free_text_but_polite_bare_answer_is_typed(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        material_question = record_question(state, "material")
        unrelated = apply_user_message(
            material_question,
            "I also need it under $100.",
            2,
        )
        self.assertEqual(unrelated.requirements[-1].source, "free_text")
        self.assertIsNone(unrelated.requirements[-1].attribute)

        polite = apply_user_message(
            material_question,
            "Leather, please.",
            2,
        )
        self.assertEqual(polite.requirements[-1].value, "Leather")
        self.assertEqual(polite.requirements[-1].attribute, "material")

        ambiguous_refinements = (
            (material_question, "I need it under $100."),
            (material_question, "Show me cheaper ones, please."),
            (record_question(state, "feature"), "Show me cheaper ones, please."),
            (record_question(state, "feature"), "Different products, please."),
            (record_question(state, "feature"), "Cheaper alternatives, please."),
            (record_question(state, "feature"), "More variety, please."),
        )
        for questioned, message in ambiguous_refinements:
            with self.subTest(message=message, attribute=questioned.last_asked_attribute):
                refinement = apply_user_message(questioned, message, 2)
                self.assertEqual(refinement.requirements[-1].source, "free_text")
                self.assertIsNone(refinement.requirements[-1].attribute)

        budget_question = record_question(state, "budget")
        budget = apply_user_message(budget_question, "Under $100, please.", 2)
        self.assertEqual(budget.requirements[-1].value, "Under $100")
        self.assertEqual(budget.requirements[-1].attribute, "budget")

    def test_canonical_policy_remains_an_exact_reversible_comparator(self) -> None:
        message = "Show me some Shoes; I'm open to options."
        canonical = apply_user_message(
            IntentState(),
            message,
            1,
            policy=CANONICAL_INTENT_POLICY,
        )
        robust = apply_user_message(
            IntentState(),
            message,
            1,
            policy=ROBUST_INTENT_POLICY,
        )

        self.assertIsNone(canonical.category)
        self.assertEqual(robust.category, "Shoes")
        self.assertNotEqual(canonical, robust)

    def test_dense_renderer_uses_frozen_section_and_attribute_order(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        state = self._answer(state, "feature", "waterproof trail grip")
        state = self._answer(state, "brand", "Alpine Works", turn=3)
        state = self._answer(state, "material", "leather", turn=4)
        state = self._answer(state, "color", "red", turn=5)
        state = self._answer(state, "size", "10 Wide", turn=6)
        state = self._answer(state, "style", "minimalist", turn=7)
        state = self._answer(state, "budget", "under $100", turn=8)

        self.assertEqual(
            render_dense_query(state).splitlines(),
            [
                "Category: Shoes",
                "Search Clues: waterproof trail grip",
                "Brand: Alpine Works",
                "Attributes: Material: leather | Color: red | Size: 10 Wide | Style: minimalist",
                "Price: under $100",
            ],
        )

    def test_updates_are_immutable_and_can_branch_from_one_state(self) -> None:
        base = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        brand_question = record_question(base, "brand")
        material_question = record_question(base, "material")
        brand_branch = apply_user_message(
            brand_question,
            "For that, what matters is: Alpine Works.",
            2,
        )
        material_branch = apply_user_message(
            material_question,
            "For that, what matters is: leather.",
            2,
        )

        self.assertIsNot(base, brand_question)
        self.assertIsNot(base, material_question)
        self.assertIsNot(brand_question, brand_branch)
        self.assertEqual(render_dense_query(base), "Category: Shoes")
        self.assertEqual(render_dense_query(brand_question), "Category: Shoes")
        self.assertEqual(render_dense_query(material_question), "Category: Shoes")
        self.assertIn("Brand: Alpine Works", render_dense_query(brand_branch))
        self.assertNotIn("leather", render_dense_query(brand_branch))
        self.assertIn("Material: leather", render_dense_query(material_branch))
        self.assertNotIn("Alpine Works", render_dense_query(material_branch))


if __name__ == "__main__":
    unittest.main()
