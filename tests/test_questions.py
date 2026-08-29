from __future__ import annotations

import unittest

from conversational_search.intent import (
    IntentState,
    apply_user_message,
    record_question,
)
from conversational_search.questions import (
    CONSERVATIVE_EARLY_OTHER_POLICY,
    PHASE1_QUESTION_POLICY,
    QuestionPolicy,
)


class QuestionPolicyTest(unittest.TestCase):
    def test_phase1_policy_retains_frozen_order(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        self.assertEqual(PHASE1_QUESTION_POLICY.choose(state), "feature")
        state = record_question(state, "feature")
        state = apply_user_message(
            state,
            "I don't have an additional preference for feature.",
            2,
        )
        self.assertEqual(PHASE1_QUESTION_POLICY.choose(state), "material")

    def test_conservative_policy_moves_other_before_low_yield_fields(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        for turn, attribute in enumerate(("feature", "material", "color"), start=2):
            self.assertEqual(
                CONSERVATIVE_EARLY_OTHER_POLICY.choose(state),
                attribute,
            )
            state = record_question(state, attribute)
            state = apply_user_message(
                state,
                f"I don't have an additional preference for {attribute}.",
                turn,
            )
        self.assertEqual(CONSERVATIVE_EARLY_OTHER_POLICY.choose(state), "other")

    def test_conservative_policy_requeues_question_interrupted_by_override(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Accessories Belts. Buckle closure",
            1,
        )
        self.assertEqual(CONSERVATIVE_EARLY_OTHER_POLICY.choose(state), "material")
        state = record_question(state, "material")
        state = apply_user_message(
            state,
            "Actually, ignore my earlier preference. What I need is: color: red.",
            3,
        )

        self.assertEqual(CONSERVATIVE_EARLY_OTHER_POLICY.choose(state), "material")
        self.assertNotEqual(PHASE1_QUESTION_POLICY.choose(state), "material")

    def test_resolved_pending_attribute_is_not_requeued(self) -> None:
        state = apply_user_message(
            IntentState(),
            "I'm looking for Shoes, but I'm still exploring.",
            1,
        )
        state = record_question(state, "feature")
        state = apply_user_message(
            state,
            "For that, what matters is: waterproof.",
            2,
        )
        self.assertEqual(CONSERVATIVE_EARLY_OTHER_POLICY.choose(state), "material")

    def test_invalid_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QuestionPolicy(name="duplicate", priority=("feature", "feature"))
        with self.assertRaises(ValueError):
            QuestionPolicy(name="unsupported", priority=("brand",))


if __name__ == "__main__":
    unittest.main()
