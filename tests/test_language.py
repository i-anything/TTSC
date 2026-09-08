from __future__ import annotations

import unittest

from conversational_search.decision import parse_protocol_event, recognize_protocol_observation
from conversational_search.language import MAX_MESSAGE_CHARACTERS, normalize_dialogue_envelope
from conversational_search.protocol import ProtocolEventKind


class DialogueEnvelopeTest(unittest.TestCase):
    def test_composed_requests_preserve_the_disclosed_value(self) -> None:
        for query in ("I am shopping for", "Please help me find", "I'd like to buy", "I want", "Find me"):
            for requirement in ("My main requirement is:", "It must satisfy this requirement:", "It must have", "Essential:"):
                for value in ("cotton", "ACME X.2; 7.5-inch, waterproof", "color: Blue"):
                    with self.subTest(query=query, requirement=requirement, value=value):
                        message = f"{query} Shoes. {requirement} {value}."
                        self.assertEqual(
                            normalize_dialogue_envelope(message, 1, None),
                            f"I'm looking for Shoes. A key requirement is: {value}.",
                        )

    def test_browsing_and_tentative_are_distinct(self) -> None:
        self.assertEqual(normalize_dialogue_envelope("Show me Shoes; just browsing for now.", 1, None),
                         "I'm looking for Shoes, but I'm still exploring.")
        self.assertEqual(normalize_dialogue_envelope("I need Shoes. Maybe warm.", 1, None),
                         "I'm looking for Shoes. warm")

    def test_answer_payload_is_not_split_or_reordered(self) -> None:
        expected = "For that, what matters is: X.2; soft; blue, 7.5 in.."
        for prefix in ("My priorities are:", "My requirements are:", "For that attribute, I prefer", "What matters to me is:"):
            with self.subTest(prefix=prefix):
                self.assertEqual(normalize_dialogue_envelope(f"{prefix} X.2; soft; blue, 7.5 in..", 2, "other"), expected)

    def test_boundary_decline_is_not_preference_exhaustion(self) -> None:
        cases = {
            "No particular preference regarding color; you can decide.": ProtocolEventKind.BOUNDARY_DECLINE,
            "I do not have a preference for color.": ProtocolEventKind.BOUNDARY_DECLINE,
            "No additional preference about color.": ProtocolEventKind.NO_ADDITIONAL,
            "Nothing else to add regarding color.": ProtocolEventKind.NO_ADDITIONAL,
        }
        for message, kind in cases.items():
            with self.subTest(message=message):
                normalized = normalize_dialogue_envelope(message, 2, "color")
                event = parse_protocol_event(normalized, recognize_protocol_observation(normalized, 2), 2, asked_attribute="color")
                self.assertEqual(event.kind, kind)

    def test_override_requires_explicit_replacement_intent(self) -> None:
        for prefix in ("I changed my mind. Please prioritize:", "Change of plan: Replace my earlier preference with", "Disregard my previous preference. I now need"):
            with self.subTest(prefix=prefix):
                self.assertEqual(normalize_dialogue_envelope(f"{prefix} waterproof.", 3, "other"),
                                 "Actually, ignore my earlier preference. What I need is: waterproof.")
        for message in ("Please prioritize waterproof.", "I changed my mind about the delivery date."):
            self.assertEqual(normalize_dialogue_envelope(message, 3, "other"), message)

    def test_missing_context_mismatched_attribute_and_arbitrary_prose_fail_open(self) -> None:
        cases = (
            ("My priorities are: cotton.", 2, None),
            ("Nothing else to add regarding color.", 2, "material"),
            ("No preference for shipping.", 2, "other"),
            ("I do not want cotton.", 2, "material"),
            ("I need Shoes. It is not a requirement: cotton.", 1, None),
            ("Someone said I need Shoes. It must be cotton.", 1, None),
            ("I need Shoes. It must be " + "x" * MAX_MESSAGE_CHARACTERS, 1, None),
            ("My priorities are: " + "x" * 363 + ".", 2, "other"),
        )
        for message, turn, attribute in cases:
            with self.subTest(message=message[:70]):
                self.assertEqual(normalize_dialogue_envelope(message, turn, attribute), message)

    def test_canonical_messages_are_byte_preserved_and_normalization_is_idempotent(self) -> None:
        cases = (
            ("I'm looking for Shoes. A key requirement is: cotton.", 1, None),
            ("I'm looking for Shoes. X.2", 1, None),
            ("For that, what matters is: cotton; Blue.", 2, "other"),
            ("I don't have an additional preference for other.", 3, "other"),
            ("Actually, ignore my earlier preference. What I need is: blue.", 4, "other"),
        )
        for message, turn, attribute in cases:
            with self.subTest(message=message):
                self.assertEqual(normalize_dialogue_envelope(message, turn, attribute), message)
        message = normalize_dialogue_envelope("I need Shoes. It must be cotton.", 1, None)
        self.assertEqual(normalize_dialogue_envelope(message, 1, None), message)


if __name__ == "__main__":
    unittest.main()
