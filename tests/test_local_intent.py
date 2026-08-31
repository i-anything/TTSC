from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conversational_search.intent import (
    IntentState,
    apply_user_message,
)
from conversational_search.local_intent import (
    LOCAL_INTENT_JSON_SCHEMA,
    LlamaCppStructuredIntentParser,
    LocalIntentTrigger,
    StructuredIntentParseResult,
    apply_structured_intent_delta,
    has_current_free_text_fallback,
    local_intent_trigger,
    parse_structured_intent_delta,
)


def _payload(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "category": None,
        "requirements": [],
        "exclusions": [],
        "clears": [],
        "full_override_source": None,
    }
    value.update(overrides)
    return value


class _FakeModel:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self.payload),
                    }
                }
            ],
            "usage": {"prompt_tokens": 37, "completion_tokens": 19},
        }


class LocalIntentTest(unittest.TestCase):
    def test_trigger_routes_complex_language_but_bypasses_official_templates(
        self,
    ) -> None:
        initial_message = (
            "I'm looking for Shoes. A key requirement is: leather."
        )
        initial = apply_user_message(IntentState(), initial_message, 1)
        self.assertIsNone(local_intent_trigger(initial, initial_message, 1))

        override_message = (
            "Actually, ignore my earlier preference. What I need is: cotton."
        )
        official_override = apply_user_message(initial, override_message, 2)
        self.assertIsNone(
            local_intent_trigger(official_override, override_message, 2)
        )

        complex_message = (
            "Change of plan: replace my earlier preference with blue and "
            "avoid acrylic."
        )
        complex_state = apply_user_message(initial, complex_message, 2)
        self.assertIs(
            local_intent_trigger(complex_state, complex_message, 2),
            LocalIntentTrigger.COMPLEX_LANGUAGE,
        )

        free_text_message = "Something packable for monsoon commutes."
        free_text_state = apply_user_message(initial, free_text_message, 2)
        self.assertIs(
            local_intent_trigger(free_text_state, free_text_message, 2),
            LocalIntentTrigger.FREE_TEXT,
        )

    def test_grounded_payload_is_validated_and_normalized(self) -> None:
        message = (
            "I need trail shoes under $120, breathable, and without wool."
        )
        delta = parse_structured_intent_delta(
            _payload(
                category={
                    "value": "trail shoes",
                    "source_text": "trail shoes",
                },
                requirements=[
                    {
                        "attribute": "budget",
                        "value": "under $120",
                        "source_text": "under $120",
                    },
                    {
                        "attribute": "feature",
                        "value": "breathable",
                        "source_text": "breathable",
                    },
                ],
                exclusions=[
                    {
                        "attribute": "material",
                        "value": "wool",
                        "source_text": "without wool",
                    }
                ],
            ),
            message,
        )

        self.assertEqual(delta.category.value, "trail shoes")
        self.assertEqual(
            tuple(item.attribute for item in delta.requirements),
            ("budget", "feature"),
        )
        self.assertEqual(delta.exclusions[0].value, "wool")

    def test_ungrounded_or_extra_model_output_is_rejected(self) -> None:
        message = "I need breathable trail shoes."
        ungrounded = _payload(
            requirements=[
                {
                    "attribute": "feature",
                    "value": "waterproof",
                    "source_text": "waterproof",
                }
            ]
        )
        with self.assertRaises(ValueError):
            parse_structured_intent_delta(ungrounded, message)

        extra = _payload()
        extra["recommendations"] = ["B000000001"]
        with self.assertRaises(ValueError):
            parse_structured_intent_delta(extra, message)

    def test_destructive_operations_require_explicit_grounded_cues(self) -> None:
        message = "Wool is fine for this sweater."
        with self.assertRaises(ValueError):
            parse_structured_intent_delta(
                _payload(
                    exclusions=[
                        {
                            "attribute": "material",
                            "value": "wool",
                            "source_text": "Wool",
                        }
                    ]
                ),
                message,
            )
        with self.assertRaises(ValueError):
            parse_structured_intent_delta(
                _payload(full_override_source="Wool is fine"),
                message,
            )

    def test_negative_source_is_deterministically_reclassified(self) -> None:
        message = "I want a rain jacket without wool."

        repaired = parse_structured_intent_delta(
            _payload(
                requirements=[
                    {
                        "attribute": "material",
                        "value": "wool",
                        "source_text": "without wool",
                    }
                ]
            ),
            message,
        )
        self.assertEqual(repaired.requirements, ())
        self.assertEqual(repaired.exclusions[0].value, "wool")

        with self.assertRaises(ValueError):
            parse_structured_intent_delta(_payload(), message)

    def test_structured_delta_replaces_only_current_free_text_fallback(self) -> None:
        prior = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
        )
        message = "Could you switch material to canvas and avoid wool?"
        fallback = apply_user_message(prior, message, 2)
        self.assertTrue(has_current_free_text_fallback(fallback, 2))
        delta = parse_structured_intent_delta(
            _payload(
                requirements=[
                    {
                        "attribute": "material",
                        "value": "canvas",
                        "source_text": "material to canvas",
                    }
                ],
                exclusions=[
                    {
                        "attribute": "material",
                        "value": "wool",
                        "source_text": "avoid wool",
                    }
                ],
                clears=[
                    {
                        "attribute": "material",
                        "source_text": "switch material to canvas",
                    }
                ],
            ),
            message,
        )

        updated = apply_structured_intent_delta(prior, fallback, delta, 2)

        self.assertEqual(updated.intent_version, prior.intent_version + 1)
        self.assertEqual(updated.excluded, ("wool",))
        self.assertEqual(
            tuple((item.attribute, item.value) for item in updated.requirements),
            (("material", "canvas"),),
        )
        self.assertFalse(has_current_free_text_fallback(updated, 2))

    def test_grounded_delta_can_correct_a_complex_partial_parse(self) -> None:
        prior = apply_user_message(
            IntentState(),
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
        )
        message = (
            "Change of plan: replace my earlier preference with cotton and "
            "avoid acrylic."
        )
        fallback = apply_user_message(prior, message, 2)
        self.assertFalse(has_current_free_text_fallback(fallback, 2))
        delta = parse_structured_intent_delta(
            _payload(
                requirements=[
                    {
                        "attribute": "material",
                        "value": "cotton",
                        "source_text": "cotton",
                    }
                ],
                exclusions=[
                    {
                        "attribute": "material",
                        "value": "acrylic",
                        "source_text": "avoid acrylic",
                    }
                ],
                full_override_source=(
                    "Change of plan: replace my earlier preference"
                ),
            ),
            message,
        )

        updated = apply_structured_intent_delta(prior, fallback, delta, 2)

        self.assertEqual(updated.excluded, ("acrylic",))
        self.assertEqual(
            tuple((item.attribute, item.value) for item in updated.requirements),
            (("material", "cotton"),),
        )
        self.assertEqual(updated.intent_version, prior.intent_version + 1)

    def test_llama_adapter_uses_schema_and_reports_usage(self) -> None:
        message = "I need breathable trail shoes."
        payload = _payload(
            category={
                "value": "trail shoes",
                "source_text": "trail shoes",
            },
            requirements=[
                {
                    "attribute": "feature",
                    "value": "breathable",
                    "source_text": "breathable",
                }
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            parser = LlamaCppStructuredIntentParser(model_path)
            fake = _FakeModel(payload)
            parser._model = fake

            result = parser.parse(IntentState(), message, 1)

        self.assertIsInstance(result, StructuredIntentParseResult)
        self.assertEqual(result.prompt_tokens, 37)
        self.assertEqual(result.completion_tokens, 19)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(
            fake.calls[0]["response_format"],
            {"type": "json_object", "schema": LOCAL_INTENT_JSON_SCHEMA},
        )
        self.assertEqual(fake.calls[0]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
