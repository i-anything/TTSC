from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError, fields
from typing import cast

from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    DEFAULT_PROFILE_RESIDUAL_WEIGHT,
    DISABLED_PROFILE_POLICY,
    EMPTY_PROFILE_PRIOR,
    MAX_PROFILE_TAG_CHARACTERS,
    MAX_PROFILE_TAGS,
    NEUTRAL_PROFILE_PRIOR,
    PROFILE_THEME_MASK_BYTES,
    ProductTheme,
    ProfilePolicy,
    ProfilePrior,
    parse_profile_prior,
)


def _profile(*tags: object, **ignored: object) -> dict[str, object]:
    return {"preference_tags": list(tags), **ignored}


class ProfilePriorParsingTests(unittest.TestCase):
    def test_every_declared_theme_accepts_its_bare_canonical_name(self) -> None:
        for theme in ProductTheme:
            if theme is ProductTheme.NONE:
                continue
            with self.subTest(theme=theme.name):
                prior = parse_profile_prior(_profile(theme.name))
                self.assertEqual(prior.theme_mask, theme)

    def test_fixed_whole_tag_aliases_map_to_expected_themes(self) -> None:
        aliases = {
            ProductTheme.COMFORT: (
                "comfort",
                "comfortable",
                "comfort focused",
                "comfort first",
            ),
            ProductTheme.DURABILITY: (
                "durability",
                "durable",
                "long lasting",
                "hard wearing",
                "rugged",
            ),
            ProductTheme.PERFORMANCE: (
                "performance",
                "high performance",
                "performance focused",
                "technical performance",
            ),
            ProductTheme.WARMTH: ("warmth", "warm", "insulated", "thermal"),
            ProductTheme.WEATHER_PROTECTION: (
                "weather protection",
                "weather resistant",
                "weatherproof",
                "water resistant",
                "waterproof",
                "wind resistant",
                "windproof",
                "rain protection",
            ),
            ProductTheme.LIGHTWEIGHT: (
                "lightweight",
                "light weight",
                "ultralight",
                "featherweight",
            ),
            ProductTheme.BREATHABILITY: (
                "breathability",
                "breathable",
                "ventilated",
                "airflow",
            ),
            ProductTheme.EASY_CARE: (
                "easy care",
                "low maintenance",
                "machine washable",
                "washable",
            ),
            ProductTheme.VERSATILITY: (
                "versatility",
                "versatile",
                "multi purpose",
                "multipurpose",
                "all purpose",
            ),
            ProductTheme.SUSTAINABILITY: (
                "sustainability",
                "sustainable",
                "eco friendly",
                "environmentally friendly",
                "recycled",
            ),
        }
        for expected, tags in aliases.items():
            for tag in tags:
                with self.subTest(tag=tag):
                    self.assertEqual(
                        parse_profile_prior(_profile(tag)).theme_mask,
                        expected,
                    )

    def test_normalization_is_exact_and_deterministic(self) -> None:
        cases = (
            ("  COMFORTABLE\t", ProductTheme.COMFORT),
            ("LONG---LASTING", ProductTheme.DURABILITY),
            ("high___performance", ProductTheme.PERFORMANCE),
            ("weather_ protection", ProductTheme.WEATHER_PROTECTION),
            ("LIGHT---WEIGHT", ProductTheme.LIGHTWEIGHT),
            ("easy_care", ProductTheme.EASY_CARE),
            ("eco-friendly", ProductTheme.SUSTAINABILITY),
            ("CÖMFORT", ProductTheme.COMFORT),
            ("ſustainable", ProductTheme.SUSTAINABILITY),
            ("\u00a0comfort\u00a0", ProductTheme.COMFORT),
            ("machine...washable", ProductTheme.EASY_CARE),
        )
        for tag, expected in cases:
            with self.subTest(tag=tag):
                first = parse_profile_prior(_profile(tag))
                second = parse_profile_prior(_profile(tag))
                self.assertEqual(first.theme_mask, expected)
                self.assertEqual(first, second)
                self.assertEqual(first.ranking_digest, second.ranking_digest)

    def test_multiple_tags_form_a_deduplicated_bitmask(self) -> None:
        first = parse_profile_prior(
            _profile("comfortable", "durable", "comfort", "breathable")
        )
        second = parse_profile_prior(
            _profile("BREATHABILITY", "DURABILITY", "COMFORT")
        )
        expected = (
            ProductTheme.COMFORT
            | ProductTheme.DURABILITY
            | ProductTheme.BREATHABILITY
        )

        self.assertEqual(first.theme_mask, expected)
        self.assertEqual(first.active_theme_count, 3)
        self.assertEqual(first, second)
        self.assertEqual(first.ranking_digest, second.ranking_digest)

    def test_dimension_only_and_prefixed_tags_are_always_neutral(self) -> None:
        dimensions = (
            "material",
            "color",
            "size",
            "style",
            "fit",
            "brand",
            "category",
            "budget",
            "price",
            "feature",
            "features",
            "use case",
            "use_case",
            "use-case",
        )
        prefixes = (
            "material: breathable",
            "color=warm",
            "size/comfortable",
            "style versatile",
            "fit-durable",
            "brand_sustainable",
            "category: lightweight",
            "budget value",
            "price: low",
            "feature waterproof",
            "features: comfort",
            "use_case: performance",
        )
        for tag in (*dimensions, *prefixes):
            with self.subTest(tag=tag):
                self.assertIs(parse_profile_prior(_profile(tag)), NEUTRAL_PROFILE_PRIOR)

    def test_dimension_tag_is_discarded_without_erasing_valid_evidence(self) -> None:
        prior = parse_profile_prior(_profile("comfort", "material: breathable"))
        self.assertEqual(prior.theme_mask, ProductTheme.COMFORT)

    def test_unknown_or_non_whole_tags_are_discarded(self) -> None:
        tags = (
            "unknown",
            "maximum comfort",
            "comfortable shoes",
            "not durable",
            "performance oriented",
            "eco friendly materials",
            "premium",
            "value",
            "portable",
            "technical",
            "rain resistant",
            "rainproof",
            "",
            "   ",
        )
        for tag in tags:
            with self.subTest(tag=tag):
                self.assertEqual(
                    parse_profile_prior(_profile("comfort", tag)).theme_mask,
                    ProductTheme.COMFORT,
                )
                self.assertIs(parse_profile_prior(_profile(tag)), NEUTRAL_PROFILE_PRIOR)

    def test_missing_empty_or_malformed_tag_container_is_neutral(self) -> None:
        malformed_profiles = (
            None,
            [],
            (),
            "profile",
            {"summary": "comfortable"},
            {"preference_tags": None},
            {"preference_tags": "comfort"},
            {"preference_tags": ("comfort",)},
            {"preference_tags": {"comfort"}},
            {"preference_tags": {}},
            {"preference_tags": []},
        )
        for profile in malformed_profiles:
            with self.subTest(profile_type=type(profile).__name__):
                self.assertIs(parse_profile_prior(profile), NEUTRAL_PROFILE_PRIOR)

    def test_malformed_tag_values_are_discarded(self) -> None:
        malformed_tags = (None, True, 1, 1.0, b"comfort", [], {}, object())
        for tag in malformed_tags:
            with self.subTest(tag_type=type(tag).__name__):
                self.assertEqual(
                    parse_profile_prior(_profile("comfort", tag)).theme_mask,
                    ProductTheme.COMFORT,
                )
                self.assertIs(parse_profile_prior(_profile(tag)), NEUTRAL_PROFILE_PRIOR)

    def test_only_the_first_bounded_number_of_tags_is_inspected(self) -> None:
        at_limit = parse_profile_prior(_profile(*(["comfort"] * MAX_PROFILE_TAGS)))
        recognized_too_late = parse_profile_prior(
            _profile(*(["unknown"] * MAX_PROFILE_TAGS), "comfort")
        )
        ignored_tail = parse_profile_prior(
            _profile("comfort", *(["unknown"] * MAX_PROFILE_TAGS))
        )

        self.assertEqual(at_limit.theme_mask, ProductTheme.COMFORT)
        self.assertIs(recognized_too_late, NEUTRAL_PROFILE_PRIOR)
        self.assertEqual(ignored_tail.theme_mask, ProductTheme.COMFORT)

    def test_overlength_tag_is_discarded_at_the_character_bound(self) -> None:
        at_limit_tag = "comfort" + " " * (
            MAX_PROFILE_TAG_CHARACTERS - len("comfort")
        )
        over_limit_tag = at_limit_tag + " "

        self.assertEqual(
            parse_profile_prior(_profile(at_limit_tag)).theme_mask,
            ProductTheme.COMFORT,
        )
        self.assertIs(
            parse_profile_prior(_profile(over_limit_tag)),
            NEUTRAL_PROFILE_PRIOR,
        )
        self.assertEqual(
            parse_profile_prior(_profile("durable", over_limit_tag)).theme_mask,
            ProductTheme.DURABILITY,
        )

    def test_only_preference_tags_is_observed(self) -> None:
        class Poison:
            def __getattribute__(self, name: str) -> object:
                raise AssertionError(f"ignored profile field was read: {name}")

            def __repr__(self) -> str:
                raise AssertionError("ignored profile field was rendered")

        baseline = parse_profile_prior(_profile("comfort"))
        with_ignored_fields = parse_profile_prior(
            _profile(
                "comfort",
                summary=Poison(),
                purchase_frequency=Poison(),
                average_prior_rating=Poison(),
                rating_style=Poison(),
                arbitrary_extra_field=Poison(),
            )
        )

        self.assertEqual(with_ignored_fields, baseline)

    def test_parser_does_not_mutate_the_caller_profile(self) -> None:
        tags = ["comfort", "durability"]
        profile = {
            "preference_tags": tags,
            "summary": {"nested": ["untouched"]},
        }
        original_tags = tags.copy()
        original_summary = profile["summary"]

        parse_profile_prior(profile)

        self.assertEqual(tags, original_tags)
        self.assertIs(profile["preference_tags"], tags)
        self.assertIs(profile["summary"], original_summary)


class ProfilePriorValueTests(unittest.TestCase):
    def test_prior_is_immutable_slotted_and_retains_no_strings(self) -> None:
        prior = parse_profile_prior(_profile("eco-friendly"))

        self.assertFalse(hasattr(prior, "__dict__"))
        self.assertEqual(tuple(field.name for field in fields(prior)), ("theme_mask",))
        self.assertTrue(
            all(
                not isinstance(getattr(prior, field.name), str)
                for field in fields(prior)
            )
        )
        self.assertNotIn("eco-friendly", repr(prior))
        with self.assertRaises(FrozenInstanceError):
            prior.theme_mask = ProductTheme.NONE  # type: ignore[misc]

    def test_prior_digest_is_exactly_32_bytes_and_mask_deterministic(self) -> None:
        comfort = ProfilePrior(ProductTheme.COMFORT)
        alias = parse_profile_prior(_profile("comfortable"))
        reordered = parse_profile_prior(_profile("comfort", "comfort"))

        self.assertIsInstance(comfort.ranking_digest, bytes)
        self.assertEqual(len(comfort.ranking_digest), 32)
        self.assertEqual(comfort.ranking_digest, alias.ranking_digest)
        self.assertEqual(comfort.ranking_digest, reordered.ranking_digest)
        self.assertEqual(
            comfort.ranking_digest.hex(),
            "80e7f7f76c93e5cdfffec32659c6d5289f56522ffe090b8b9ced43e44d4d053d",
        )
        self.assertEqual(
            NEUTRAL_PROFILE_PRIOR.ranking_digest.hex(),
            "5b5166ce1e4cd153b8a26244150eb9e32ff859cc57fe0237fa0ec5b3e0e32439",
        )

    def test_different_theme_masks_have_different_digests(self) -> None:
        digests = {
            ProfilePrior(theme).ranking_digest
            for theme in ProductTheme
            if theme is not ProductTheme.NONE
        }
        self.assertEqual(len(digests), len(ProductTheme))
        self.assertNotIn(NEUTRAL_PROFILE_PRIOR.ranking_digest, digests)

    def test_neutral_prior_aliases_are_the_same_singleton(self) -> None:
        self.assertIs(EMPTY_PROFILE_PRIOR, NEUTRAL_PROFILE_PRIOR)
        self.assertEqual(PROFILE_THEME_MASK_BYTES, 2)
        self.assertTrue(NEUTRAL_PROFILE_PRIOR.is_neutral)
        self.assertEqual(NEUTRAL_PROFILE_PRIOR.active_theme_count, 0)

    def test_constructor_rejects_raw_int_boolean_and_unknown_bits(self) -> None:
        invalid_masks = (
            cast(ProductTheme, 1),
            cast(ProductTheme, True),
            ProductTheme(1 << 10),
            ProductTheme.COMFORT | ProductTheme(1 << 15),
        )
        for mask in invalid_masks:
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    ProfilePrior(mask)


class ProfilePolicyTests(unittest.TestCase):
    def test_policy_constants_are_exact_and_reversible(self) -> None:
        self.assertIs(DISABLED_PROFILE_POLICY, ProfilePolicy.DISABLED)
        self.assertIs(
            BOUNDED_RESIDUAL_PROFILE_POLICY,
            ProfilePolicy.BOUNDED_RESIDUAL,
        )
        self.assertEqual(
            DISABLED_PROFILE_POLICY.value,
            "phase7-profile-disabled-v1",
        )
        self.assertEqual(
            BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            "phase9-bounded-profile-residual-v1",
        )
        with self.assertRaises(AttributeError):
            DISABLED_PROFILE_POLICY.value = "changed"  # type: ignore[misc]

    def test_default_candidate_weight_is_the_frozen_bounded_value(self) -> None:
        self.assertEqual(DEFAULT_PROFILE_RESIDUAL_WEIGHT, 0.05)
        self.assertTrue(math.isfinite(DEFAULT_PROFILE_RESIDUAL_WEIGHT))
        self.assertGreater(DEFAULT_PROFILE_RESIDUAL_WEIGHT, 0.0)
        self.assertLessEqual(DEFAULT_PROFILE_RESIDUAL_WEIGHT, 0.05)

    def test_disabled_policy_always_returns_zero(self) -> None:
        prior = ProfilePrior(
            ProductTheme.COMFORT
            | ProductTheme.PERFORMANCE
            | ProductTheme.SUSTAINABILITY
        )
        for numeric_mask in range(1 << len(ProductTheme)):
            candidate = ProductTheme(numeric_mask)
            with self.subTest(candidate=numeric_mask):
                self.assertEqual(
                    DISABLED_PROFILE_POLICY.residual(prior, candidate),
                    0.0,
                )

    def test_bounded_residual_is_overlap_over_active_prior_themes(self) -> None:
        prior = ProfilePrior(
            ProductTheme.COMFORT
            | ProductTheme.DURABILITY
            | ProductTheme.BREATHABILITY
            | ProductTheme.SUSTAINABILITY
        )
        for numeric_mask in range(1 << len(ProductTheme)):
            candidate = ProductTheme(numeric_mask)
            expected = int(prior.theme_mask & candidate).bit_count() / 4
            actual = BOUNDED_RESIDUAL_PROFILE_POLICY.residual(prior, candidate)
            with self.subTest(candidate=numeric_mask):
                self.assertEqual(actual, expected)
                self.assertTrue(math.isfinite(actual))
                self.assertGreaterEqual(actual, 0.0)
                self.assertLessEqual(actual, 1.0)

    def test_residual_is_neutral_for_empty_or_malformed_evidence(self) -> None:
        active = ProfilePrior(ProductTheme.COMFORT)
        malformed_prior = cast(ProfilePrior, object())
        malformed_candidates = (
            cast(ProductTheme, 1),
            cast(ProductTheme, True),
            ProductTheme(1 << 10),
            ProductTheme.COMFORT | ProductTheme(1 << 15),
        )

        self.assertEqual(
            BOUNDED_RESIDUAL_PROFILE_POLICY.residual(
                NEUTRAL_PROFILE_PRIOR,
                ProductTheme.COMFORT,
            ),
            0.0,
        )
        self.assertEqual(
            BOUNDED_RESIDUAL_PROFILE_POLICY.residual(
                malformed_prior,
                ProductTheme.COMFORT,
            ),
            0.0,
        )
        for candidate in malformed_candidates:
            with self.subTest(candidate=candidate):
                self.assertEqual(
                    BOUNDED_RESIDUAL_PROFILE_POLICY.residual(active, candidate),
                    0.0,
                )

    def test_policy_digest_is_32_bytes_deterministic_and_policy_scoped(self) -> None:
        comfort = ProfilePrior(ProductTheme.COMFORT)
        durable = ProfilePrior(ProductTheme.DURABILITY)
        disabled_comfort = DISABLED_PROFILE_POLICY.ranking_digest(comfort)
        disabled_durable = DISABLED_PROFILE_POLICY.ranking_digest(durable)
        bounded_comfort = BOUNDED_RESIDUAL_PROFILE_POLICY.ranking_digest(comfort)
        bounded_durable = BOUNDED_RESIDUAL_PROFILE_POLICY.ranking_digest(durable)

        self.assertEqual(disabled_comfort, disabled_durable)
        self.assertEqual(len(disabled_comfort), 32)
        self.assertEqual(len(bounded_comfort), 32)
        self.assertNotEqual(disabled_comfort, bounded_comfort)
        self.assertNotEqual(bounded_comfort, bounded_durable)
        self.assertEqual(bounded_comfort, comfort.ranking_digest)
        self.assertEqual(
            disabled_comfort.hex(),
            "5dab9e5c2e388827c05a435b5d447a1f86913faceb3804bbb1aaf48d4c5b0870",
        )

    def test_policy_digest_fails_malformed_prior_to_neutral(self) -> None:
        malformed = cast(ProfilePrior, object())
        self.assertEqual(
            DISABLED_PROFILE_POLICY.ranking_digest(malformed),
            DISABLED_PROFILE_POLICY.ranking_digest(NEUTRAL_PROFILE_PRIOR),
        )
        self.assertEqual(
            BOUNDED_RESIDUAL_PROFILE_POLICY.ranking_digest(malformed),
            BOUNDED_RESIDUAL_PROFILE_POLICY.ranking_digest(NEUTRAL_PROFILE_PRIOR),
        )


if __name__ == "__main__":
    unittest.main()
