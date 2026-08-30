from __future__ import annotations

import gc
import hashlib
import unittest
import weakref
from dataclasses import fields
from typing import cast
from unittest import mock

from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    DISABLED_PROFILE_POLICY,
    ProductTheme,
    ProfilePolicy,
    ProfilePrior,
)
from conversational_search.ranking import FUSED_ONLY_RANKING_POLICY
from conversational_search.service import ConversationalSearchAgent
from tests.test_service import CacheableRecordingRetriever, RecordingRetriever


_BROWSING_MESSAGE = "I'm looking for Shoes, but I'm still exploring."
_BUYING_MESSAGE = "I'm looking for Shoes. A key requirement is: leather."
_NO_FEATURE_PREFERENCE = (
    "I don't have a preference for feature; please use your judgment."
)
_IDS = ("B000000001", "B000000002")


def _agent(
    retriever: object,
    *,
    profile_policy: ProfilePolicy = BOUNDED_RESIDUAL_PROFILE_POLICY,
    ranking_policy: object = None,
) -> ConversationalSearchAgent:
    kwargs: dict[str, object] = {
        "retriever": retriever,
        "profile_policy": profile_policy,
    }
    if ranking_policy is not None:
        kwargs["ranking_policy"] = ranking_policy
    return ConversationalSearchAgent("unused.jsonl", **kwargs)


class ServiceProfileRankingTests(unittest.TestCase):
    def test_default_bounded_policy_changes_only_stage_a_ordering(self) -> None:
        documents = {
            _IDS[0]: "ordinary product",
            _IDS[1]: "comfortable product",
        }
        bounded_retriever = RecordingRetriever(
            _IDS,
            fused_ids=_IDS,
            documents=documents,
        )
        disabled_retriever = RecordingRetriever(
            _IDS,
            fused_ids=_IDS,
            documents=documents,
        )
        bounded = _agent(bounded_retriever)
        disabled = _agent(
            disabled_retriever,
            profile_policy=DISABLED_PROFILE_POLICY,
        )
        profile = {"preference_tags": ["comfort"]}
        bounded.reset("bounded", profile)
        disabled.reset("disabled", profile)

        bounded_response = bounded.respond("bounded", _BROWSING_MESSAGE, 1, 2)
        disabled_response = disabled.respond("disabled", _BROWSING_MESSAGE, 1, 2)

        self.assertEqual(
            bounded_response["recommendations"],
            [
                {"parent_asin": _IDS[1]},
                {"parent_asin": _IDS[0]},
            ],
        )
        self.assertEqual(
            disabled_response["recommendations"],
            [
                {"parent_asin": _IDS[0]},
                {"parent_asin": _IDS[1]},
            ],
        )
        for field in ("message", "ask_attribute", "usage"):
            self.assertEqual(bounded_response[field], disabled_response[field])
        self.assertEqual(bounded_retriever.calls, disabled_retriever.calls)
        self.assertEqual(
            bounded_retriever.document_calls,
            disabled_retriever.document_calls,
        )
        self.assertEqual(
            bounded.profile_health["successful_residual_applications"],
            1,
        )
        self.assertEqual(disabled.profile_health["eligible_stage_a_attempts"], 0)

    def test_any_active_requirement_has_absolute_precedence(self) -> None:
        documents = {
            _IDS[0]: "leather shoe",
            _IDS[1]: "comfortable shoe",
        }
        recognized = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents)
        )
        neutral = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents)
        )
        disabled = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents),
            profile_policy=DISABLED_PROFILE_POLICY,
        )
        recognized.reset("recognized", {"preference_tags": ["comfort"]})
        neutral.reset("neutral", {"preference_tags": ["unknown"]})
        disabled.reset("disabled", {"preference_tags": ["comfort"]})

        recognized_response = recognized.respond(
            "recognized",
            _BUYING_MESSAGE,
            1,
            2,
        )
        neutral.respond("neutral", _BUYING_MESSAGE, 1, 2)
        disabled.respond("disabled", _BUYING_MESSAGE, 1, 2)

        self.assertEqual(
            recognized_response["recommendations"],
            [
                {"parent_asin": _IDS[0]},
                {"parent_asin": _IDS[1]},
            ],
        )
        self.assertEqual(
            recognized.profile_health["turns_disabled_by_active_requirements"],
            1,
        )
        self.assertEqual(recognized.profile_health["eligible_stage_a_attempts"], 0)
        self.assertEqual(
            neutral.profile_health["turns_disabled_by_active_requirements"],
            0,
        )
        self.assertEqual(
            disabled.profile_health["turns_disabled_by_active_requirements"],
            0,
        )

    def test_profile_neutral_fallback_classes_keep_exact_base_order(self) -> None:
        cases = (
            (
                "no represented theme",
                {
                    _IDS[0]: "ordinary product",
                    _IDS[1]: "another ordinary product",
                },
                "empty_represented_theme_fallbacks",
            ),
            (
                "constant profile score",
                {
                    _IDS[0]: "comfortable product",
                    _IDS[1]: "another comfortable product",
                },
                "constant_score_neutral_fallbacks",
            ),
        )
        for label, documents, counter in cases:
            with self.subTest(label=label):
                agent = _agent(
                    RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents)
                )
                agent.reset("session", {"preference_tags": ["comfort"]})

                response = agent.respond("session", _BROWSING_MESSAGE, 1, 2)

                self.assertEqual(
                    response["recommendations"],
                    [
                        {"parent_asin": _IDS[0]},
                        {"parent_asin": _IDS[1]},
                    ],
                )
                self.assertEqual(agent.profile_health[counter], 1)
                self.assertEqual(
                    agent.profile_health["eligible_stage_a_attempts"],
                    1,
                )
                self.assertEqual(
                    agent.profile_health["successful_residual_applications"],
                    0,
                )

    def test_profile_scoring_fault_returns_phase7_ranking(self) -> None:
        documents = {
            _IDS[0]: "ordinary product",
            _IDS[1]: "comfortable product",
        }
        baseline = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents),
            profile_policy=DISABLED_PROFILE_POLICY,
        )
        candidate = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents)
        )
        profile = {"preference_tags": ["comfort"]}
        baseline.reset("baseline", profile)
        candidate.reset("candidate", profile)
        baseline_response = baseline.respond("baseline", _BROWSING_MESSAGE, 1, 2)

        with mock.patch.object(
            ProfilePolicy,
            "residual",
            side_effect=RuntimeError("synthetic profile fault"),
        ):
            candidate_response = candidate.respond(
                "candidate",
                _BROWSING_MESSAGE,
                1,
                2,
            )

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(candidate.profile_health["eligible_stage_a_attempts"], 1)
        self.assertEqual(
            candidate.profile_health["parsing_or_scoring_fallbacks"],
            1,
        )
        self.assertEqual(candidate.ranking_health["successes"], 1)
        self.assertEqual(candidate.ranking_health["failures"], 0)

    def test_unexpected_profile_wrapper_fault_recomputes_exact_phase7_stage_a(
        self,
    ) -> None:
        documents = {
            _IDS[0]: "ordinary product",
            _IDS[1]: "comfortable product",
        }
        baseline = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents),
            profile_policy=DISABLED_PROFILE_POLICY,
        )
        candidate = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents)
        )
        profile = {"preference_tags": ["comfort"]}
        baseline.reset("baseline", profile)
        candidate.reset("candidate", profile)
        baseline_response = baseline.respond("baseline", _BROWSING_MESSAGE, 1, 2)

        with mock.patch(
            "conversational_search.service.rerank_stage_a_with_profile",
            side_effect=RuntimeError("synthetic wrapper fault"),
        ):
            candidate_response = candidate.respond(
                "candidate",
                _BROWSING_MESSAGE,
                1,
                2,
            )

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(candidate.ranking_health["successes"], 1)
        self.assertEqual(candidate.ranking_health["failures"], 0)
        self.assertEqual(
            candidate.profile_health["parsing_or_scoring_fallbacks"],
            1,
        )

    def test_ineligible_profile_wrapper_fault_also_recovers_exact_phase7(
        self,
    ) -> None:
        documents = {
            _IDS[0]: "ordinary product",
            _IDS[1]: "comfortable product",
        }
        baseline = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents),
            profile_policy=DISABLED_PROFILE_POLICY,
        )
        candidate = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS, documents=documents)
        )
        baseline.reset("baseline", {"preference_tags": ["unknown"]})
        candidate.reset("candidate", {"preference_tags": ["unknown"]})
        baseline_response = baseline.respond("baseline", _BROWSING_MESSAGE, 1, 2)

        with mock.patch(
            "conversational_search.service.rerank_stage_a_with_profile",
            side_effect=RuntimeError("synthetic neutral-wrapper fault"),
        ):
            candidate_response = candidate.respond(
                "candidate",
                _BROWSING_MESSAGE,
                1,
                2,
            )

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(candidate.profile_health["eligible_stage_a_attempts"], 0)
        self.assertEqual(
            candidate.profile_health["parsing_or_scoring_fallbacks"],
            1,
        )
        self.assertEqual(candidate.ranking_health["successes"], 1)
        self.assertEqual(candidate.ranking_health["failures"], 0)

    def test_fused_only_policy_never_attempts_profile_stage_a(self) -> None:
        retriever = RecordingRetriever(
            _IDS,
            fused_ids=_IDS,
            documents={_IDS[1]: "comfortable product"},
            document_fail=True,
        )
        agent = _agent(
            retriever,
            ranking_policy=FUSED_ONLY_RANKING_POLICY,
        )
        agent.reset("session", {"preference_tags": ["comfort"]})

        response = agent.respond("session", _BROWSING_MESSAGE, 1, 2)

        self.assertEqual(retriever.document_calls, [])
        self.assertEqual(agent.profile_health["eligible_stage_a_attempts"], 0)
        self.assertEqual(
            response["recommendations"],
            [
                {"parent_asin": _IDS[0]},
                {"parent_asin": _IDS[1]},
            ],
        )


class ServiceProfileStateAndCacheTests(unittest.TestCase):
    def test_reset_retains_only_hashed_key_and_bounded_prior(self) -> None:
        class ProfileMarker:
            pass

        marker = ProfileMarker()
        marker_reference = weakref.ref(marker)
        raw_session_id = "raw-session-id-must-not-enter-profile-store"
        raw_summary = "raw-summary-must-not-be-retained"
        raw_profile = {
            "preference_tags": ["comfort", "durable"],
            "summary": marker,
            "rating_style": raw_summary,
        }
        agent = _agent(RecordingRetriever(_IDS, fused_ids=_IDS))

        agent.reset(raw_session_id, raw_profile)
        del raw_profile
        del marker
        gc.collect()

        self.assertIsNone(marker_reference())
        self.assertEqual(
            tuple(agent._profile_priors),  # type: ignore[attr-defined]
            (hashlib.sha256(raw_session_id.encode("utf-8")).digest(),),
        )
        retained_prior = next(
            iter(agent._profile_priors.values())  # type: ignore[attr-defined]
        )
        self.assertEqual(
            tuple(field.name for field in fields(retained_prior)),
            ("theme_mask",),
        )
        retained_profile_repr = repr(
            agent._profile_priors  # type: ignore[attr-defined]
        )
        self.assertNotIn(raw_session_id, retained_profile_repr)
        self.assertNotIn(raw_summary, retained_profile_repr)

        agent.reset(raw_session_id, {"preference_tags": "malformed"})
        health = agent.profile_health
        self.assertEqual(health["session_entries"], 1)
        self.assertEqual(health["logical_profile_bytes"], 2)
        self.assertEqual(health["profiles_reset"], 2)
        self.assertEqual(health["zero_mask_profiles"], 1)
        self.assertEqual(health["nonzero_mask_profiles"], 1)
        self.assertEqual(health["recognized_theme_count"], 2)

    def test_parser_exception_is_a_neutral_session_safe_fallback(self) -> None:
        retriever = RecordingRetriever(_IDS, fused_ids=_IDS)
        agent = _agent(retriever)

        with mock.patch(
            "conversational_search.service.parse_profile_prior",
            side_effect=RuntimeError("synthetic parser fault"),
        ):
            agent.reset("session", {"preference_tags": ["comfort"]})
        response = agent.respond("session", _BROWSING_MESSAGE, 1, 2)

        self.assertEqual(
            response["recommendations"],
            [
                {"parent_asin": _IDS[0]},
                {"parent_asin": _IDS[1]},
            ],
        )
        self.assertEqual(agent.profile_health["zero_mask_profiles"], 1)
        self.assertEqual(
            agent.profile_health["parsing_or_scoring_fallbacks"],
            1,
        )

    def test_disabled_policy_never_reads_or_parses_user_profile(self) -> None:
        agent = _agent(
            RecordingRetriever(_IDS, fused_ids=_IDS),
            profile_policy=DISABLED_PROFILE_POLICY,
        )

        with mock.patch(
            "conversational_search.service.parse_profile_prior",
            side_effect=AssertionError("disabled policy touched the profile"),
        ) as parser, mock.patch(
            "conversational_search.service.rerank_stage_a_with_profile",
            side_effect=AssertionError("disabled policy used the profile path"),
        ) as profile_reranker:
            agent.reset(
                "session",
                {
                    "preference_tags": ["comfort"],
                    "summary": object(),
                },
            )
            response = agent.respond("session", _BROWSING_MESSAGE, 1, 2)

        parser.assert_not_called()
        profile_reranker.assert_not_called()
        self.assertEqual(
            response["recommendations"],
            [
                {"parent_asin": _IDS[0]},
                {"parent_asin": _IDS[1]},
            ],
        )
        self.assertEqual(agent.profile_health["profiles_reset"], 1)
        self.assertEqual(agent.profile_health["zero_mask_profiles"], 1)
        self.assertEqual(agent.profile_health["nonzero_mask_profiles"], 0)
        self.assertEqual(agent.profile_health["recognized_theme_count"], 0)
        self.assertEqual(
            agent.profile_health["parsing_or_scoring_fallbacks"],
            0,
        )

    def test_exact_policy_digest_is_passed_to_orchestration(self) -> None:
        retriever = RecordingRetriever(
            _IDS,
            fused_ids=_IDS,
            documents={_IDS[1]: "comfortable product"},
        )
        agent = _agent(retriever)
        prior = ProfilePrior(ProductTheme.COMFORT)
        agent.reset("session", {"preference_tags": ["comfort"]})

        with mock.patch.object(
            agent._orchestrator,  # type: ignore[attr-defined]
            "decide",
            wraps=agent._orchestrator.decide,  # type: ignore[attr-defined]
        ) as decide:
            agent.respond("session", _BROWSING_MESSAGE, 1, 2)

        self.assertEqual(
            decide.call_args.kwargs["profile_digest"],
            BOUNDED_RESIDUAL_PROFILE_POLICY.ranking_digest(prior),
        )

    def test_exact_reuse_performs_no_second_profile_cue_pass(self) -> None:
        retriever = CacheableRecordingRetriever(
            _IDS,
            fused_ids=_IDS,
            documents={
                _IDS[0]: "ordinary product",
                _IDS[1]: "comfortable product",
            },
        )
        agent = _agent(retriever)
        agent.reset("session", {"preference_tags": ["comfort"]})

        agent.respond("session", _BROWSING_MESSAGE, 1, 2)
        agent.respond("session", _NO_FEATURE_PREFERENCE, 2, 2)

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(retriever.document_calls), 1)
        self.assertEqual(agent.orchestration_health["hits"], 1)
        self.assertEqual(agent.profile_health["eligible_stage_a_attempts"], 1)
        self.assertEqual(
            agent.profile_health["successful_residual_applications"],
            1,
        )

    def test_transient_profile_scoring_fallback_is_never_cached(self) -> None:
        retriever = CacheableRecordingRetriever(
            _IDS,
            fused_ids=_IDS,
            documents={
                _IDS[0]: "ordinary product",
                _IDS[1]: "comfortable product",
            },
        )
        agent = _agent(retriever)
        agent.reset("session", {"preference_tags": ["comfort"]})

        with mock.patch.object(
            ProfilePolicy,
            "residual",
            side_effect=RuntimeError("transient profile fault"),
        ):
            first = agent.respond("session", _BROWSING_MESSAGE, 1, 2)
        recovered = agent.respond("session", _NO_FEATURE_PREFERENCE, 2, 2)

        self.assertEqual(
            first["recommendations"],
            [
                {"parent_asin": _IDS[0]},
                {"parent_asin": _IDS[1]},
            ],
        )
        self.assertEqual(
            recovered["recommendations"],
            [
                {"parent_asin": _IDS[1]},
                {"parent_asin": _IDS[0]},
            ],
        )
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(len(retriever.document_calls), 2)
        self.assertEqual(agent.orchestration_health["stores"], 1)
        self.assertEqual(agent.orchestration_health["hits"], 0)
        self.assertEqual(
            agent.profile_health["parsing_or_scoring_fallbacks"],
            1,
        )
        self.assertEqual(
            agent.profile_health["successful_residual_applications"],
            1,
        )

    def test_profile_health_has_only_fixed_aggregate_fields(self) -> None:
        agent = _agent(RecordingRetriever(_IDS, fused_ids=_IDS))
        agent.reset("private-session", {"preference_tags": ["comfort"]})

        health = agent.profile_health

        self.assertEqual(
            set(health),
            {
                "policy",
                "session_entries",
                "logical_profile_bytes",
                "profiles_reset",
                "zero_mask_profiles",
                "nonzero_mask_profiles",
                "recognized_theme_count",
                "turns_disabled_by_active_requirements",
                "eligible_stage_a_attempts",
                "empty_represented_theme_fallbacks",
                "constant_score_neutral_fallbacks",
                "successful_residual_applications",
                "parsing_or_scoring_fallbacks",
            },
        )
        self.assertEqual(
            health["policy"],
            "phase9-bounded-profile-residual-v1",
        )
        self.assertTrue(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for key, value in health.items()
                if key != "policy"
            )
        )
        self.assertNotIn("private-session", repr(health))
        self.assertNotIn("comfort", repr(health))
        self.assertEqual(
            set(agent.ranking_health),
            {"policy", "attempts", "successes", "failures", "unavailable_skips"},
        )

    def test_constructor_rejects_an_unknown_profile_policy(self) -> None:
        with self.assertRaisesRegex(TypeError, "profile_policy"):
            _agent(
                RecordingRetriever(),
                profile_policy=cast(ProfilePolicy, "bounded"),
            )


if __name__ == "__main__":
    unittest.main()
