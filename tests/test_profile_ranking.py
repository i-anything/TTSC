from __future__ import annotations

import unittest
from unittest.mock import patch

from conversational_search.intent import IntentState, Requirement
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    DISABLED_PROFILE_POLICY,
    NEUTRAL_PROFILE_PRIOR,
    ProductTheme,
    ProfilePrior,
)
from conversational_search.ranking import (
    _PROFILE_CUE_TEXT,
    _StageAComputation,
    CandidateDocument,
    ProfileResidualStatus,
    RankingResult,
    RankingTrace,
    _candidate_theme_mask,
    _significant_tokens,
    rerank_stage_a,
    rerank_stage_a_with_profile,
)
from conversational_search.strategy import RouteWeights


_WEIGHTS = RouteWeights(bm25=0.5, dense=0.5)


def _base(
    state: IntentState,
    documents: tuple[CandidateDocument, ...],
) -> object:
    identifiers = tuple(document.parent_asin for document in documents)
    return rerank_stage_a(
        state,
        documents,
        bm25_ids=identifiers,
        dense_ids=identifiers,
        fused_ids=identifiers,
        route_weights=_WEIGHTS,
    )


def _profiled(
    state: IntentState,
    documents: tuple[CandidateDocument, ...],
    prior: ProfilePrior,
    *,
    policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
):
    identifiers = tuple(document.parent_asin for document in documents)
    return rerank_stage_a_with_profile(
        state,
        documents,
        bm25_ids=identifiers,
        dense_ids=identifiers,
        fused_ids=identifiers,
        route_weights=_WEIGHTS,
        profile_prior=prior,
        profile_policy=policy,
    )


class ProfileCueContractTests(unittest.TestCase):
    def test_every_frozen_candidate_cue_maps_to_its_declared_theme(self) -> None:
        for theme, cues in _PROFILE_CUE_TEXT.items():
            for cue in cues:
                with self.subTest(theme=theme.name, cue=cue):
                    self.assertTrue(
                        _candidate_theme_mask(_significant_tokens(cue)) & theme
                    )

    def test_cues_require_exact_tokens_and_contiguous_multiword_phrases(self) -> None:
        cases = (
            ("uncomfortable", ProductTheme.COMFORT),
            ("water proof", ProductTheme.WEATHER_PROTECTION),
            ("moisture advanced wicking", ProductTheme.BREATHABILITY),
            ("environmentally very friendly", ProductTheme.SUSTAINABILITY),
        )
        for text, forbidden_theme in cases:
            with self.subTest(text=text):
                self.assertFalse(
                    _candidate_theme_mask(_significant_tokens(text))
                    & forbidden_theme
                )

        self.assertTrue(
            _candidate_theme_mask(_significant_tokens("moisture-wicking shell"))
            & ProductTheme.BREATHABILITY
        )


class ProfileResidualRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = (
            CandidateDocument("BASE", "plain everyday item"),
            CandidateDocument("PROFILE", "ergonomic cushioned item"),
        )
        self.comfort = ProfilePrior(ProductTheme.COMFORT)

    def assertExactBaseFallback(self, state, prior, policy, status) -> None:
        baseline = _base(state, self.documents)
        candidate = _profiled(state, self.documents, prior, policy=policy)

        self.assertEqual(candidate.status, status)
        self.assertEqual(candidate.ranking, baseline)
        self.assertEqual(candidate.ranking.trace, baseline.trace)

    def test_eligible_residual_can_promote_a_theme_match(self) -> None:
        baseline = _base(IntentState(), self.documents)
        candidate = _profiled(IntentState(), self.documents, self.comfort)

        self.assertEqual(baseline.ranked_ids, ("BASE", "PROFILE"))
        self.assertEqual(candidate.ranking.ranked_ids, ("PROFILE", "BASE"))
        self.assertEqual(candidate.status, ProfileResidualStatus.APPLIED)
        self.assertEqual(candidate.requested_theme_count, 1)
        self.assertEqual(candidate.represented_theme_count, 1)
        self.assertEqual(candidate.ranking.trace.input_ids, baseline.trace.input_ids)
        self.assertEqual(candidate.ranking.trace.beta, baseline.trace.beta)

    def test_category_only_remains_eligible(self) -> None:
        candidate = _profiled(
            IntentState(category="office chair"),
            self.documents,
            self.comfort,
        )

        self.assertEqual(candidate.status, ProfileResidualStatus.APPLIED)

    def test_disabled_neutral_and_active_requirement_are_exact_fallbacks(self) -> None:
        cases = (
            (
                IntentState(),
                self.comfort,
                DISABLED_PROFILE_POLICY,
                ProfileResidualStatus.DISABLED,
            ),
            (
                IntentState(),
                NEUTRAL_PROFILE_PRIOR,
                BOUNDED_RESIDUAL_PROFILE_POLICY,
                ProfileResidualStatus.NEUTRAL,
            ),
            (
                IntentState(
                    requirements=(Requirement("blue", "answer", 2, "color"),)
                ),
                self.comfort,
                BOUNDED_RESIDUAL_PROFILE_POLICY,
                ProfileResidualStatus.ACTIVE_REQUIREMENTS,
            ),
        )
        for state, prior, policy, status in cases:
            with self.subTest(status=status.value):
                self.assertExactBaseFallback(state, prior, policy, status)

    def test_unrepresented_and_constant_signals_are_exact_fallbacks(self) -> None:
        unrepresented = (
            CandidateDocument("A", "plain item"),
            CandidateDocument("B", "ordinary item"),
        )
        baseline = _base(IntentState(), unrepresented)
        candidate = _profiled(IntentState(), unrepresented, self.comfort)
        self.assertEqual(
            candidate.status,
            ProfileResidualStatus.NO_REPRESENTED_THEME,
        )
        self.assertEqual(candidate.ranking, baseline)

        constant = (
            CandidateDocument("A", "comfortable item"),
            CandidateDocument("B", "cushioned item"),
        )
        baseline = _base(IntentState(), constant)
        candidate = _profiled(IntentState(), constant, self.comfort)
        self.assertEqual(candidate.status, ProfileResidualStatus.CONSTANT_SCORE)
        self.assertEqual(candidate.ranking, baseline)

    def test_only_represented_requested_themes_form_the_denominator(self) -> None:
        documents = (
            CandidateDocument("PLAIN", "plain item"),
            CandidateDocument("ONE", "comfortable item"),
            CandidateDocument("BOTH", "comfortable durable item"),
        )
        prior = ProfilePrior(
            ProductTheme.COMFORT
            | ProductTheme.DURABILITY
            | ProductTheme.SUSTAINABILITY
        )

        candidate = _profiled(IntentState(), documents, prior)

        self.assertEqual(candidate.status, ProfileResidualStatus.APPLIED)
        self.assertEqual(candidate.requested_theme_count, 3)
        self.assertEqual(candidate.represented_theme_count, 2)
        self.assertEqual(candidate.ranking.ranked_ids[0], "BOTH")

    def test_exact_final_score_tie_uses_phase7_order(self) -> None:
        trace = RankingTrace(
            input_ids=("FIRST", "SECOND"),
            output_ids=("SECOND", "FIRST"),
            beta=0.2,
            observable_clause_count=0,
        )
        computation = _StageAComputation(
            ranking=RankingResult(("SECOND", "FIRST"), trace),
            # 0.95 * (0.05 / 0.95) is exactly 0.05 in binary64 here.
            base_scores=(0.0, 0.05 / 0.95),
            tokenized_documents=(("comfort",), ("plain",)),
        )
        documents = (
            CandidateDocument("FIRST", "comfort"),
            CandidateDocument("SECOND", "plain"),
        )
        with (
            patch(
                "conversational_search.ranking._compute_stage_a",
                return_value=computation,
            ),
            patch(
                "conversational_search.ranking._profile_residual_scores",
                return_value=(1.0, 0.0),
            ),
        ):
            candidate = _profiled(IntentState(), documents, self.comfort)

        self.assertEqual(candidate.status, ProfileResidualStatus.APPLIED)
        self.assertEqual(candidate.ranking.ranked_ids, ("SECOND", "FIRST"))

    def test_profile_scoring_faults_and_malformed_inputs_fail_closed(self) -> None:
        baseline = _base(IntentState(), self.documents)
        fault_values = ((float("nan"), 0.0), (1.01, 0.0), (0.0,))
        for values in fault_values:
            with self.subTest(values=values):
                with patch(
                    "conversational_search.ranking._profile_residual_scores",
                    return_value=values,
                ):
                    candidate = _profiled(
                        IntentState(),
                        self.documents,
                        self.comfort,
                    )
                self.assertEqual(
                    candidate.status,
                    ProfileResidualStatus.SCORING_FALLBACK,
                )
                self.assertEqual(candidate.ranking, baseline)

        malformed = _profiled(
            IntentState(),
            self.documents,
            object(),  # type: ignore[arg-type]
        )
        self.assertEqual(
            malformed.status,
            ProfileResidualStatus.SCORING_FALLBACK,
        )
        self.assertEqual(malformed.ranking, baseline)

    def test_empty_candidate_pool_is_an_exact_neutral_fallback(self) -> None:
        candidate = _profiled(IntentState(), (), self.comfort)

        self.assertEqual(
            candidate.status,
            ProfileResidualStatus.NO_REPRESENTED_THEME,
        )
        self.assertEqual(candidate.ranking, _base(IntentState(), ()))


if __name__ == "__main__":
    unittest.main()
