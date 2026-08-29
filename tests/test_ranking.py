from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields

from conversational_search.intent import IntentState, Requirement
from conversational_search.ranking import (
    FUSED_ONLY_RANKING_POLICY,
    MAX_CANDIDATE_TEXT_CHARACTERS,
    MAX_CLAUSES,
    MAX_CLAUSE_TOKENS,
    STAGE_A_RANKING_POLICY,
    CandidateDocument,
    RankingPolicy,
    RankingResult,
    RankingTrace,
    _AtomicClause,
    _clause_match,
    _clauses,
    _requirement_satisfaction,
    _significant_tokens,
    rerank_stage_a,
)
from conversational_search.strategy import RouteWeights


def _rank(
    state: IntentState,
    documents: list[CandidateDocument],
    *,
    bm25_ids: tuple[str, ...] | None = None,
    dense_ids: tuple[str, ...] | None = None,
    fused_ids: tuple[str, ...] | None = None,
    weights: RouteWeights = RouteWeights(bm25=0.5, dense=0.5),
) -> RankingResult:
    document_ids = tuple(document.parent_asin for document in documents)
    return rerank_stage_a(
        state,
        documents,
        bm25_ids=document_ids if bm25_ids is None else bm25_ids,
        dense_ids=document_ids if dense_ids is None else dense_ids,
        fused_ids=document_ids if fused_ids is None else fused_ids,
        route_weights=weights,
    )


class RequirementMatchingTest(unittest.TestCase):
    def test_exact_all_token_and_partial_matches_have_declared_scores(self) -> None:
        clause = _AtomicClause(_significant_tokens("deep red"), 1.0)

        self.assertEqual(
            _clause_match(clause, _significant_tokens("A polished deep-red finish")),
            1.0,
        )
        self.assertEqual(
            _clause_match(clause, _significant_tokens("Red finish with a deep tone")),
            0.8,
        )
        self.assertEqual(
            _clause_match(clause, _significant_tokens("A deep finish")),
            0.25,
        )
        self.assertEqual(
            _clause_match(clause, _significant_tokens("Plain blue finish")),
            0.0,
        )

    def test_semicolon_clauses_category_and_provenance_are_weighted(self) -> None:
        state = IntentState(
            category="Trail Shoes",
            requirements=(
                Requirement("Color: red; Material: cotton", "answer", 2),
                Requirement("waterproof", "free_text", 3),
                Requirement("Budget: under 9999", "override", 4),
            ),
        )
        documents = tuple(
            _significant_tokens(text)
            for text in ("bright red", "soft cotton", "fully waterproof", "trail shoes")
        )

        satisfaction, observable = _requirement_satisfaction(
            _clauses(state),
            documents,
        )

        # The unrepresented budget clause is excluded. The denominator is
        # red(1) + cotton(1) + waterproof(.5) + category(.5) = 3.
        self.assertEqual(observable, 4)
        self.assertAlmostEqual(satisfaction[0], 1.0 / 3.0)
        self.assertAlmostEqual(satisfaction[1], 1.0 / 3.0)
        self.assertAlmostEqual(satisfaction[2], 1.0 / 6.0)
        self.assertAlmostEqual(satisfaction[3], 1.0 / 6.0)

        result = _rank(
            state,
            [
                CandidateDocument("RED", "bright red"),
                CandidateDocument("COTTON", "soft cotton"),
                CandidateDocument("WATER", "fully waterproof"),
                CandidateDocument("CATEGORY", "trail shoes"),
            ],
        )
        self.assertEqual(result.trace.observable_clause_count, 4)

    def test_labels_unicode_case_and_stopwords_are_normalized(self) -> None:
        state = IntentState(
            requirements=(Requirement("Color: for CAFÉ use", "answer", 2),)
        )
        result = _rank(state, [CandidateDocument("A", "Designed for cafe-use")])

        self.assertEqual(result.ranked_ids, ("A",))
        self.assertEqual(result.trace.observable_clause_count, 1)

    def test_budget_is_excluded_until_candidate_price_is_represented(self) -> None:
        state = IntentState(
            requirements=(
                Requirement("Budget: under 50", "answer", 2, "budget"),
            )
        )
        result = _rank(
            state,
            [
                CandidateDocument("INCUMBENT", "plain item"),
                CandidateDocument("FALSE_MATCH", "package of 50 pieces"),
            ],
        )

        self.assertEqual(result.ranked_ids, ("INCUMBENT", "FALSE_MATCH"))
        self.assertEqual(result.trace.observable_clause_count, 0)

    def test_clause_count_and_token_work_are_bounded(self) -> None:
        values = [" ".join(f"token0_{token}" for token in range(100))]
        values.extend(f"value{index}" for index in range(100))
        state = IntentState(
            requirements=(Requirement(";".join(values), "answer", 2),)
        )

        clauses = _clauses(state)

        self.assertEqual(len(clauses), MAX_CLAUSES)
        self.assertTrue(all(len(clause.tokens) <= MAX_CLAUSE_TOKENS for clause in clauses))


class StageAScoringTest(unittest.TestCase):
    def test_route_weights_reconstruct_the_original_weighted_rrf_leader(self) -> None:
        result = _rank(
            IntentState(),
            [CandidateDocument("A", ""), CandidateDocument("B", "")],
            bm25_ids=("A", "B"),
            dense_ids=("B", "A"),
            fused_ids=("A", "B"),
            weights=RouteWeights(bm25=0.6, dense=0.4),
        )

        self.assertEqual(result.ranked_ids, ("A", "B"))
        self.assertEqual(result.trace.beta, 0.20)
        self.assertEqual(result.trace.observable_clause_count, 0)

    def test_normalization_and_intermediate_beta_are_equation_locked(self) -> None:
        identifiers = tuple(f"P{index:03d}" for index in range(100))
        documents = [CandidateDocument(parent_asin, "plain item") for parent_asin in identifiers]
        documents[-1] = CandidateDocument(identifiers[-1], "waterproof item")
        state = IntentState(
            requirements=(Requirement("waterproof", "answer", 2),)
        )

        result = _rank(
            state,
            documents,
            bm25_ids=identifiers,
            dense_ids=(),
            fused_ids=identifiers,
        )

        self.assertAlmostEqual(result.trace.beta, 0.20 + 0.25 / 3.0)
        # With max-normalized RRF the rank-1 incumbent remains ahead of the
        # exact match at rank 100. Removing normalization makes the small raw
        # RRF values negligible and incorrectly flips this order.
        self.assertEqual(result.ranked_ids[0], identifiers[0])
        self.assertEqual(result.ranked_ids.index(identifiers[-1]), 18)

    def test_completeness_bounds_beta_and_can_promote_a_requirement_match(self) -> None:
        state = IntentState(
            requirements=(
                Requirement("waterproof", "answer", 2),
                Requirement("unrepresented alpha", "override", 3),
                Requirement("unrepresented omega", "initial_explicit", 1),
            )
        )
        result = _rank(
            state,
            [
                CandidateDocument("A", "plain shoe"),
                CandidateDocument("B", "waterproof shoe"),
            ],
        )

        self.assertEqual(result.trace.beta, 0.45)
        self.assertEqual(result.trace.observable_clause_count, 1)
        self.assertEqual(result.ranked_ids, ("B", "A"))

    def test_exact_score_ties_keep_original_fused_order(self) -> None:
        result = _rank(
            IntentState(),
            [CandidateDocument("B", "same"), CandidateDocument("A", "same")],
            bm25_ids=("A", "B"),
            dense_ids=("B", "A"),
            fused_ids=("B", "A"),
        )

        self.assertEqual(result.ranked_ids, ("B", "A"))
        self.assertEqual(result.trace.output_ids, ("B", "A"))

    def test_empty_union_is_empty_and_has_a_bounded_trace(self) -> None:
        result = _rank(IntentState(), [])

        self.assertEqual(result.ranked_ids, ())
        self.assertEqual(result.trace.input_ids, ())
        self.assertEqual(result.trace.output_ids, ())
        self.assertEqual(result.trace.beta, 0.20)
        self.assertEqual(result.trace.observable_clause_count, 0)


class RankingContractTest(unittest.TestCase):
    def test_policy_result_and_trace_are_immutable_and_label_free(self) -> None:
        self.assertIs(FUSED_ONLY_RANKING_POLICY, RankingPolicy.FUSED_ONLY)
        self.assertIs(STAGE_A_RANKING_POLICY, RankingPolicy.STAGE_A)
        document = CandidateDocument("A", "transient text")
        result = _rank(IntentState(), [document])

        with self.assertRaises(FrozenInstanceError):
            document.text = "changed"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.ranked_ids = ()  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.trace.beta = 0.0  # type: ignore[misc]
        self.assertEqual(
            {field.name for field in fields(RankingResult)},
            {"ranked_ids", "trace"},
        )
        self.assertEqual(
            {field.name for field in fields(RankingTrace)},
            {"input_ids", "output_ids", "beta", "observable_clause_count"},
        )
        self.assertNotIn("transient text", repr(result))
        for forbidden in ("target", "label", "ground_truth", "profile"):
            self.assertNotIn(forbidden, repr(result).casefold())

        with self.assertRaisesRegex(ValueError, "character limit"):
            CandidateDocument(
                "TOO_LONG",
                "x" * (MAX_CANDIDATE_TEXT_CHARACTERS + 1),
            )

    def test_alignment_duplicate_and_union_mismatches_are_rejected(self) -> None:
        documents = [CandidateDocument("A", "one"), CandidateDocument("B", "two")]
        weights = RouteWeights(bm25=0.5, dense=0.5)
        cases = (
            {
                "bm25_ids": ("A", "B"),
                "dense_ids": ("A", "B"),
                "fused_ids": ("B", "A"),
            },
            {
                "bm25_ids": ("A", "A"),
                "dense_ids": ("B",),
                "fused_ids": ("A", "B"),
            },
            {
                "bm25_ids": ("A",),
                "dense_ids": (),
                "fused_ids": ("A", "B"),
            },
        )
        for ranks in cases:
            with self.subTest(ranks=ranks):
                with self.assertRaises(ValueError):
                    rerank_stage_a(
                        IntentState(),
                        documents,
                        route_weights=weights,
                        **ranks,
                    )


if __name__ == "__main__":
    unittest.main()
