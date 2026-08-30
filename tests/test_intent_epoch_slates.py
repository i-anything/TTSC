from __future__ import annotations

import itertools
import unittest

from conversational_search.slates import (
    INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
    IntentEpochSlateStatus,
    SlateState,
    select_slate,
    select_slate_with_intent_epoch_novelty,
)
from scripts.verify_phase13_slate_oracle import (
    EXPECTED_SHA256,
    verify as verify_phase13_oracle,
)


class IntentEpochSlateTests(unittest.TestCase):
    def test_first_slate_is_exact_protected_baseline(self) -> None:
        pool = ("A", "B", "C")
        state = SlateState()
        signature = (0, "first")
        expected = select_slate(
            STAGNATION_AWARE_SLATE_POLICY,
            state,
            signature,
            pool,
            2,
        )
        result = select_slate_with_intent_epoch_novelty(
            state,
            signature,
            pool,
            2,
        )
        self.assertEqual(result.selection, expected)
        self.assertIs(result.status, IntentEpochSlateStatus.FIRST)
        self.assertEqual(result.eligible_prior_shown, 0)

    def test_unchanged_signature_is_exact_protected_baseline(self) -> None:
        signature = (0, "same")
        state = SlateState(signature=signature, shown_ids=("A", "B"))
        pool = ("A", "B", "C", "D")
        expected = select_slate(
            STAGNATION_AWARE_SLATE_POLICY,
            state,
            signature,
            pool,
            2,
        )
        result = select_slate_with_intent_epoch_novelty(
            state,
            signature,
            pool,
            2,
        )
        self.assertEqual(result.selection, expected)
        self.assertEqual(result.selection.selected_ids, ("C", "D"))
        self.assertIs(result.status, IntentEpochSlateStatus.UNCHANGED)
        self.assertEqual(result.eligible_prior_shown, 2)

    def test_changed_epoch_resets_to_exact_protected_baseline(self) -> None:
        state = SlateState(signature=(0, "old"), shown_ids=("A", "B"))
        signature = (1, "replacement")
        pool = ("A", "C", "B", "D")
        expected = select_slate(
            STAGNATION_AWARE_SLATE_POLICY,
            state,
            signature,
            pool,
            2,
        )
        result = select_slate_with_intent_epoch_novelty(
            state,
            signature,
            pool,
            2,
        )
        self.assertEqual(result.selection, expected)
        self.assertEqual(result.selection.selected_ids, ("A", "C"))
        self.assertIs(result.status, IntentEpochSlateStatus.EPOCH_RESET)
        self.assertEqual(result.eligible_prior_shown, 0)

    def test_same_epoch_changed_signature_carries_history(self) -> None:
        state = SlateState(signature=(0, "old"), shown_ids=("A", "B"))
        signature = (0, "refined")
        pool = ("A", "C", "B", "D")
        result = select_slate_with_intent_epoch_novelty(
            state,
            signature,
            pool,
            2,
        )
        self.assertEqual(result.selection.selected_ids, ("C", "D"))
        self.assertEqual(result.selection.state.shown_ids, ("A", "B", "C", "D"))
        self.assertTrue(result.selection.trace.signature_changed)
        self.assertFalse(result.selection.trace.stagnant_turn)
        self.assertEqual(result.selection.trace.unseen_selected, 2)
        self.assertEqual(result.selection.trace.repeat_backfills, 0)
        self.assertIs(result.status, IntentEpochSlateStatus.CARRIED)
        self.assertEqual(result.eligible_prior_shown, 2)

    def test_same_epoch_exhaustion_backfills_in_current_rank_order(self) -> None:
        state = SlateState(
            signature=(2, "old"),
            shown_ids=("C", "A", "B"),
        )
        result = select_slate_with_intent_epoch_novelty(
            state,
            (2, "refined"),
            ("A", "B", "C"),
            2,
        )
        self.assertEqual(result.selection.selected_ids, ("A", "B"))
        self.assertEqual(result.selection.state.shown_ids, ("C", "A", "B"))
        self.assertEqual(result.selection.trace.unseen_selected, 0)
        self.assertEqual(result.selection.trace.repeat_backfills, 2)

    def test_malformed_epoch_fails_closed_to_protected_baseline(self) -> None:
        state = SlateState(
            signature=("not-an-epoch", "old"),
            shown_ids=("A", "B"),
        )
        signature = (0, "new")
        pool = ("A", "C", "B", "D")
        expected = select_slate(
            STAGNATION_AWARE_SLATE_POLICY,
            state,
            signature,
            pool,
            2,
        )
        result = select_slate_with_intent_epoch_novelty(
            state,
            signature,
            pool,
            2,
        )
        self.assertEqual(result.selection, expected)
        self.assertIs(
            result.status,
            IntentEpochSlateStatus.VALIDATION_FALLBACK,
        )

    def test_empty_and_zero_limit_are_exact(self) -> None:
        for pool, limit in (((), 2), (("A",), 0)):
            with self.subTest(pool=pool, limit=limit):
                state = SlateState(signature=(0, "old"), shown_ids=("A",))
                signature = (0, "new")
                expected = select_slate(
                    STAGNATION_AWARE_SLATE_POLICY,
                    state,
                    signature,
                    pool,
                    limit,
                )
                result = select_slate_with_intent_epoch_novelty(
                    state,
                    signature,
                    pool,
                    limit,
                )
                self.assertEqual(result.selection, expected)
                self.assertIs(result.status, IntentEpochSlateStatus.EMPTY)

    def test_public_candidate_policy_matches_traced_selector(self) -> None:
        state = SlateState(signature=(3, "old"), shown_ids=("A",))
        signature = (3, "new")
        pool = ("A", "B", "C")
        direct = select_slate(
            INTENT_EPOCH_NOVELTY_SLATE_POLICY,
            state,
            signature,
            pool,
            2,
        )
        traced = select_slate_with_intent_epoch_novelty(
            state,
            signature,
            pool,
            2,
        )
        self.assertEqual(direct, traced.selection)

    def test_target_rank_cannot_worsen_when_target_was_not_shown(self) -> None:
        ids = ("A", "B", "C", "D")
        for pool in itertools.permutations(ids):
            for shown_size in range(len(ids) + 1):
                for shown in itertools.combinations(ids, shown_size):
                    state = SlateState(
                        signature=(0, "old"),
                        shown_ids=shown,
                    )
                    baseline = select_slate(
                        STAGNATION_AWARE_SLATE_POLICY,
                        state,
                        (0, "new"),
                        pool,
                        3,
                    ).selected_ids
                    candidate = select_slate_with_intent_epoch_novelty(
                        state,
                        (0, "new"),
                        pool,
                        3,
                    ).selection.selected_ids
                    for target in ids:
                        if target in shown:
                            continue
                        baseline_rank = (
                            baseline.index(target) + 1
                            if target in baseline
                            else 10_000
                        )
                        candidate_rank = (
                            candidate.index(target) + 1
                            if target in candidate
                            else 10_000
                        )
                        self.assertLessEqual(candidate_rank, baseline_rank)

    def test_invalid_inputs_retain_existing_validation_contract(self) -> None:
        with self.assertRaises(TypeError):
            select_slate_with_intent_epoch_novelty(  # type: ignore[arg-type]
                object(),
                (0,),
                ("A",),
                1,
            )

    def test_frozen_phase13_oracle(self) -> None:
        result = verify_phase13_oracle()
        self.assertEqual(result["cases"], 45_600)
        self.assertEqual(result["exhaustive_cases"], 15_600)
        self.assertEqual(result["random_cases"], 30_000)
        self.assertEqual(result["digest"], EXPECTED_SHA256)
        with self.assertRaises(ValueError):
            select_slate_with_intent_epoch_novelty(
                SlateState(),
                (0,),
                ("A", "A"),
                1,
            )


if __name__ == "__main__":
    unittest.main()
