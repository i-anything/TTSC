from __future__ import annotations

import unittest

from conversational_search.utility_planner import (
    CandidateHypothesis,
    ExpectedUtilityCandidate,
    RetrievalChoice,
    SimulatedQuestion,
    hit_utility,
    plan_expected_utility,
    plan_one_step_action,
)


def _hypothesis(
    candidate_id: str,
    rank: int,
    weight: float,
    **answers: str,
) -> CandidateHypothesis:
    return CandidateHypothesis(
        candidate_id=candidate_id,
        rank=rank,
        weight=weight,
        answer_signatures=tuple(answers.items()),
    )


class OfficialUtilityTest(unittest.TestCase):
    def test_hit_utility_uses_the_official_formula(self) -> None:
        self.assertEqual(hit_utility(1, 1), 1.0)
        self.assertAlmostEqual(hit_utility(10, 2), 0.67)


class ExpectedUtilityPlannerTest(unittest.TestCase):
    def test_protocol_lock_withholds_and_uses_counterfactual_reranks(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.5),
            ExpectedUtilityCandidate("B", 2, 0.5),
        )
        question = SimulatedQuestion(
            "color",
            ordinary_post_ranks=(("A", 1), ("B", 1)),
        )

        plan = plan_expected_utility(
            candidates,
            (question,),
            current_turn=1,
            top_k=1,
            widths=(0, 1),
            retrieval_choices=(RetrievalChoice.REUSE,),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
            protocol_locked=True,
        )

        self.assertEqual((plan.selected.question, plan.selected.width), ("color", 0))
        self.assertAlmostEqual(plan.selected.expected_reward, hit_utility(2, 1))

    def test_post_reply_rank_is_used_instead_of_preserving_old_order(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.1),
            ExpectedUtilityCandidate("B", 3, 0.9),
        )
        question = SimulatedQuestion(
            "material",
            ordinary_post_ranks=(("A", 1), ("B", 1)),
        )

        plan = plan_expected_utility(
            candidates,
            (question,),
            current_turn=2,
            top_k=1,
            widths=(0,),
            retrieval_choices=(RetrievalChoice.REUSE,),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
            allow_zero_width=True,
        )

        self.assertEqual(plan.selected.question, "material")
        self.assertAlmostEqual(plan.selected.continuation_reward, hit_utility(3, 1))

    def test_shared_reply_world_is_mixed_without_extra_probability_mass(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.5),
            ExpectedUtilityCandidate("B", 2, 0.5),
        )
        question = SimulatedQuestion(
            "color",
            ordinary_post_ranks=(("A", 1), ("B", 1)),
            shared_post_ranks=(("A", 1), ("B", 2)),
            shared_reply_probability=0.5,
        )

        plan = plan_expected_utility(
            candidates,
            (question,),
            current_turn=1,
            top_k=2,
            widths=(0,),
            retrieval_choices=(RetrievalChoice.REUSE,),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
            allow_zero_width=True,
        )

        expected = 0.5 * hit_utility(2, 1) + 0.25 * (
            hit_utility(2, 1) + hit_utility(2, 2)
        )
        self.assertAlmostEqual(plan.selected.expected_reward, expected)

    def test_equal_deterministic_reretrieval_loses_to_reuse_on_cost(self) -> None:
        candidates = (ExpectedUtilityCandidate("A", 1, 1.0),)

        plan = plan_expected_utility(
            candidates,
            (),
            current_turn=1,
            top_k=1,
            widths=(1,),
            retrieval_choices=(
                RetrievalChoice.REUSE,
                RetrievalChoice.RERETRIEVE,
            ),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
            reretrieve_computation_cost=0.001,
        )

        self.assertIs(plan.selected.retrieval, RetrievalChoice.REUSE)
        self.assertIsNotNone(plan.runner_up)

    def test_reretrieval_can_win_when_it_has_modeled_residual_recovery(self) -> None:
        candidates = (ExpectedUtilityCandidate("A", 1, 0.2),)

        plan = plan_expected_utility(
            candidates,
            (),
            current_turn=1,
            top_k=1,
            widths=(0,),
            retrieval_choices=(
                RetrievalChoice.REUSE,
                RetrievalChoice.RERETRIEVE,
            ),
            out_of_pool_probability=0.8,
            protocol_confidence=0.0,
            protocol_locked=True,
            reretrieve_recovery_probability=0.5,
            reretrieve_recovery_rank=1,
            reretrieve_computation_cost=0.001,
        )

        self.assertIs(plan.selected.retrieval, RetrievalChoice.RERETRIEVE)
        self.assertGreater(plan.selected.expected_reward, plan.runner_up.expected_reward)

    def test_final_turn_has_no_question_and_full_legal_width(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.5),
            ExpectedUtilityCandidate("B", 2, 0.5),
        )

        plan = plan_expected_utility(
            candidates,
            (),
            current_turn=10,
            top_k=2,
            widths=(2,),
            retrieval_choices=(RetrievalChoice.REUSE,),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
        )

        self.assertIsNone(plan.selected.question)
        self.assertEqual(plan.selected.width, 2)

    def test_width_specific_post_rank_models_novelty_after_current_slate(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.1),
            ExpectedUtilityCandidate("B", 2, 0.9),
        )
        question = SimulatedQuestion(
            "style",
            ordinary_post_ranks=(("A", 1), ("B", 2)),
            ordinary_post_ranks_by_width=(
                (1, (("A", None), ("B", 1))),
            ),
        )

        plan = plan_expected_utility(
            candidates,
            (question,),
            current_turn=1,
            top_k=1,
            widths=(1,),
            retrieval_choices=(RetrievalChoice.REUSE,),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
        )

        self.assertEqual((plan.selected.question, plan.selected.width), ("style", 1))
        self.assertAlmostEqual(
            plan.selected.expected_reward,
            0.1 * hit_utility(1, 1) + 0.9 * hit_utility(2, 1),
        )

    def test_width_specific_rank_maps_must_be_complete_and_legal(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.5),
            ExpectedUtilityCandidate("B", 2, 0.5),
        )
        invalid_questions = (
            SimulatedQuestion(
                "color",
                ordinary_post_ranks=(("A", 1), ("B", 2)),
                ordinary_post_ranks_by_width=(
                    (2, (("A", 1), ("B", 2))),
                ),
            ),
            SimulatedQuestion(
                "color",
                ordinary_post_ranks=(("A", 1), ("B", 2)),
                ordinary_post_ranks_by_width=(
                    (1, (("A", 1),)),
                ),
            ),
        )

        for question in invalid_questions:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    plan_expected_utility(
                        candidates,
                        (question,),
                        current_turn=1,
                        top_k=2,
                        widths=(0, 1),
                        retrieval_choices=(RetrievalChoice.REUSE,),
                        out_of_pool_probability=0.0,
                        protocol_confidence=1.0,
                        allow_zero_width=True,
                    )

    def test_no_question_action_uses_the_actual_next_novelty_slate(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.1),
            ExpectedUtilityCandidate("B", 2, 0.9),
        )

        plan = plan_expected_utility(
            candidates,
            (),
            current_turn=1,
            top_k=1,
            widths=(1,),
            retrieval_choices=(RetrievalChoice.REUSE,),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
            no_question_post_ranks_by_width=(
                (1, (("A", None), ("B", 1))),
            ),
        )

        self.assertIsNone(plan.selected.question)
        self.assertAlmostEqual(
            plan.selected.expected_reward,
            0.1 * hit_utility(1, 1) + 0.9 * hit_utility(2, 1),
        )

    def test_zero_width_requires_an_explicit_exact_protocol_permission(self) -> None:
        candidates = (ExpectedUtilityCandidate("A", 1, 1.0),)

        with self.assertRaises(ValueError):
            plan_expected_utility(
                candidates,
                (),
                current_turn=1,
                top_k=1,
                widths=(0,),
                retrieval_choices=(RetrievalChoice.REUSE,),
                out_of_pool_probability=0.0,
                protocol_confidence=1.0,
            )
        with self.assertRaises(ValueError):
            plan_expected_utility(
                candidates,
                (),
                current_turn=1,
                top_k=1,
                widths=(0,),
                retrieval_choices=(RetrievalChoice.REUSE,),
                out_of_pool_probability=0.0,
                protocol_confidence=0.5,
                allow_zero_width=True,
            )

    def test_protocol_confidence_does_not_double_discount_shared_probability(
        self,
    ) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.2),
            ExpectedUtilityCandidate("B", 2, 0.8),
        )
        question = SimulatedQuestion(
            "color",
            ordinary_post_ranks=(("A", 1), ("B", 1)),
            shared_post_ranks=(("A", 2), ("B", 1)),
            shared_reply_probability=1.0,
        )

        plans = tuple(
            plan_expected_utility(
                candidates,
                (question,),
                current_turn=1,
                top_k=1,
                widths=(0,),
                retrieval_choices=(RetrievalChoice.REUSE,),
                out_of_pool_probability=0.0,
                protocol_confidence=confidence,
                protocol_locked=True,
                no_question_post_ranks_by_width=(
                    (0, (("A", 1), ("B", None))),
                ),
            )
            for confidence in (0.0, 1.0)
        )

        self.assertEqual(plans[0].selected.question, "color")
        self.assertAlmostEqual(
            plans[0].selected.expected_reward,
            0.8 * hit_utility(2, 1),
        )
        self.assertEqual(
            plans[0].selected.expected_reward,
            plans[1].selected.expected_reward,
        )

    def test_shared_reply_maps_must_describe_one_common_ranking(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.5),
            ExpectedUtilityCandidate("B", 2, 0.5),
        )
        question = SimulatedQuestion(
            "color",
            ordinary_post_ranks=(("A", 1), ("B", 1)),
            shared_post_ranks=(("A", 1), ("B", 1)),
            shared_reply_probability=0.5,
        )

        with self.assertRaises(ValueError):
            plan_expected_utility(
                candidates,
                (question,),
                current_turn=1,
                top_k=1,
                widths=(1,),
                retrieval_choices=(RetrievalChoice.REUSE,),
                out_of_pool_probability=0.0,
                protocol_confidence=1.0,
            )

    def test_top_k_cannot_exceed_the_official_limit(self) -> None:
        with self.assertRaises(ValueError):
            plan_expected_utility(
                (ExpectedUtilityCandidate("A", 1, 1.0),),
                (),
                current_turn=1,
                top_k=11,
                widths=(1,),
                retrieval_choices=(RetrievalChoice.REUSE,),
                out_of_pool_probability=0.0,
                protocol_confidence=1.0,
            )

    def test_probability_mass_uses_an_absolute_conservation_contract(self) -> None:
        with self.assertRaises(ValueError):
            plan_expected_utility(
                (ExpectedUtilityCandidate("A", 1, 0.5),),
                (),
                current_turn=1,
                top_k=1,
                widths=(1,),
                retrieval_choices=(RetrievalChoice.REUSE,),
                out_of_pool_probability=0.500000000002,
                protocol_confidence=1.0,
            )

    def test_no_question_projection_must_describe_one_common_ranking(self) -> None:
        candidates = (
            ExpectedUtilityCandidate("A", 1, 0.5),
            ExpectedUtilityCandidate("B", 2, 0.5),
        )
        with self.assertRaises(ValueError):
            plan_expected_utility(
                candidates,
                (),
                current_turn=1,
                top_k=2,
                widths=(1,),
                retrieval_choices=(RetrievalChoice.REUSE,),
                out_of_pool_probability=0.0,
                protocol_confidence=1.0,
                no_question_post_ranks_by_width=(
                    (1, (("A", 1), ("B", 1))),
                ),
            )

    def test_final_turn_reretrieval_gets_no_fictitious_recovery_credit(self) -> None:
        plan = plan_expected_utility(
            (ExpectedUtilityCandidate("A", 2, 0.2),),
            (),
            current_turn=10,
            top_k=1,
            widths=(1,),
            retrieval_choices=(
                RetrievalChoice.RERETRIEVE,
                RetrievalChoice.REUSE,
            ),
            out_of_pool_probability=0.8,
            protocol_confidence=1.0,
            reretrieve_recovery_probability=1.0,
            reretrieve_recovery_rank=1,
        )

        self.assertIs(plan.selected.retrieval, RetrievalChoice.REUSE)
        self.assertEqual(plan.selected.residual_risk, 0.8)

    def test_exact_utility_tie_prefers_no_question_reuse_and_wider_slate(self) -> None:
        candidates = (ExpectedUtilityCandidate("A", 1, 1.0),)
        question = SimulatedQuestion(
            "color",
            ordinary_post_ranks=(("A", 1),),
        )

        plan = plan_expected_utility(
            candidates,
            (question,),
            current_turn=1,
            top_k=2,
            widths=(1, 2),
            retrieval_choices=(
                RetrievalChoice.RERETRIEVE,
                RetrievalChoice.REUSE,
            ),
            out_of_pool_probability=0.0,
            protocol_confidence=1.0,
        )

        self.assertIsNone(plan.selected.question)
        self.assertIs(plan.selected.retrieval, RetrievalChoice.REUSE)
        self.assertEqual(plan.selected.width, 2)


class OneStepPlannerTest(unittest.TestCase):
    def test_waits_and_asks_the_question_that_makes_every_reply_rank_one(self) -> None:
        hypotheses = (
            _hypothesis("A", 1, 1, broad="same", precise="red"),
            _hypothesis("B", 2, 1, broad="same", precise="blue"),
            _hypothesis("C", 3, 1, broad="same", precise="green"),
        )

        action = plan_one_step_action(
            hypotheses,
            ("broad", "precise"),
            current_turn=1,
            top_k=1,
            protocol_locked=True,
        )

        self.assertEqual((action.question, action.width), ("precise", 0))
        self.assertAlmostEqual(action.value, hit_utility(2, 1))

    def test_exposes_full_width_when_it_reaches_high_probability_tail(self) -> None:
        hypotheses = (
            _hypothesis("A", 1, 0.01, detail="same"),
            _hypothesis("B", 2, 0.01, detail="same"),
            _hypothesis("C", 3, 0.01, detail="same"),
            _hypothesis("D", 4, 0.97, detail="same"),
        )

        action = plan_one_step_action(
            hypotheses,
            ("detail",),
            current_turn=1,
            top_k=2,
        )

        self.assertEqual((action.question, action.width), ("detail", 2))

    def test_candidate_independent_reply_world_discourages_unsafe_withholding(
        self,
    ) -> None:
        hypotheses = tuple(
            _hypothesis(
                f"ID{index}",
                index,
                1.0,
                detail=f"reply-{index}",
            )
            for index in range(1, 9)
        )

        informative_only = plan_one_step_action(
            hypotheses,
            ("detail",),
            current_turn=1,
            top_k=3,
        )
        latent_boundary = plan_one_step_action(
            hypotheses,
            ("detail",),
            current_turn=1,
            top_k=3,
            shared_reply_probability=0.5,
        )

        self.assertEqual(informative_only.width, 1)
        self.assertEqual(latent_boundary.width, 3)

    def test_final_turn_forces_full_width_without_a_question(self) -> None:
        hypotheses = (
            _hypothesis("A", 1, 0.1),
            _hypothesis("B", 2, 0.2),
            _hypothesis("C", 3, 0.3),
            _hypothesis("D", 4, 0.4),
        )

        action = plan_one_step_action(
            hypotheses,
            (),
            current_turn=10,
            top_k=3,
        )

        expected = (
            0.1 * hit_utility(10, 1)
            + 0.2 * hit_utility(10, 2)
            + 0.3 * hit_utility(10, 3)
        )
        self.assertEqual((action.question, action.width), (None, 3))
        self.assertAlmostEqual(action.value, expected)

    def test_final_turn_never_withholds_the_slate_even_if_lock_is_still_set(
        self,
    ) -> None:
        hypotheses = (
            _hypothesis("A", 1, 1),
            _hypothesis("B", 2, 1),
        )

        action = plan_one_step_action(
            hypotheses,
            (),
            current_turn=10,
            top_k=2,
            protocol_locked=True,
        )

        self.assertEqual((action.question, action.width), (None, 2))
        self.assertEqual(action.value, 0.0)

    def test_empty_width_is_available_only_during_a_protocol_lock(self) -> None:
        hypotheses = (
            _hypothesis("A", 1, 1, detail="a"),
            _hypothesis("B", 2, 1, detail="b"),
        )

        locked = plan_one_step_action(
            hypotheses,
            ("detail",),
            current_turn=1,
            top_k=1,
            protocol_locked=True,
        )
        unlocked = plan_one_step_action(
            hypotheses,
            ("detail",),
            current_turn=1,
            top_k=1,
        )

        self.assertEqual(locked.width, 0)
        self.assertEqual(unlocked.width, 1)

    def test_exact_ties_use_caller_question_then_width_order(self) -> None:
        hypotheses = (
            _hypothesis("A", 1, 1, alpha="x", beta="x"),
            _hypothesis("B", 2, 1, alpha="y", beta="y"),
        )

        action = plan_one_step_action(
            hypotheses,
            ("beta", "alpha"),
            current_turn=1,
            top_k=1,
            protocol_locked=True,
        )

        self.assertEqual((action.question, action.width), ("beta", 0))

    def test_weights_are_normalized_internally(self) -> None:
        base = (
            _hypothesis("A", 1, 1, detail="x"),
            _hypothesis("B", 2, 3, detail="x"),
        )
        scaled = (
            _hypothesis("A", 1, 10, detail="x"),
            _hypothesis("B", 2, 30, detail="x"),
        )

        first = plan_one_step_action(base, ("detail",), current_turn=3, top_k=1)
        second = plan_one_step_action(scaled, ("detail",), current_turn=3, top_k=1)

        self.assertEqual(first, second)

    def test_invalid_bounds_weights_ranks_and_signatures_fail_closed(self) -> None:
        valid = (
            _hypothesis("A", 1, 1, detail="x"),
            _hypothesis("B", 2, 1, detail="y"),
        )
        invalid_cases = (
            ((), ("detail",), 1, 1),
            (valid, ("detail",), 0, 1),
            (valid, ("detail",), 1, 0),
            (valid, ("detail",), 1, 3),
            ((_hypothesis("A", 1, 0, detail="x"),), ("detail",), 1, 1),
            ((_hypothesis("A", 1, -1, detail="x"),), ("detail",), 1, 1),
            (
                (
                    _hypothesis("A", 1, 1, detail="x"),
                    _hypothesis("B", 3, 1, detail="y"),
                ),
                ("detail",),
                1,
                1,
            ),
            ((_hypothesis("A", 1, 1, other="x"),), ("detail",), 1, 1),
        )

        for hypotheses, questions, turn, top_k in invalid_cases:
            with self.subTest(
                hypotheses=hypotheses,
                questions=questions,
                turn=turn,
                top_k=top_k,
            ):
                with self.assertRaises((TypeError, ValueError)):
                    plan_one_step_action(
                        hypotheses,
                        questions,
                        current_turn=turn,
                        top_k=top_k,
                    )


if __name__ == "__main__":
    unittest.main()
