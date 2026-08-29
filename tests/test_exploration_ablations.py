from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

from conversational_search.ranking import RankingPolicy
from conversational_search.retrieval import RetrievalResult, RetrievalTrace
from conversational_search.slates import SlatePolicy
from conversational_search.strategy import RouteWeights
from scripts import run_exploration_ablations as ablations


class _Backend:
    dense_available = True
    bm25_available = True

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
    ) -> RetrievalResult:
        return RetrievalResult(
            recommendations=("A",),
            trace=RetrievalTrace(
                bm25_ids=("A",),
                dense_ids=("A",),
                fused_ids=("A",),
                bm25_status="ok",
                dense_status="ok",
                used_fallback=False,
            ),
        )

    def candidate_documents(self, parent_asins: tuple[str, ...]) -> tuple:
        return ()


def _result(hit: bool) -> dict:
    return {
        "sample_count": 1,
        "hit_rate_at_10": float(hit),
        "mrr": float(hit),
        "mttc": 1.0 if hit else 11.0,
        "efficiency": float(hit),
        "recommended_technical_score": float(hit),
        "reported_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "private": "TARGET",
        },
        "scenario_metrics": {
            "buying": {
                "sample_count": 1,
                "hit_rate_at_10": float(hit),
                "mrr": float(hit),
                "mttc": 1.0 if hit else 11.0,
                "target": "TARGET",
            }
        },
        "sessions": [
            {
                "sample_id": "SECRET_SAMPLE",
                "scenario_type": "buying",
                "hit": hit,
                "first_hit_turn": 1 if hit else None,
                "best_rank": 1 if hit else None,
                "reciprocal_rank": float(hit),
            }
        ],
    }


def _run(verify_determinism: bool) -> tuple[dict, list]:
    backend = _Backend()

    class _Agent:
        instances: list["_Agent"] = []

        def __init__(
            self,
            catalog: Path,
            *,
            retriever: object | None = None,
            question_policy: object,
            fusion_policy: object,
            ranking_policy: RankingPolicy,
            slate_policy: SlatePolicy,
        ) -> None:
            self.retrieval_backend = backend if retriever is None else retriever
            self.retriever = retriever
            self.ranking_policy = ranking_policy
            self.slate_policy = slate_policy
            self.ranking_attempts = 0
            self.slate_attempts = 0
            self.instances.append(self)

        def reset(self, session_id: str, user_profile: dict) -> None:
            pass

        def respond(
            self,
            session_id: str,
            user_message: str,
            turn: int,
            top_k: int,
        ) -> dict:
            self.retrieval_backend.search_with_trace(
                "dense",
                "lexical",
                top_k,
                route_weights=RouteWeights(bm25=0.4, dense=0.6),
            )
            self.ranking_attempts += 1
            if self.slate_policy is SlatePolicy.STAGNATION_AWARE:
                self.slate_attempts += 1
            return {"message": "SECRET_MESSAGE", "recommendations": []}

        @property
        def ranking_health(self) -> dict[str, int | str]:
            return {
                "policy": self.ranking_policy.value,
                "attempts": self.ranking_attempts,
                "successes": self.ranking_attempts,
                "failures": 0,
                "unavailable_skips": 0,
            }

        @property
        def slate_health(self) -> dict[str, int | str]:
            return {
                "policy": self.slate_policy.value,
                "attempts": self.slate_attempts,
                "successes": self.slate_attempts,
                "failures": 0,
                "initializations": int(self.slate_attempts > 0),
                "ranking_resets": 0,
                "stagnant_turns": max(0, self.slate_attempts - 1),
                "repeat_backfills": 0,
            }

    def _evaluate(agent: object, *args: object) -> dict:
        hit = agent._delegate.slate_policy is SlatePolicy.STAGNATION_AWARE
        agent.reset("SECRET_SESSION", {"profile": "SECRET_PROFILE"})
        for turn in range(1, 2 if hit else 11):
            agent.respond("SECRET_SESSION", "SECRET_MESSAGE", turn, 10)
        return _result(hit)

    with (
        mock.patch.object(ablations, "load_jsonl", return_value=[{}]),
        mock.patch.object(
            ablations,
            "catalog_index",
            return_value=({"A"}, {"A": ["Shoes"]}, {"A": {}}),
        ),
        mock.patch.object(ablations, "ConversationalSearchAgent", _Agent),
        mock.patch.object(ablations, "evaluate", side_effect=_evaluate),
        mock.patch.object(ablations, "_sha256", return_value="0" * 64),
        mock.patch.object(ablations, "_validate_phase4_baseline"),
    ):
        payload = ablations.run_exploration_ablations(
            "catalog.jsonl",
            "public.jsonl",
            verify_determinism=verify_determinism,
        )
    return payload, _Agent.instances


class ExplorationAblationTest(unittest.TestCase):
    def test_phase4_to_phase5_order_deltas_health_and_privacy(self) -> None:
        payload, agents = _run(False)

        self.assertEqual(payload["run_order"], ["repeat_top", "stagnation_aware"])
        self.assertEqual(
            [agent.slate_policy.value for agent in agents],
            ["repeat_top", "repeat_top", "stagnation_aware"],
        )
        self.assertTrue(
            all(agent.ranking_policy is RankingPolicy.STAGE_A for agent in agents)
        )
        self.assertIs(agents[1].retriever._backend, agents[0].retrieval_backend)
        self.assertIs(agents[2].retriever._backend, agents[0].retrieval_backend)
        self.assertEqual(
            payload["comparison"]["scenario_hit_count_delta"],
            {"buying": 1},
        )
        self.assertEqual(
            payload["variants"]["stagnation_aware"]["slate_health"]["policy"],
            "stagnation_aware",
        )
        self.assertIn(
            "docs/phase5_experiment_contract.json",
            payload["source_sha256"],
        )
        self.assertIn(
            "scripts/run_reranking_ablations.py",
            payload["source_sha256"],
        )
        serialized = json.dumps(payload)
        for forbidden in (
            "SECRET_SAMPLE",
            "SECRET_SESSION",
            "SECRET_MESSAGE",
            "SECRET_PROFILE",
            "TARGET",
            "sample_id",
            '"sessions"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_candidate_replay_compares_the_full_evaluator_payload(self) -> None:
        payload, agents = _run(True)

        self.assertEqual(
            payload["run_order"],
            ["repeat_top", "stagnation_aware", "stagnation_aware"],
        )
        self.assertEqual(
            [agent.slate_policy.value for agent in agents[-2:]],
            ["stagnation_aware", "stagnation_aware"],
        )
        self.assertTrue(
            payload["determinism"]["candidate_evaluator_payload_equal"]
        )

    def test_phase4_baseline_validation_is_exact(self) -> None:
        result = {
            **ablations.PHASE4_BASELINE,
            "reported_token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        ablations._validate_phase4_baseline(result)
        drifted = {**result, "mrr": result["mrr"] + 0.000001}
        with self.assertRaisesRegex(RuntimeError, "baseline metrics drifted"):
            ablations._validate_phase4_baseline(drifted)

    def test_variant_health_rejects_reranker_slate_or_fallback_faults(self) -> None:
        route = {
            "bm25": {"ok": 3},
            "dense": {"ok": 3},
            "fallback_turns": 0,
        }
        ranking = {
            "attempts": 3,
            "successes": 3,
            "failures": 0,
            "unavailable_skips": 0,
        }
        slate = {
            "attempts": 3,
            "successes": 3,
            "failures": 0,
            "initializations": 1,
            "ranking_resets": 1,
            "stagnant_turns": 1,
        }

        ablations._validate_variant_health(
            3,
            ablations.STAGNATION_AWARE_SLATE_POLICY,
            route,
            ranking,
            slate,
        )
        for bad_route, bad_ranking, bad_slate in (
            ({**route, "fallback_turns": 1}, ranking, slate),
            (route, {**ranking, "failures": 1}, slate),
            (route, ranking, {**slate, "failures": 1}),
        ):
            with self.assertRaises(RuntimeError):
                ablations._validate_variant_health(
                    3,
                    ablations.STAGNATION_AWARE_SLATE_POLICY,
                    bad_route,
                    bad_ranking,
                    bad_slate,
                )

    def test_output_path_protects_contract_and_imported_helpers(self) -> None:
        catalog = Path("catalog.jsonl").resolve()
        dataset = Path("public.jsonl").resolve()
        repository_root = Path(ablations.__file__).resolve().parents[1]

        for relative in (
            "docs/phase5_experiment_contract.json",
            "scripts/run_reranking_ablations.py",
        ):
            with self.subTest(relative=relative):
                with self.assertRaisesRegex(ValueError, "input or source"):
                    ablations._validate_output_path(
                        repository_root / relative,
                        catalog,
                        dataset,
                    )


if __name__ == "__main__":
    unittest.main()
