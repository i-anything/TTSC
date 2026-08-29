from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conversational_search.ranking import RankingPolicy
from conversational_search.retrieval import RetrievalResult, RetrievalTrace
from conversational_search.strategy import RouteWeights
from scripts import run_reranking_ablations as ablations


class _Backend:
    dense_available = True
    bm25_available = True

    def __init__(self) -> None:
        self.calls = 0

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
    ) -> RetrievalResult:
        self.calls += 1
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


def _result(candidate: bool) -> dict:
    hit = candidate
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
                "ground_truth": "TARGET",
            }
        ],
    }


def _run_with_fakes(verify_determinism: bool) -> tuple[dict, list, _Backend, int]:
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
        ) -> None:
            self.retrieval_backend = backend if retriever is None else retriever
            self.retriever = retriever
            self.question_policy = question_policy
            self.fusion_policy = fusion_policy
            self.ranking_policy = ranking_policy
            self._attempts = 0
            self._successes = 0
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
            retrieval = self.retrieval_backend.search_with_trace(
                "dense",
                "lexical",
                top_k,
                route_weights=RouteWeights(bm25=0.4, dense=0.6),
            )
            if self.ranking_policy is RankingPolicy.STAGE_A:
                self._attempts += 1
                self.retrieval_backend.candidate_documents(
                    retrieval.trace.fused_ids
                )
                self._successes += 1
            return {"message": "SECRET_MESSAGE", "recommendations": []}

        @property
        def ranking_health(self) -> dict[str, int | str]:
            return {
                "policy": self.ranking_policy.value,
                "attempts": self._attempts,
                "successes": self._successes,
                "failures": 0,
            }

    evaluate_calls = 0

    def _evaluate(agent: object, *args: object) -> dict:
        nonlocal evaluate_calls
        evaluate_calls += 1
        candidate = agent._delegate.ranking_policy is RankingPolicy.STAGE_A
        agent.reset("SECRET_SESSION", {"profile": "SECRET_PROFILE"})
        for turn in range(1, 2 if candidate else 11):
            agent.respond("SECRET_SESSION", "SECRET_MESSAGE", turn, 10)
        return _result(candidate)

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
    ):
        payload = ablations.run_reranking_ablations(
            "catalog.jsonl",
            "public.jsonl",
            verify_determinism=verify_determinism,
        )
    return payload, _Agent.instances, backend, evaluate_calls


class RespondLatencyTest(unittest.TestCase):
    def test_summary_uses_milliseconds_and_retains_no_request_fields(self) -> None:
        class _Delegate:
            def reset(self, session_id: str, user_profile: dict) -> None:
                pass

            def respond(self, *args: object) -> dict:
                return {"message": "secret"}

        ticks = iter((0, 1_000_000, 2_000_000, 5_000_000))
        agent = ablations.RespondLatencyAgent(
            _Delegate(),  # type: ignore[arg-type]
            clock_ns=lambda: next(ticks),
        )
        agent.respond("session", "first secret", 1, 10)
        agent.respond("session", "second secret", 2, 10)

        self.assertEqual(
            agent.latency_summary(),
            {
                "count": 2,
                "warm_count": 1,
                "p50": 2.0,
                "p90": 2.8,
                "p95": 2.9,
                "p99": 2.98,
                "warm_p95": 3.0,
                "max": 3.0,
                "total": 4.0,
            },
        )


class RerankingAblationTest(unittest.TestCase):
    def test_fixed_order_shared_backend_deltas_and_privacy_projection(self) -> None:
        payload, agents, backend, evaluate_calls = _run_with_fakes(False)

        self.assertEqual(evaluate_calls, 2)
        self.assertEqual(payload["run_order"], ["fused_only", "stage_a"])
        self.assertEqual(
            [agent.ranking_policy.value for agent in agents],
            ["fused_only", "fused_only", "stage_a"],
        )
        self.assertIs(agents[1].retriever._backend, backend)
        self.assertIs(agents[2].retriever._backend, backend)
        for agent in agents:
            self.assertIs(
                agent.question_policy,
                ablations.CONSERVATIVE_EARLY_OTHER_POLICY,
            )
            self.assertIs(
                agent.fusion_policy,
                ablations.COMPLETENESS_ADAPTIVE_RRF_POLICY,
            )
        self.assertEqual(backend.calls, 11)
        self.assertEqual(
            payload["comparison"]["scenario_hit_count_delta"],
            {"buying": 1},
        )
        self.assertEqual(
            payload["comparison"]["metric_delta"],
            {
                "hit_rate_at_10": 1.0,
                "mrr": 1.0,
                "mttc": -10.0,
                "efficiency": 1.0,
                "recommended_technical_score": 1.0,
            },
        )
        self.assertEqual(
            payload["variants"]["fused_only"]["route_health"],
            {
                "bm25": {"ok": 10},
                "dense": {"ok": 10},
                "fallback_turns": 0,
            },
        )
        self.assertEqual(
            payload["variants"]["stage_a"]["ranking_health"]["attempts"],
            1,
        )
        self.assertEqual(
            payload["variants"]["stage_a"]["respond_latency_ms"]["count"],
            1,
        )
        self.assertEqual(
            payload["determinism"],
            {
                "requested": False,
                "stage_a_evaluator_payload_equal": None,
                "repeat_diagnostics": None,
            },
        )
        serialized = json.dumps(payload)
        for forbidden in (
            "SECRET_SAMPLE",
            "SECRET_SESSION",
            "SECRET_MESSAGE",
            "SECRET_PROFILE",
            "TARGET",
            "ground_truth",
            "sample_id",
            '"sessions"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_determinism_replay_is_opt_in_and_compares_full_payload(self) -> None:
        payload, agents, backend, evaluate_calls = _run_with_fakes(True)

        self.assertEqual(evaluate_calls, 3)
        self.assertEqual(
            payload["run_order"],
            ["fused_only", "stage_a", "stage_a"],
        )
        self.assertEqual(
            [agent.ranking_policy.value for agent in agents[-2:]],
            ["stage_a", "stage_a"],
        )
        self.assertIs(agents[-1].retriever._backend, backend)
        self.assertTrue(
            payload["determinism"]["stage_a_evaluator_payload_equal"]
        )
        self.assertEqual(
            payload["determinism"]["repeat_diagnostics"]
            ["respond_latency_ms"]["count"],
            1,
        )

    def test_output_path_protects_inputs_and_source_files(self) -> None:
        catalog = Path("catalog.jsonl").resolve()
        dataset = Path("public.jsonl").resolve()
        with self.assertRaisesRegex(ValueError, "input or source"):
            ablations._validate_output_path(catalog, catalog, dataset)
        with self.assertRaisesRegex(ValueError, "input or source"):
            ablations._validate_output_path(
                Path(ablations.__file__).resolve(),
                catalog,
                dataset,
            )
        ablations._validate_output_path(
            Path("results-phase4.json").resolve(),
            catalog,
            dataset,
        )

    def test_cli_determinism_flag_defaults_off_and_uses_atomic_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            arguments = [
                "run_reranking_ablations",
                "--catalog",
                str(root / "catalog.jsonl"),
                "--dataset",
                str(root / "public.jsonl"),
                "--output",
                str(output),
            ]
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch.object(
                    ablations,
                    "run_reranking_ablations",
                    return_value={"safe": True},
                ) as run,
                mock.patch.object(ablations, "_write_json_atomic") as write,
            ):
                ablations.main()

            self.assertFalse(run.call_args.kwargs["verify_determinism"])
            write.assert_called_once_with(output.resolve(), {"safe": True})


if __name__ == "__main__":
    unittest.main()
