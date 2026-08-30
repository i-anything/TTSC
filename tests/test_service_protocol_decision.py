from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from conversational_search.decision import PROTOCOL_UTILITY_DECISION_POLICY
from conversational_search.decision_policy import PROTECTED_DECISION_POLICY
from conversational_search.protocol import DisclosureCard, ProductProtocolEvidence
from conversational_search.ranking import CandidateDocument
from conversational_search.retrieval import (
    PROTOCOL_EVIDENCE_CAPABILITY,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights


PRODUCT_IDS = ("BLUE", "RED", "GREEN", "BLACK")
BM25_IDS = ("GREEN", "RED", "BLUE", "BLACK")


def _protocol_evidence(parent_asin: str) -> ProductProtocolEvidence:
    color = parent_asin.casefold()
    return ProductProtocolEvidence(
        parent_asin=parent_asin,
        coarse_category="Shoes",
        card=DisclosureCard(
            target_category=f"{color.title()} shoe",
            hard_constraints=("rubber sole", f"color: {color}"),
            soft_preferences=(),
        ),
        text=f"Shoes rubber sole {color}",
    )


class _ProtocolRetriever:
    """Small deterministic double; it never reads a catalog or loads a model."""

    def __init__(
        self,
        *,
        capability_error: bool = False,
        evidence_error: bool = False,
        bm25_status: str = "ok",
    ) -> None:
        self._capability_error = capability_error
        self._evidence_error = evidence_error
        self._bm25_status = bm25_status
        self.protocol_accesses: list[str] = []
        self.search_calls = 0
        self.dense_requests: list[bool] = []
        self._evidence = {
            parent_asin: _protocol_evidence(parent_asin)
            for parent_asin in PRODUCT_IDS
        }

    @property
    def protocol_evidence_capability(self) -> object:
        self.protocol_accesses.append("capability")
        if self._capability_error:
            raise RuntimeError("protocol capability unavailable")
        return PROTOCOL_EVIDENCE_CAPABILITY

    def protocol_exact_candidates(
        self,
        category: str,
        constraints: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        self.protocol_accesses.append("exact")
        if limit < 1:
            return ()
        return PRODUCT_IDS[:limit]

    def protocol_exact_constraint_count(
        self,
        category: str,
        constraints: tuple[str, ...],
    ) -> int:
        self.protocol_accesses.append("constraint_count")
        return len(constraints)

    def protocol_category_exists(self, category: str) -> bool:
        self.protocol_accesses.append("category")
        return " ".join(category.split()).casefold() == "shoes"

    def candidate_protocol_evidence(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[ProductProtocolEvidence, ...]:
        self.protocol_accesses.append("evidence")
        if self._evidence_error:
            raise RuntimeError("protocol evidence unavailable")
        return tuple(self._evidence[parent_asin] for parent_asin in parent_asins)

    def search_with_trace(
        self,
        dense_query: str,
        lexical_query: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
        **_: object,
    ) -> RetrievalResult:
        self.search_calls += 1
        use_dense = bool(_.get("use_dense", True))
        structural_support = _.get("bm25_only_support_ids")
        if self._bm25_status != "ok":
            use_dense = True
        elif structural_support is not None:
            use_dense = use_dense or not bool(
                set(BM25_IDS).intersection(structural_support)
            )
        self.dense_requests.append(use_dense)
        fused_ids = PRODUCT_IDS if use_dense else BM25_IDS
        bm25_ids = BM25_IDS if self._bm25_status == "ok" else ()
        return RetrievalResult(
            recommendations=fused_ids[:top_k],
            trace=RetrievalTrace(
                bm25_ids=bm25_ids,
                dense_ids=PRODUCT_IDS if use_dense else (),
                fused_ids=fused_ids,
                bm25_status=self._bm25_status,
                dense_status="ok" if use_dense else "skipped",
                used_fallback=False,
            ),
        )

    def candidate_documents(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[CandidateDocument, ...]:
        return tuple(
            CandidateDocument(parent_asin, self._evidence[parent_asin].text)
            for parent_asin in parent_asins
        )


def _agent(
    retriever: _ProtocolRetriever,
    *,
    candidate: bool,
) -> ConversationalSearchAgent:
    kwargs = (
        {"decision_policy": PROTOCOL_UTILITY_DECISION_POLICY}
        if candidate
        else {}
    )
    return ConversationalSearchAgent(
        "unused.jsonl",
        retriever=retriever,
        **kwargs,
    )


class ServiceProtocolDecisionTest(unittest.TestCase):
    def test_protected_service_import_does_not_load_candidate_modules(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            catalog.write_text(
                '{"parent_asin":"ONLY","title":"Only product"}\n',
                encoding="utf-8",
            )
            missing = Path(directory) / "missing"
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; from conversational_search.service import "
                        "ConversationalSearchAgent; from pathlib import Path; "
                        f"ConversationalSearchAgent({str(catalog)!r}, "
                        f"model_assets=Path({str(missing)!r}), "
                        f"dense_index_path=Path({str(missing)!r})); "
                        "blocked = ('conversational_search.protocol', "
                        "'conversational_search.exact_evidence', "
                        "'conversational_search.utility_planner'); "
                        "assert not any(name in sys.modules for name in blocked)"
                    ),
                ],
                cwd=repository_root,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_protected_default_never_touches_protocol_backend(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=False)
        self.assertIs(agent.decision_policy, PROTECTED_DECISION_POLICY)
        self.assertNotIn("_decision_policy", vars(agent))
        self.assertFalse(
            any(name.startswith("_protocol_") for name in vars(agent))
        )
        agent.reset("protected", {})

        response = agent.respond(
            "protected",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )

        self.assertEqual(len(response["recommendations"]), 3)
        self.assertEqual(response["ask_attribute"], "feature")
        self.assertEqual(retriever.protocol_accesses, [])
        self.assertIsNone(agent.last_action_trace("protected"))

    def test_candidate_jointly_selects_informative_question_and_width(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("candidate", {})

        response = agent.respond(
            "candidate",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )

        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(len(response["recommendations"]), 1)
        self.assertEqual(
            retriever.protocol_accesses,
            [
                "capability",
                "category",
                "constraint_count",
                "exact",
                "evidence",
                "evidence",
            ],
        )
        self.assertEqual(retriever.dense_requests, [False])
        self.assertEqual(agent.protocol_decision_health["applied"], 1)
        trace = agent.last_action_trace("candidate")
        self.assertEqual(trace["protocol_mode"], "applied")
        self.assertEqual(
            trace["dense_policy"],
            "protocol-bm25-first-structural-gate-v2",
        )
        self.assertEqual(trace["dense_status"], "skipped")
        self.assertEqual(trace["derived_track"], "bm25_only")
        self.assertEqual(
            trace["bm25_only_conditions"],
            {
                "message_is_exact_protocol": True,
                "session_state_is_consistent": True,
                "category_is_exactly_recognized": True,
                "exact_product_constraints": 1,
                "no_unparsed_or_free_text_requirement": True,
                "no_tentative_override_or_contradiction": True,
                "bm25_contains_structurally_valid_candidate": True,
            },
        )
        self.assertEqual(trace["presented_width"], 1)
        self.assertNotIn("BLUE", repr(trace))
        self.assertNotIn("rubber sole", repr(trace))

    def test_unsupported_message_fails_open_without_protocol_queries(self) -> None:
        protected_retriever = _ProtocolRetriever()
        candidate_retriever = _ProtocolRetriever()
        protected = _agent(protected_retriever, candidate=False)
        candidate = _agent(candidate_retriever, candidate=True)
        protected.reset("protected", {})
        candidate.reset("candidate", {})
        message = "Show me some Shoes; I'm open to options."

        expected = protected.respond("protected", message, 1, 3)
        actual = candidate.respond("candidate", message, 1, 3)

        self.assertEqual(actual, expected)
        self.assertEqual(candidate_retriever.protocol_accesses, [])
        self.assertEqual(candidate_retriever.dense_requests, [True])
        trace = candidate.last_action_trace("candidate")
        self.assertEqual(trace["protocol_mode"], "free_form_fail_open")
        self.assertEqual(trace["dense_policy"], "dense-always-v1")
        self.assertFalse(
            trace["bm25_only_conditions"]["message_is_exact_protocol"]
        )
        self.assertEqual(
            candidate.protocol_decision_health["unsupported_or_disabled"],
            1,
        )

    def test_capability_error_fails_open_to_protected_response(self) -> None:
        protected_retriever = _ProtocolRetriever()
        candidate_retriever = _ProtocolRetriever(capability_error=True)
        protected = _agent(protected_retriever, candidate=False)
        candidate = _agent(candidate_retriever, candidate=True)
        protected.reset("protected", {})
        candidate.reset("candidate", {})
        message = "I'm looking for Shoes, but I'm still exploring."

        expected = protected.respond("protected", message, 1, 3)
        actual = candidate.respond("candidate", message, 1, 3)

        self.assertEqual(actual, expected)
        self.assertEqual(candidate_retriever.protocol_accesses, ["capability"])
        self.assertEqual(
            candidate.protocol_decision_health["capability_unavailable"],
            1,
        )

    def test_pre_skip_protocol_failure_fails_open_to_hybrid_base_order(self) -> None:
        retriever = _ProtocolRetriever(evidence_error=True)
        agent = _agent(retriever, candidate=True)
        agent.reset("post-skip-failure", {})

        response = agent.respond(
            "post-skip-failure",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )

        self.assertEqual(retriever.dense_requests, [True])
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            list(PRODUCT_IDS[:3]),
        )
        trace = agent.last_action_trace("post-skip-failure")
        self.assertEqual(trace["dense_status"], "ok")
        self.assertEqual(trace["derived_track"], "hybrid")
        self.assertEqual(trace["planner_outcome"], "candidate_or_evidence_error")
        self.assertEqual(trace["presented_width"], 3)

    def test_zero_structural_support_runs_dense_and_preserves_hybrid(self) -> None:
        retriever = _ProtocolRetriever()
        retriever._evidence = {
            parent_asin: ProductProtocolEvidence(
                parent_asin=parent_asin,
                coarse_category="Shoes",
                card=DisclosureCard(
                    target_category=f"{parent_asin.title()} shoe",
                    hard_constraints=("different requirement",),
                    soft_preferences=(),
                ),
                text="Shoes different requirement",
            )
            for parent_asin in PRODUCT_IDS
        }
        agent = _agent(retriever, candidate=True)
        agent.reset("zero-support", {})

        agent.respond(
            "zero-support",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )

        self.assertEqual(retriever.dense_requests, [True])
        trace = agent.last_action_trace("zero-support")
        self.assertEqual(trace["derived_track"], "hybrid")
        self.assertFalse(
            trace["bm25_only_conditions"][
                "bm25_contains_structurally_valid_candidate"
            ]
        )

    def test_exact_constraint_count_is_independent_of_joint_candidates(self) -> None:
        class NoJointCandidateRetriever(_ProtocolRetriever):
            def protocol_exact_candidates(
                self,
                category: str,
                constraints: tuple[str, ...],
                *,
                limit: int,
            ) -> tuple[str, ...]:
                self.protocol_accesses.append("exact")
                return ()

        retriever = NoJointCandidateRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("no-joint-candidate", {})

        agent.respond(
            "no-joint-candidate",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )

        self.assertEqual(retriever.dense_requests, [True])
        trace = agent.last_action_trace("no-joint-candidate")
        self.assertEqual(
            trace["bm25_only_conditions"]["exact_product_constraints"],
            1,
        )
        self.assertFalse(
            trace["bm25_only_conditions"][
                "bm25_contains_structurally_valid_candidate"
            ]
        )

    def test_overlong_exact_protocol_value_forces_hybrid(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("overlong", {})
        message = (
            "I'm looking for Shoes. A key requirement is: "
            + "x" * 1_025
            + "."
        )

        agent.respond("overlong", message, 1, 3)

        self.assertEqual(retriever.dense_requests, [True])
        trace = agent.last_action_trace("overlong")
        self.assertFalse(
            trace["bm25_only_conditions"]["message_is_exact_protocol"]
        )

    def test_bm25_failure_states_use_one_dense_rescue(self) -> None:
        for status in ("unavailable", "error", "empty"):
            with self.subTest(status=status):
                retriever = _ProtocolRetriever(bm25_status=status)
                agent = _agent(retriever, candidate=True)
                session_id = f"bm25-{status}"
                agent.reset(session_id, {})

                agent.respond(
                    session_id,
                    "I'm looking for Shoes. "
                    "A key requirement is: rubber sole.",
                    1,
                    3,
                )

                self.assertEqual(retriever.search_calls, 1)
                self.assertEqual(retriever.dense_requests, [True])
                trace = agent.last_action_trace(session_id)
                self.assertEqual(trace["bm25_status"], status)
                self.assertEqual(trace["dense_status"], "ok")
                self.assertEqual(trace["derived_track"], "dense_rescue")

    def test_exact_category_requires_case_and_whitespace_only_match(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("category", {})

        agent.respond(
            "category",
            "I'm looking for Shoe. A key requirement is: rubber sole.",
            1,
            3,
        )

        self.assertEqual(retriever.dense_requests, [True])
        trace = agent.last_action_trace("category")
        self.assertFalse(
            trace["bm25_only_conditions"]["category_is_exactly_recognized"]
        )

    def test_tentative_session_stays_hybrid_until_reset(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("tentative-sticky", {})

        agent.respond(
            "tentative-sticky",
            "I'm looking for Shoes. classic",
            1,
            3,
        )
        agent.respond(
            "tentative-sticky",
            "Actually, ignore my earlier preference. "
            "What I need is: rubber sole.",
            2,
            3,
        )

        self.assertEqual(retriever.dense_requests, [True, True])
        trace = agent.last_action_trace("tentative-sticky")
        self.assertFalse(
            trace["bm25_only_conditions"][
                "no_tentative_override_or_contradiction"
            ]
        )

        agent.reset("tentative-sticky", {})
        agent.respond(
            "tentative-sticky",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )
        self.assertEqual(retriever.dense_requests[-1], False)

    def test_tentative_fail_open_outputs_are_remembered_after_override(self) -> None:
        retriever = _ProtocolRetriever(evidence_error=True)
        agent = _agent(retriever, candidate=True)
        agent.reset("tentative-memory", {})

        first = agent.respond(
            "tentative-memory",
            "I'm looking for Shoes. classic",
            1,
            3,
        )
        retriever._evidence_error = False
        second = agent.respond(
            "tentative-memory",
            "Actually, ignore my earlier preference. "
            "What I need is: rubber sole.",
            2,
            3,
        )

        first_ids = {
            item["parent_asin"] for item in first["recommendations"]
        }
        second_ids = {
            item["parent_asin"] for item in second["recommendations"]
        }
        self.assertTrue(first_ids)
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_any_override_switches_the_remaining_session_to_hybrid(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("override-sticky", {})

        agent.respond(
            "override-sticky",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )
        agent.respond(
            "override-sticky",
            "Actually, ignore my earlier preference. "
            "What I need is: rubber sole.",
            2,
            3,
        )
        agent.respond(
            "override-sticky",
            "For that, what matters is: color: blue.",
            3,
            3,
        )

        self.assertEqual(retriever.dense_requests, [False, True, True])
        trace = agent.last_action_trace("override-sticky")
        self.assertEqual(trace["derived_track"], "hybrid")
        self.assertFalse(
            trace["bm25_only_conditions"][
                "no_tentative_override_or_contradiction"
            ]
        )

    def test_candidate_module_import_failure_fails_open_to_protected_response(
        self,
    ) -> None:
        protected_retriever = _ProtocolRetriever()
        candidate_retriever = _ProtocolRetriever()
        protected = _agent(protected_retriever, candidate=False)
        candidate = _agent(candidate_retriever, candidate=True)
        protected.reset("protected", {})
        candidate.reset("candidate", {})
        message = "I'm looking for Shoes, but I'm still exploring."
        expected = protected.respond("protected", message, 1, 3)
        real_import = __import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "conversational_search.decision":
                raise ImportError("candidate decision import failure")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            actual = candidate.respond("candidate", message, 1, 3)

        self.assertEqual(actual, expected)
        self.assertEqual(candidate_retriever.protocol_accesses, [])
        self.assertEqual(
            candidate.protocol_decision_health["candidate_or_evidence_error"],
            1,
        )

    def test_tentative_override_interval_allows_zero_width(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("override", {})

        response = agent.respond(
            "override",
            "I'm looking for Shoes. classic",
            1,
            3,
        )

        self.assertEqual(response["recommendations"], [])
        self.assertEqual(response["ask_attribute"], "other")
        self.assertEqual(agent.protocol_decision_health["applied"], 1)

    def test_override_lock_survives_same_value_answer_deduplication(self) -> None:
        retriever = _ProtocolRetriever()
        retriever._evidence = {
            parent_asin: ProductProtocolEvidence(
                parent_asin=parent_asin,
                coarse_category="Shoes",
                card=DisclosureCard(
                    target_category=f"{parent_asin.title()} shoe",
                    hard_constraints=(
                        "rubber sole",
                        f"budget around ${index}0",
                    ),
                    soft_preferences=(f"color: {parent_asin.casefold()}",),
                ),
                text=f"Shoes rubber sole budget {index}0",
                price=f"{index}0",
            )
            for index, parent_asin in enumerate(PRODUCT_IDS, start=1)
        }
        agent = _agent(retriever, candidate=True)
        agent.reset("sticky-override", {})

        first = agent.respond(
            "sticky-override",
            "I'm looking for Shoes. budget around $10",
            1,
            3,
        )
        self.assertEqual(first["recommendations"], [])
        self.assertEqual(first["ask_attribute"], "other")

        second = agent.respond(
            "sticky-override",
            "For that, what matters is: rubber sole; budget around $10.",
            2,
            3,
        )

        self.assertEqual(second["recommendations"], [])
        self.assertTrue(
            agent.protocol_decision_health["applied"] >= 2
        )

    def test_impossible_no_additional_reply_fails_open_at_full_width(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("negative", {})
        first = agent.respond(
            "negative",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )
        self.assertEqual(first["ask_attribute"], "other")

        second = agent.respond(
            "negative",
            "I don't have an additional preference for other.",
            2,
            3,
        )

        self.assertEqual(len(second["recommendations"]), 3)
        self.assertEqual(
            agent.protocol_decision_health["fail_open_evidence"],
            1,
        )

    def test_surviving_session_never_repeats_a_protocol_shown_candidate(
        self,
    ) -> None:
        retriever = _ProtocolRetriever()
        colors = {
            "BLUE": "blue",
            "RED": "blue",
            "GREEN": "green",
            "BLACK": "black",
        }
        retriever._evidence = {
            parent_asin: ProductProtocolEvidence(
                parent_asin=parent_asin,
                coarse_category="Shoes",
                card=DisclosureCard(
                    target_category=f"{parent_asin.title()} shoe",
                    hard_constraints=(
                        "rubber sole",
                        f"color: {colors[parent_asin]}",
                    ),
                    soft_preferences=(),
                ),
                text=f"Shoes rubber sole {colors[parent_asin]}",
            )
            for parent_asin in PRODUCT_IDS
        }
        agent = _agent(retriever, candidate=True)
        agent.reset("no-repeat", {})

        first = agent.respond(
            "no-repeat",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )
        second = agent.respond(
            "no-repeat",
            "For that, what matters is: color: blue.",
            2,
            3,
        )

        first_ids = [item["parent_asin"] for item in first["recommendations"]]
        second_ids = [item["parent_asin"] for item in second["recommendations"]]
        self.assertEqual(first_ids, ["GREEN"])
        self.assertNotIn("GREEN", second_ids)
        self.assertEqual(second_ids[0], "RED")

    def test_exact_rescue_never_displaces_the_protected_candidate_pool(self) -> None:
        base_ids = tuple(f"BASE{index:03d}" for index in range(200))

        class ProtectedPoolRetriever:
            def __init__(self) -> None:
                self.protocol_request: tuple[str, ...] = ()
                self.dense_ran = False
                self.evidence = {
                    parent_asin: ProductProtocolEvidence(
                        parent_asin=parent_asin,
                        coarse_category="Shoes",
                        card=DisclosureCard(
                            target_category=f"{parent_asin} shoe",
                            hard_constraints=("rubber sole",),
                            soft_preferences=(),
                        ),
                        text="Shoes rubber sole",
                    )
                    for parent_asin in (*base_ids, "RESCUE")
                }

            @property
            def protocol_evidence_capability(self) -> object:
                return PROTOCOL_EVIDENCE_CAPABILITY

            def protocol_exact_candidates(
                self,
                category: str,
                constraints: tuple[str, ...],
                *,
                limit: int,
            ) -> tuple[str, ...]:
                return ("RESCUE",)

            def protocol_exact_constraint_count(
                self,
                category: str,
                constraints: tuple[str, ...],
            ) -> int:
                return len(constraints)

            def protocol_category_exists(self, category: str) -> bool:
                return category.casefold() == "shoes"

            def candidate_protocol_evidence(
                self,
                parent_asins: tuple[str, ...],
            ) -> tuple[ProductProtocolEvidence, ...]:
                self.protocol_request = parent_asins
                return tuple(self.evidence[value] for value in parent_asins)

            def search_with_trace(
                self,
                dense_query: str,
                lexical_query: str,
                top_k: int,
                *,
                route_weights: RouteWeights,
                **_: object,
            ) -> RetrievalResult:
                use_dense = bool(_.get("use_dense", True))
                structural_support = _.get("bm25_only_support_ids")
                if structural_support is not None:
                    use_dense = use_dense or not bool(
                        set(base_ids).intersection(structural_support)
                    )
                self.dense_ran = use_dense
                return RetrievalResult(
                    recommendations=base_ids,
                    trace=RetrievalTrace(
                        bm25_ids=base_ids,
                        dense_ids=base_ids if use_dense else (),
                        fused_ids=base_ids,
                        bm25_status="ok",
                        dense_status="ok" if use_dense else "skipped",
                        used_fallback=False,
                    ),
                )

            def candidate_documents(
                self,
                parent_asins: tuple[str, ...],
            ) -> tuple[CandidateDocument, ...]:
                return tuple(
                    CandidateDocument(value, "Shoes rubber sole")
                    for value in parent_asins
                )

        retriever = ProtectedPoolRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            decision_policy=PROTOCOL_UTILITY_DECISION_POLICY,
        )
        agent.reset("protected-union", {})

        agent.respond(
            "protected-union",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            3,
        )

        self.assertEqual(len(retriever.protocol_request), len(base_ids))
        self.assertEqual(set(retriever.protocol_request), set(base_ids))
        self.assertNotIn("RESCUE", retriever.protocol_request)
        self.assertTrue(retriever.dense_ran)

    def test_final_turn_uses_full_available_width_and_asks_nothing(self) -> None:
        retriever = _ProtocolRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("final", {})
        first = agent.respond(
            "final",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(len(first["recommendations"]), 1)

        final = agent.respond(
            "final",
            "For that, what matters is: rubber sole; color: black.",
            10,
            3,
        )

        self.assertEqual(final["ask_attribute"], None)
        self.assertEqual(len(final["recommendations"]), 3)
        self.assertEqual(agent.protocol_decision_health["applied"], 2)


if __name__ == "__main__":
    unittest.main()
