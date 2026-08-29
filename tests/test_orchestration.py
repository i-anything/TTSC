from __future__ import annotations

import dataclasses
import unittest
from collections.abc import Mapping
from dataclasses import replace
from enum import Enum

from conversational_search.intent import IntentState, Requirement
from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    BackendSnapshotToken,
    DEFAULT_RANKING_CACHE_CAPACITY,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    MAX_CACHED_RANKED_IDS,
    MAX_CACHED_ID_CHARACTERS,
    OrchestrationPlanner,
    OrchestrationPolicy,
    QueryAction,
    TurnDecision,
)
from conversational_search.ranking import (
    FUSED_ONLY_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    RankingPolicy,
)
from conversational_search.strategy import RouteWeights, intent_completeness


_SNAPSHOT = BackendSnapshotToken()
_WEIGHTS = RouteWeights(bm25=0.5, dense=0.5)
_DENSE_QUERY = (
    "Category: synthetic shoes\nAttributes: Material: synthetic leather"
)
_LEXICAL_QUERY = "synthetic shoes synthetic leather"
_RANKED_IDS = ("P0001", "P0002", "P0003")
_BASE_STATE = IntentState(
    category="synthetic shoes",
    requirements=(
        Requirement("synthetic leather", "answer", 2, "material"),
        Requirement("synthetic commute", "free_text", 3, "use_case"),
    ),
    excluded=("synthetic suede", "synthetic plastic"),
    last_turn=3,
)


def _decide(
    planner: OrchestrationPlanner,
    *,
    session_id: str = "session",
    state: IntentState = _BASE_STATE,
    dense_query: str = _DENSE_QUERY,
    lexical_query: str = _LEXICAL_QUERY,
    route_weights: RouteWeights = _WEIGHTS,
    ranking_policy: RankingPolicy = STAGE_A_RANKING_POLICY,
    result_count: int = 10,
    backend_snapshot_token: BackendSnapshotToken | None = _SNAPSHOT,
    cache_eligible: bool = True,
) -> TurnDecision:
    return planner.decide(
        session_id,
        state,
        dense_query,
        lexical_query,
        route_weights,
        ranking_policy,
        result_count,
        backend_snapshot_token,
        cache_eligible,
    )


def _prime(
    planner: OrchestrationPlanner,
    *,
    session_id: str = "session",
    state: IntentState = _BASE_STATE,
    dense_query: str = _DENSE_QUERY,
    lexical_query: str = _LEXICAL_QUERY,
    route_weights: RouteWeights = _WEIGHTS,
    ranking_policy: RankingPolicy = STAGE_A_RANKING_POLICY,
    result_count: int = 10,
    backend_snapshot_token: BackendSnapshotToken = _SNAPSHOT,
    ranked_ids: tuple[str, ...] = _RANKED_IDS,
) -> TurnDecision:
    decision = _decide(
        planner,
        session_id=session_id,
        state=state,
        dense_query=dense_query,
        lexical_query=lexical_query,
        route_weights=route_weights,
        ranking_policy=ranking_policy,
        result_count=result_count,
        backend_snapshot_token=backend_snapshot_token,
    )
    if decision.action is not QueryAction.SEARCH:
        raise AssertionError(f"expected cold SEARCH, got {decision.action!r}")
    if not planner.commit(
        session_id,
        decision,
        backend_snapshot_token,
        ranked_ids,
    ):
        raise AssertionError("expected complete synthetic ranking to be cached")
    return decision


def _retained_scalars(root: object) -> tuple[tuple[str, ...], tuple[bytes, ...]]:
    """Recursively inspect instance state without following classes or modules."""

    strings: list[str] = []
    byte_strings: list[bytes] = []
    seen: set[int] = set()

    def visit(value: object) -> None:
        if isinstance(value, str):
            strings.append(value)
            return
        if isinstance(value, bytes):
            byte_strings.append(value)
            return
        if value is None or isinstance(value, (bool, int, float)):
            return
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, Enum):
            visit(value.value)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name))
            return
        instance_dict = getattr(value, "__dict__", None)
        if isinstance(instance_dict, dict):
            for item in instance_dict.values():
                visit(item)
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot not in {"__dict__", "__weakref__"} and hasattr(
                    value, slot
                ):
                    visit(getattr(value, slot))

    visit(root)
    return tuple(strings), tuple(byte_strings)


class OrchestrationDecisionTests(unittest.TestCase):
    def test_policy_and_action_enums_are_explicit_and_stable(self) -> None:
        self.assertIs(
            ALWAYS_SEARCH_ORCHESTRATION_POLICY,
            OrchestrationPolicy.ALWAYS_SEARCH,
        )
        self.assertIs(
            EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            OrchestrationPolicy.EXACT_RANKING_REUSE,
        )
        self.assertEqual(QueryAction.SEARCH.value, "search")
        self.assertEqual(QueryAction.REUSE.value, "reuse")
        self.assertEqual(QueryAction.SKIP.value, "skip")

    def test_cold_search_then_exact_reuse_preserves_full_ranked_pool(self) -> None:
        planner = OrchestrationPlanner()

        search = _prime(planner)
        reuse = _decide(planner)

        self.assertIs(search.action, QueryAction.SEARCH)
        self.assertEqual(search.reason, "cold_cache")
        self.assertIsInstance(search.dependency_digest, bytes)
        self.assertEqual(len(search.dependency_digest or b""), 32)
        self.assertEqual(search.cached_ranked_ids, ())
        self.assertEqual(search.recomputation_value, 1.0)
        self.assertEqual(
            search.intent_evidence_completeness,
            intent_completeness(_BASE_STATE),
        )
        self.assertIs(reuse.action, QueryAction.REUSE)
        self.assertEqual(reuse.reason, "exact_dependency_hit")
        self.assertEqual(reuse.dependency_digest, search.dependency_digest)
        self.assertEqual(reuse.cached_ranked_ids, _RANKED_IDS)
        self.assertEqual(reuse.recomputation_value, 0.0)

        health = planner.health
        self.assertEqual(health["decisions"], 2)
        self.assertEqual(health["searches"], 1)
        self.assertEqual(health["reuses"], 1)
        self.assertEqual(health["skips"], 0)
        self.assertEqual(health["lookups"], 2)
        self.assertEqual(health["hits"], 1)
        self.assertEqual(health["cold_misses"], 1)
        self.assertEqual(health["dependency_misses"], 0)
        self.assertEqual(health["stores"], 1)
        self.assertEqual(health["entries"], 1)
        self.assertEqual(health["cached_id_references"], len(_RANKED_IDS))
        self.assertEqual(health["retrievals_avoided"], 1)
        self.assertEqual(health["reranks_avoided"], 1)

    def test_always_search_policy_never_reads_or_populates_cache(self) -> None:
        planner = OrchestrationPlanner(policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY)

        first = _decide(planner)
        self.assertIs(first.action, QueryAction.SEARCH)
        self.assertEqual(first.reason, "policy_requires_search")
        self.assertFalse(planner.commit("session", first, _SNAPSHOT, _RANKED_IDS))
        second = _decide(planner, result_count=0)

        self.assertIs(second.action, QueryAction.SEARCH)
        self.assertEqual(planner.health["lookups"], 0)
        self.assertEqual(planner.health["hits"], 0)
        self.assertEqual(planner.health["stores"], 0)
        self.assertEqual(planner.health["entries"], 0)

    def test_non_positive_result_count_skips_without_touching_cache(self) -> None:
        planner = OrchestrationPlanner()
        _prime(planner)
        before = planner.health

        for result_count in (0, -1, -100):
            with self.subTest(result_count=result_count):
                decision = _decide(planner, result_count=result_count)
                self.assertIs(decision.action, QueryAction.SKIP)
                self.assertEqual(decision.reason, "empty_result_request")
                self.assertEqual(decision.recomputation_value, 0.0)
                self.assertIsNone(decision.dependency_digest)
                self.assertEqual(decision.cached_ranked_ids, ())

        after = planner.health
        for key in (
            "lookups",
            "hits",
            "cold_misses",
            "dependency_misses",
            "stores",
            "entries",
            "cached_id_references",
        ):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(after["skips"], before["skips"] + 3)
        self.assertIs(_decide(planner).action, QueryAction.REUSE)

    def test_positive_result_count_is_not_a_ranking_dependency(self) -> None:
        planner = OrchestrationPlanner()
        _prime(planner, result_count=10)

        for result_count in (1, 5, 11, 10_000):
            with self.subTest(result_count=result_count):
                decision = _decide(planner, result_count=result_count)
                self.assertIs(decision.action, QueryAction.REUSE)
                self.assertEqual(decision.cached_ranked_ids, _RANKED_IDS)


class RankingDependencyTests(unittest.TestCase):
    def test_every_predeclared_ranking_dependency_forces_search(self) -> None:
        requirements = _BASE_STATE.requirements
        cases = {
            "category": {
                "state": replace(_BASE_STATE, category="different category")
            },
            "requirement value": {
                "state": replace(
                    _BASE_STATE,
                    requirements=(
                        replace(requirements[0], value="different material"),
                        requirements[1],
                    ),
                )
            },
            "requirement source": {
                "state": replace(
                    _BASE_STATE,
                    requirements=(
                        replace(requirements[0], source="override"),
                        requirements[1],
                    ),
                )
            },
            "requirement attribute": {
                "state": replace(
                    _BASE_STATE,
                    requirements=(
                        replace(requirements[0], attribute="feature"),
                        requirements[1],
                    ),
                )
            },
            "requirement order": {
                "state": replace(_BASE_STATE, requirements=requirements[::-1])
            },
            "exclusion value": {
                "state": replace(_BASE_STATE, excluded=("different exclusion",))
            },
            "exclusion order": {
                "state": replace(
                    _BASE_STATE,
                    excluded=_BASE_STATE.excluded[::-1],
                )
            },
            "dense query": {"dense_query": _DENSE_QUERY + " changed"},
            "lexical query": {"lexical_query": _LEXICAL_QUERY + " changed"},
            "exact route weights": {
                "route_weights": RouteWeights(bm25=0.4, dense=0.6)
            },
            "ranking policy": {"ranking_policy": FUSED_ONLY_RANKING_POLICY},
        }

        for dependency, changes in cases.items():
            with self.subTest(dependency=dependency):
                planner = OrchestrationPlanner()
                original = _prime(planner)
                changed = _decide(planner, **changes)
                self.assertIs(changed.action, QueryAction.SEARCH)
                self.assertEqual(changed.reason, "ranking_dependencies_changed")
                self.assertNotEqual(
                    changed.dependency_digest,
                    original.dependency_digest,
                )
                self.assertEqual(planner.health["dependency_misses"], 1)

    def test_dialogue_only_fields_and_requirement_turn_are_excluded(self) -> None:
        planner = OrchestrationPlanner()
        search = _prime(planner)
        dialogue_only_change = replace(
            _BASE_STATE,
            requirements=tuple(
                replace(requirement, turn=requirement.turn + 100)
                for requirement in _BASE_STATE.requirements
            ),
            no_preference=frozenset({"color", "brand"}),
            asked_attributes=("feature", "material", "color"),
            last_asked_attribute="color",
            intent_version=91,
            last_turn=103,
        )

        decision = _decide(
            planner,
            state=dialogue_only_change,
            result_count=1,
        )

        self.assertIs(decision.action, QueryAction.REUSE)
        self.assertEqual(decision.dependency_digest, search.dependency_digest)
        self.assertEqual(decision.cached_ranked_ids, _RANKED_IDS)

    def test_digest_is_deterministic_across_independent_planners(self) -> None:
        first = _decide(OrchestrationPlanner())
        second = _decide(OrchestrationPlanner())

        self.assertEqual(first.dependency_digest, second.dependency_digest)
        self.assertIsInstance(first.dependency_digest, bytes)
        self.assertEqual(len(first.dependency_digest or b""), 32)

    def test_route_weight_digest_uses_exact_float_values(self) -> None:
        planner = OrchestrationPlanner()
        original_weights = RouteWeights(bm25=0.4, dense=0.6)
        next_float_weights = RouteWeights(
            bm25=0.4000000000000001,
            dense=0.5999999999999999,
        )
        original = _prime(planner, route_weights=original_weights)

        changed = _decide(planner, route_weights=next_float_weights)

        self.assertIs(changed.action, QueryAction.SEARCH)
        self.assertNotEqual(changed.dependency_digest, original.dependency_digest)


class SnapshotAndEligibilityTests(unittest.TestCase):
    class _PayloadToken(BackendSnapshotToken):
        def __init__(self) -> None:
            self.retained_canary = "TOKEN_PAYLOAD_MUST_NOT_BE_RETAINED"

    def test_backend_snapshot_is_compared_by_identity_not_equality(self) -> None:
        planner = OrchestrationPlanner()
        first_token = BackendSnapshotToken()
        equal_but_distinct_token = BackendSnapshotToken()
        self.assertIsNot(first_token, equal_but_distinct_token)
        _prime(planner, backend_snapshot_token=first_token)

        hit = _decide(planner, backend_snapshot_token=first_token)
        invalidated = _decide(
            planner,
            backend_snapshot_token=equal_but_distinct_token,
        )

        self.assertIs(hit.action, QueryAction.REUSE)
        self.assertIs(invalidated.action, QueryAction.SEARCH)
        self.assertEqual(invalidated.reason, "backend_snapshot_changed")
        self.assertEqual(planner.health["backend_invalidations"], 1)
        self.assertEqual(planner.health["entries"], 0)

    def test_unavailable_snapshot_and_ineligible_result_fail_closed(self) -> None:
        cases = (
            ({"backend_snapshot_token": None}, "backend_snapshot_unavailable"),
            ({"backend_snapshot_token": object()}, "backend_snapshot_unavailable"),
            (
                {"backend_snapshot_token": self._PayloadToken()},
                "backend_snapshot_unavailable",
            ),
            ({"cache_eligible": False}, "ranking_not_cache_eligible"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                planner = OrchestrationPlanner()
                decision = _decide(planner, **changes)
                self.assertIs(decision.action, QueryAction.SEARCH)
                self.assertEqual(decision.reason, reason)
                self.assertIsNone(decision.dependency_digest)
                self.assertFalse(
                    planner.commit("session", decision, _SNAPSHOT, _RANKED_IDS)
                )
                self.assertEqual(planner.health["entries"], 0)
                self.assertIs(_decide(planner).action, QueryAction.SEARCH)

    def test_commit_rejects_a_different_snapshot_reference(self) -> None:
        planner = OrchestrationPlanner()
        decision = _decide(planner)

        self.assertFalse(
            planner.commit(
                "session",
                decision,
                BackendSnapshotToken(),
                _RANKED_IDS,
            )
        )
        self.assertEqual(planner.health["stores"], 0)
        self.assertEqual(planner.health["store_rejections"], 1)
        self.assertIs(_decide(planner).action, QueryAction.SEARCH)


class CacheSafetyTests(unittest.TestCase):
    def test_malformed_or_incomplete_ranked_pools_are_never_cached(self) -> None:
        malformed_pools: tuple[object, ...] = (
            (),
            ("P0001", "P0001"),
            ("",),
            (" P0001",),
            ("P0001 ",),
            ("P0001", 2),
            ["P0001"],
            ("P0001雪",),
            ("P" * (MAX_CACHED_ID_CHARACTERS + 1),),
            tuple(f"P{index:04d}" for index in range(MAX_CACHED_RANKED_IDS + 1)),
        )
        for ranked_ids in malformed_pools:
            with self.subTest(kind=type(ranked_ids).__name__, size=len(ranked_ids)):
                planner = OrchestrationPlanner()
                decision = _decide(planner)
                stored = planner.commit(
                    "session",
                    decision,
                    _SNAPSHOT,
                    ranked_ids,  # type: ignore[arg-type]
                )
                self.assertFalse(stored)
                self.assertEqual(planner.health["stores"], 0)
                self.assertEqual(planner.health["store_rejections"], 1)
                self.assertEqual(planner.health["entries"], 0)
                self.assertIs(_decide(planner).action, QueryAction.SEARCH)

    def test_full_two_hundred_id_pool_is_cached_without_truncation(self) -> None:
        planner = OrchestrationPlanner()
        full_pool = tuple(
            f"P{index:04d}" for index in range(1, MAX_CACHED_RANKED_IDS + 1)
        )
        self.assertEqual(len(full_pool), 200)

        _prime(planner, ranked_ids=full_pool)
        reuse = _decide(planner, result_count=1)

        self.assertIs(reuse.action, QueryAction.REUSE)
        self.assertEqual(reuse.cached_ranked_ids, full_pool)
        self.assertEqual(planner.health["cached_id_references"], 200)
        self.assertEqual(planner.health["maximum_ids_per_entry"], 200)

    def test_new_ranking_replaces_the_only_entry_for_that_session(self) -> None:
        planner = OrchestrationPlanner()
        _prime(planner)
        changed_query = _DENSE_QUERY + " with a changed dependency"
        changed = _decide(planner, dense_query=changed_query)
        self.assertIs(changed.action, QueryAction.SEARCH)
        self.assertTrue(
            planner.commit("session", changed, _SNAPSHOT, ("Q0001", "Q0002"))
        )

        current = _decide(planner, dense_query=changed_query)
        stale = _decide(planner)

        self.assertEqual(planner.health["entries"], 1)
        self.assertEqual(planner.health["cached_id_references"], 2)
        self.assertIs(current.action, QueryAction.REUSE)
        self.assertEqual(current.cached_ranked_ids, ("Q0001", "Q0002"))
        self.assertIs(stale.action, QueryAction.SEARCH)

    def test_sessions_are_isolated_and_reset_is_scoped(self) -> None:
        planner = OrchestrationPlanner()
        _prime(planner, session_id="alpha")
        _prime(planner, session_id="beta", ranked_ids=("BETA1", "BETA2"))

        self.assertEqual(
            _decide(planner, session_id="alpha").cached_ranked_ids,
            _RANKED_IDS,
        )
        self.assertEqual(
            _decide(planner, session_id="beta").cached_ranked_ids,
            ("BETA1", "BETA2"),
        )
        planner.reset("alpha")

        self.assertIs(
            _decide(planner, session_id="alpha").action,
            QueryAction.SEARCH,
        )
        beta = _decide(planner, session_id="beta")
        self.assertIs(beta.action, QueryAction.REUSE)
        self.assertEqual(beta.cached_ranked_ids, ("BETA1", "BETA2"))
        planner.reset("not-cached")
        self.assertEqual(planner.health["reset_invalidations"], 1)
        self.assertEqual(planner.health["entries"], 1)

    def test_capacity_uses_deterministic_session_lru_eviction(self) -> None:
        planner = OrchestrationPlanner(capacity=2)
        _prime(planner, session_id="least-recent")
        _prime(planner, session_id="most-recent")
        self.assertIs(
            _decide(planner, session_id="least-recent").action,
            QueryAction.REUSE,
        )
        _prime(planner, session_id="new-session")

        self.assertEqual(planner.health["entries"], 2)
        self.assertEqual(planner.health["capacity_evictions"], 1)
        self.assertIs(
            _decide(planner, session_id="most-recent").action,
            QueryAction.SEARCH,
        )
        self.assertIs(
            _decide(planner, session_id="least-recent").action,
            QueryAction.REUSE,
        )

    def test_capacity_cannot_exceed_frozen_global_bound(self) -> None:
        for capacity in (0, -1, DEFAULT_RANKING_CACHE_CAPACITY + 1):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    OrchestrationPlanner(capacity=capacity)
        for capacity in (True, 1.5, "2"):
            with self.subTest(capacity=capacity):
                with self.assertRaises(TypeError):
                    OrchestrationPlanner(capacity=capacity)  # type: ignore[arg-type]

    def test_maximum_capacity_and_id_width_stay_below_eight_mib(self) -> None:
        planner = OrchestrationPlanner()
        for session_index in range(DEFAULT_RANKING_CACHE_CAPACITY):
            ranked_ids = tuple(
                (
                    f"P{session_index:03d}{item_index:03d}"
                    + "X" * (MAX_CACHED_ID_CHARACTERS - 7)
                )
                for item_index in range(MAX_CACHED_RANKED_IDS)
            )
            _prime(
                planner,
                session_id=f"synthetic-session-{session_index}",
                ranked_ids=ranked_ids,
            )

        health = planner.health
        self.assertEqual(health["entries"], DEFAULT_RANKING_CACHE_CAPACITY)
        self.assertEqual(
            health["cached_id_references"],
            DEFAULT_RANKING_CACHE_CAPACITY * MAX_CACHED_RANKED_IDS,
        )
        self.assertEqual(
            health["cached_id_utf8_bytes"],
            DEFAULT_RANKING_CACHE_CAPACITY
            * MAX_CACHED_RANKED_IDS
            * MAX_CACHED_ID_CHARACTERS,
        )
        self.assertLessEqual(health["retained_cache_bytes"], 8 * 1024 * 1024)

        _prime(
            planner,
            session_id="capacity-plus-one",
            ranked_ids=("CAPACITY_PLUS_ONE",),
        )
        self.assertEqual(planner.health["entries"], DEFAULT_RANKING_CACHE_CAPACITY)
        self.assertEqual(planner.health["capacity_evictions"], 1)

    def test_corrupted_entry_type_is_evicted_and_recomputed(self) -> None:
        planner = OrchestrationPlanner()
        _prime(planner)
        # Intentional white-box injection: persisted state is never trusted.
        session_key = next(iter(planner._entries))
        planner._entries[session_key] = object()  # type: ignore[assignment]

        decision = _decide(planner)

        self.assertIs(decision.action, QueryAction.SEARCH)
        self.assertEqual(decision.reason, "cache_fault")
        self.assertEqual(planner.health["fault_invalidations"], 1)
        self.assertEqual(planner.health["entries"], 0)

    def test_corrupted_entry_payload_is_evicted_and_recomputed(self) -> None:
        corruptions = (
            ("dependency_digest", b"not-a-sha256-digest"),
            ("ranked_ids", ("P0001", "P0001")),
            ("ranked_ids", (" P0001",)),
        )
        for field_name, value in corruptions:
            with self.subTest(field=field_name, value=value):
                planner = OrchestrationPlanner()
                _prime(planner)
                entry = next(iter(planner._entries.values()))
                object.__setattr__(entry, field_name, value)

                decision = _decide(planner)

                self.assertIs(decision.action, QueryAction.SEARCH)
                self.assertEqual(decision.reason, "cache_fault")
                self.assertEqual(planner.health["fault_invalidations"], 1)
                self.assertEqual(planner.health["entries"], 0)

    def test_health_counters_form_an_exact_lookup_partition(self) -> None:
        planner = OrchestrationPlanner()
        _prime(planner)
        self.assertIs(_decide(planner).action, QueryAction.REUSE)
        changed = _decide(planner, lexical_query=_LEXICAL_QUERY + " changed")
        self.assertIs(changed.action, QueryAction.SEARCH)
        self.assertTrue(
            planner.commit("session", changed, _SNAPSHOT, ("Q0001",))
        )
        object.__setattr__(
            next(iter(planner._entries.values())),
            "dependency_digest",
            b"broken",
        )
        self.assertIs(
            _decide(
                planner,
                lexical_query=_LEXICAL_QUERY + " changed",
            ).action,
            QueryAction.SEARCH,
        )

        health = planner.health
        classified_lookups = sum(
            int(health[key])
            for key in (
                "hits",
                "cold_misses",
                "dependency_misses",
                "backend_invalidations",
                "fault_invalidations",
            )
        )
        self.assertEqual(health["lookups"], classified_lookups)
        self.assertEqual(health["hits"], health["retrievals_avoided"])
        self.assertEqual(health["hits"], health["reranks_avoided"])
        self.assertEqual(health["stores"], 2)

    def test_cache_retains_no_sensitive_raw_text(self) -> None:
        canaries = (
            "CATEGORY_CANARY_f237c4",
            "REQUIREMENT_CANARY_94e18b",
            "EXCLUSION_CANARY_a8d0ef",
            "DENSE_QUERY_CANARY_95c1ea",
            "LEXICAL_QUERY_CANARY_7f0b2d",
        )
        state = IntentState(
            category=canaries[0],
            requirements=(
                Requirement(canaries[1], "answer", 1, "feature"),
            ),
            excluded=(canaries[2],),
        )
        planner = OrchestrationPlanner()
        _prime(
            planner,
            session_id="safe-session-key",
            state=state,
            dense_query=canaries[3],
            lexical_query=canaries[4],
            ranked_ids=("SAFE_PRODUCT_ID",),
        )

        strings, byte_strings = _retained_scalars(planner)
        for canary in canaries:
            with self.subTest(canary=canary):
                self.assertNotIn(canary, strings)
                self.assertFalse(
                    any(canary.encode("utf-8") in value for value in byte_strings)
                )
        self.assertIn("SAFE_PRODUCT_ID", strings)
        self.assertLessEqual(planner.health["entries"], 256)
        self.assertLessEqual(
            planner.health["cached_id_references"],
            MAX_CACHED_RANKED_IDS,
        )
        self.assertLessEqual(
            planner.health["cached_id_utf8_bytes"],
            MAX_CACHED_RANKED_IDS * MAX_CACHED_ID_CHARACTERS,
        )
        self.assertLess(planner.health["retained_cache_bytes"], 8 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
