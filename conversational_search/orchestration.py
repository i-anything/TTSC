"""Deterministic, label-free pre-query orchestration.

The planner treats retrieval/ranking as a pure computation over an explicit
dependency set.  It reuses work only after exact equality; intent-completeness
is diagnostic evidence, never a probabilistic permission to serve stale work.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from enum import Enum

from conversational_search.intent import IntentState
from conversational_search.ranking import RankingPolicy
from conversational_search.slates import MAX_SLATE_CANDIDATES
from conversational_search.strategy import RouteWeights, intent_completeness


RANKING_DEPENDENCY_VERSION = "exact-ranking-dependencies-v1"
DEFAULT_RANKING_CACHE_CAPACITY = 256
MAX_CACHED_RANKED_IDS = MAX_SLATE_CANDIDATES
MAX_CACHED_ID_CHARACTERS = 64
EXACT_RANKING_BACKEND_CONTRACT = "immutable-complete-fused-ranking-v1"


class BackendSnapshotToken:
    """Bounded opaque identity; instances deliberately cannot retain payloads."""

    __slots__ = ()


class _ExactRankingCacheCapability:
    __slots__ = ()


# An exact backend must deliberately expose this singleton and a bounded token.
# Merely defining a similarly named property is not enough to enable reuse.
EXACT_RANKING_CACHE_CAPABILITY = _ExactRankingCacheCapability()


class OrchestrationPolicy(str, Enum):
    """Reversible policies for the Phase 7 exact-value experiment."""

    ALWAYS_SEARCH = "always_search"
    EXACT_RANKING_REUSE = "exact_ranking_reuse"


ALWAYS_SEARCH_ORCHESTRATION_POLICY = OrchestrationPolicy.ALWAYS_SEARCH
EXACT_RANKING_REUSE_ORCHESTRATION_POLICY = (
    OrchestrationPolicy.EXACT_RANKING_REUSE
)


class QueryAction(str, Enum):
    """One pre-query action selected without labels or outcomes."""

    SEARCH = "search"
    REUSE = "reuse"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class RankingCacheEntry:
    """Bounded pre-slate artifact; deliberately contains no query text."""

    dependency_digest: bytes
    backend_snapshot_token: BackendSnapshotToken = field(repr=False)
    ranked_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dependency_digest, bytes)
            or len(self.dependency_digest) != hashlib.sha256().digest_size
        ):
            raise ValueError("dependency_digest must be a 32-byte SHA-256 digest")
        if type(self.backend_snapshot_token) is not BackendSnapshotToken:
            raise TypeError("backend_snapshot_token must be a bounded opaque token")
        _validate_ranked_ids(self.ranked_ids)


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """Auditable decision with only bounded, label-free signals."""

    action: QueryAction
    reason: str
    intent_evidence_completeness: float
    recomputation_value: float
    dependency_digest: bytes | None = field(default=None, repr=False)
    cached_ranked_ids: tuple[str, ...] = field(default=(), repr=False)
    backend_snapshot_token: BackendSnapshotToken | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.action, QueryAction):
            raise TypeError("action must be QueryAction")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if not 0.0 <= self.intent_evidence_completeness <= 1.0:
            raise ValueError("intent evidence completeness must be in [0, 1]")
        if self.recomputation_value not in {0.0, 1.0}:
            raise ValueError("recomputation value must be exactly zero or one")
        if self.action is QueryAction.REUSE:
            _validate_ranked_ids(self.cached_ranked_ids)
            if self.dependency_digest is None:
                raise ValueError("reuse requires a dependency digest")
        elif self.cached_ranked_ids:
            raise ValueError("only reuse decisions may contain cached IDs")


def ranking_dependency_digest(
    state: IntentState,
    dense_query: str,
    lexical_query: str,
    route_weights: RouteWeights,
    ranking_policy: RankingPolicy,
) -> bytes:
    """Hash every input that can affect retrieval or Stage-A ordering.

    Turn numbers, question history, no-preference markers, ``intent_version``,
    and requested top-k are intentionally absent.  They do not affect the
    complete ranking; the existing slate signature retains the fields that
    must reset presentation history.
    """

    if not isinstance(state, IntentState):
        raise TypeError("state must be IntentState")
    if not isinstance(dense_query, str) or not isinstance(lexical_query, str):
        raise TypeError("rendered queries must be strings")
    if not isinstance(route_weights, RouteWeights):
        raise TypeError("route_weights must be RouteWeights")
    if not isinstance(ranking_policy, RankingPolicy):
        raise TypeError("ranking_policy must be RankingPolicy")

    payload = {
        "version": RANKING_DEPENDENCY_VERSION,
        "backend_contract": EXACT_RANKING_BACKEND_CONTRACT,
        "category": state.category,
        "requirements": [
            [requirement.value, requirement.source, requirement.attribute]
            for requirement in state.requirements
        ],
        "excluded": list(state.excluded),
        "dense_query": dense_query,
        "lexical_query": lexical_query,
        "route_weights": [route_weights.bm25.hex(), route_weights.dense.hex()],
        "ranking_policy": ranking_policy.value,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).digest()


class OrchestrationPlanner:
    """Choose SEARCH/REUSE/SKIP and own one bounded ranking per session."""

    def __init__(
        self,
        policy: OrchestrationPolicy = EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        *,
        capacity: int = DEFAULT_RANKING_CACHE_CAPACITY,
    ) -> None:
        if not isinstance(policy, OrchestrationPolicy):
            raise TypeError("policy must be OrchestrationPolicy")
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if capacity > DEFAULT_RANKING_CACHE_CAPACITY:
            raise ValueError(
                f"capacity must not exceed {DEFAULT_RANKING_CACHE_CAPACITY}"
            )
        self._policy = policy
        self._capacity = capacity
        self._entries: OrderedDict[bytes, RankingCacheEntry] = OrderedDict()
        self._actions: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._lookups = 0
        self._hits = 0
        self._cold_misses = 0
        self._dependency_misses = 0
        self._backend_invalidations = 0
        self._fault_invalidations = 0
        self._stores = 0
        self._store_rejections = 0
        self._reset_invalidations = 0
        self._capacity_evictions = 0

    @property
    def policy(self) -> OrchestrationPolicy:
        return self._policy

    def reset(self, session_id: str) -> None:
        _validate_session_id(session_id)
        if self._entries.pop(_session_cache_key(session_id), None) is not None:
            self._reset_invalidations += 1

    def decide(
        self,
        session_id: str,
        state: IntentState,
        dense_query: str,
        lexical_query: str,
        route_weights: RouteWeights,
        ranking_policy: RankingPolicy,
        result_count: int,
        backend_snapshot_token: BackendSnapshotToken | None,
        cache_eligible: bool,
    ) -> TurnDecision:
        _validate_session_id(session_id)
        if isinstance(result_count, bool) or not isinstance(result_count, int):
            raise TypeError("result_count must be an integer")
        if not isinstance(cache_eligible, bool):
            raise TypeError("cache_eligible must be a boolean")
        evidence = intent_completeness(state)

        if result_count <= 0 and self._policy is OrchestrationPolicy.EXACT_RANKING_REUSE:
            return self._record(
                TurnDecision(
                    action=QueryAction.SKIP,
                    reason="empty_result_request",
                    intent_evidence_completeness=evidence,
                    recomputation_value=0.0,
                )
            )
        if self._policy is OrchestrationPolicy.ALWAYS_SEARCH:
            return self._record(
                TurnDecision(
                    action=QueryAction.SEARCH,
                    reason="policy_requires_search",
                    intent_evidence_completeness=evidence,
                    recomputation_value=1.0,
                )
            )
        if not cache_eligible:
            return self._record(
                TurnDecision(
                    action=QueryAction.SEARCH,
                    reason="ranking_not_cache_eligible",
                    intent_evidence_completeness=evidence,
                    recomputation_value=1.0,
                )
            )
        if type(backend_snapshot_token) is not BackendSnapshotToken:
            return self._record(
                TurnDecision(
                    action=QueryAction.SEARCH,
                    reason="backend_snapshot_unavailable",
                    intent_evidence_completeness=evidence,
                    recomputation_value=1.0,
                )
            )

        digest = ranking_dependency_digest(
            state,
            dense_query,
            lexical_query,
            route_weights,
            ranking_policy,
        )
        status, entry = self._lookup(
            session_id,
            digest,
            backend_snapshot_token,
        )
        if entry is not None:
            return self._record(
                TurnDecision(
                    action=QueryAction.REUSE,
                    reason="exact_dependency_hit",
                    intent_evidence_completeness=evidence,
                    recomputation_value=0.0,
                    dependency_digest=digest,
                    cached_ranked_ids=entry.ranked_ids,
                    backend_snapshot_token=backend_snapshot_token,
                )
            )
        return self._record(
            TurnDecision(
                action=QueryAction.SEARCH,
                reason=status,
                intent_evidence_completeness=evidence,
                recomputation_value=1.0,
                dependency_digest=digest,
                backend_snapshot_token=backend_snapshot_token,
            )
        )

    def commit(
        self,
        session_id: str,
        decision: TurnDecision,
        backend_snapshot_token: BackendSnapshotToken | None,
        ranked_ids: tuple[str, ...],
    ) -> bool:
        """Store one complete ranking; invalid stores fail closed."""

        _validate_session_id(session_id)
        if (
            self._policy is not OrchestrationPolicy.EXACT_RANKING_REUSE
            or decision.action is not QueryAction.SEARCH
            or decision.dependency_digest is None
            or type(backend_snapshot_token) is not BackendSnapshotToken
            or backend_snapshot_token is not decision.backend_snapshot_token
        ):
            self._store_rejections += 1
            return False
        try:
            entry = RankingCacheEntry(
                dependency_digest=decision.dependency_digest,
                backend_snapshot_token=backend_snapshot_token,
                ranked_ids=ranked_ids,
            )
        except (TypeError, ValueError):
            self._store_rejections += 1
            return False

        session_key = _session_cache_key(session_id)
        self._entries[session_key] = entry
        self._entries.move_to_end(session_key)
        self._stores += 1
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
            self._capacity_evictions += 1
        return True

    def _lookup(
        self,
        session_id: str,
        dependency_digest: bytes,
        backend_snapshot_token: BackendSnapshotToken,
    ) -> tuple[str, RankingCacheEntry | None]:
        self._lookups += 1
        session_key = _session_cache_key(session_id)
        entry = self._entries.get(session_key)
        if entry is None:
            self._cold_misses += 1
            return "cold_cache", None
        if not _valid_cache_entry(entry):
            self._entries.pop(session_key, None)
            self._fault_invalidations += 1
            return "cache_fault", None
        if entry.backend_snapshot_token is not backend_snapshot_token:
            self._entries.pop(session_key, None)
            self._backend_invalidations += 1
            return "backend_snapshot_changed", None
        if entry.dependency_digest != dependency_digest:
            self._dependency_misses += 1
            return "ranking_dependencies_changed", None
        self._entries.move_to_end(session_key)
        self._hits += 1
        return "exact_dependency_hit", entry

    def _record(self, decision: TurnDecision) -> TurnDecision:
        self._actions[decision.action.value] += 1
        self._reasons[decision.reason] += 1
        return decision

    @property
    def health(self) -> dict[str, object]:
        cached_id_references = sum(
            len(entry.ranked_ids)
            for entry in self._entries.values()
            if _valid_cache_entry(entry)
        )
        return {
            "policy": self._policy.value,
            "capacity": self._capacity,
            "maximum_ids_per_entry": MAX_CACHED_RANKED_IDS,
            "maximum_id_characters": MAX_CACHED_ID_CHARACTERS,
            "decisions": sum(self._actions.values()),
            "searches": self._actions[QueryAction.SEARCH.value],
            "reuses": self._actions[QueryAction.REUSE.value],
            "skips": self._actions[QueryAction.SKIP.value],
            "reasons": dict(sorted(self._reasons.items())),
            "lookups": self._lookups,
            "hits": self._hits,
            "cold_misses": self._cold_misses,
            "dependency_misses": self._dependency_misses,
            "backend_invalidations": self._backend_invalidations,
            "fault_invalidations": self._fault_invalidations,
            "stores": self._stores,
            "store_rejections": self._store_rejections,
            "reset_invalidations": self._reset_invalidations,
            "capacity_evictions": self._capacity_evictions,
            "retrievals_avoided": self._hits,
            "reranks_avoided": self._hits,
            "entries": len(self._entries),
            "cached_id_references": cached_id_references,
            "cached_id_utf8_bytes": sum(
                len(parent_asin.encode("utf-8"))
                for entry in self._entries.values()
                if _valid_cache_entry(entry)
                for parent_asin in entry.ranked_ids
            ),
            "retained_cache_bytes": _retained_cache_bytes(self._entries),
        }


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")


def _session_cache_key(session_id: str) -> bytes:
    """Avoid retaining arbitrary-length or sensitive session identifiers."""

    return hashlib.sha256(session_id.encode("utf-8")).digest()


def _validate_ranked_ids(ranked_ids: tuple[str, ...]) -> None:
    if not isinstance(ranked_ids, tuple):
        raise TypeError("ranked_ids must be a tuple")
    if not ranked_ids:
        raise ValueError("ranked_ids must not be empty")
    if len(ranked_ids) > MAX_CACHED_RANKED_IDS:
        raise ValueError(
            f"at most {MAX_CACHED_RANKED_IDS} ranked IDs may be cached"
        )
    if any(
        not isinstance(parent_asin, str)
        or not parent_asin
        or parent_asin != parent_asin.strip()
        or not parent_asin.isascii()
        or len(parent_asin) > MAX_CACHED_ID_CHARACTERS
        for parent_asin in ranked_ids
    ):
        raise ValueError(
            "ranked_ids must contain normalized bounded ASCII strings"
        )
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked_ids must not contain duplicates")


def _valid_cache_entry(entry: object) -> bool:
    if not isinstance(entry, RankingCacheEntry):
        return False
    try:
        if (
            not isinstance(entry.dependency_digest, bytes)
            or len(entry.dependency_digest) != hashlib.sha256().digest_size
            or type(entry.backend_snapshot_token) is not BackendSnapshotToken
        ):
            return False
        _validate_ranked_ids(entry.ranked_ids)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _retained_cache_bytes(entries: OrderedDict[bytes, RankingCacheEntry]) -> int:
    """Return a conservative recursive CPython footprint of retained cache data."""

    seen: set[int] = set()

    def size(value: object) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(value)
        if isinstance(value, OrderedDict):
            return total + sum(size(key) + size(item) for key, item in value.items())
        if isinstance(value, tuple):
            return total + sum(size(item) for item in value)
        if isinstance(value, RankingCacheEntry):
            return total + size(value.dependency_digest) + size(
                value.backend_snapshot_token
            ) + size(value.ranked_ids)
        return total

    return size(entries)
