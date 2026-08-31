"""Full-catalog protocol resolution over the frozen product cards.

The lexical and dense routes are intentionally bounded.  This module supplies
the complementary guarantee needed by the published evaluator protocol: a
product remains reachable whenever its reconstructed card could have emitted
the complete observed transcript.  It never consumes target labels, sample
identifiers, or evaluation datasets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence, Set
from dataclasses import dataclass
from enum import Enum

from conversational_search.protocol import (
    CandidateReplyStatus,
    DisclosureCard,
    ObservedProtocolEvent,
    ProductProtocolEvidence,
    ProtocolEventKind,
    remaining_reply,
)


MAX_PROTOCOL_CATEGORY_PRODUCTS = 5_000
MAX_PROTOCOL_OUTPUT_CANDIDATES = 200
PROTOCOL_RRF_K = 60


class ProtocolCatalogPolicy(str, Enum):
    """Whether exact published-protocol transcripts may reach the full catalog."""

    DISABLED = "disabled"
    FULL_TRANSCRIPT = "full-catalog-transcript-v1"


DISABLED_PROTOCOL_CATALOG_POLICY = ProtocolCatalogPolicy.DISABLED
FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY = ProtocolCatalogPolicy.FULL_TRANSCRIPT


class ProtocolRefutationPolicy(str, Enum):
    """Whether a continued score-eligible session refutes its prior slate."""

    DISABLED = "disabled"
    ELIGIBLE_CONTINUATION = "eligible-continuation-refutation-v1"


DISABLED_PROTOCOL_REFUTATION_POLICY = ProtocolRefutationPolicy.DISABLED
ELIGIBLE_CONTINUATION_REFUTATION_POLICY = (
    ProtocolRefutationPolicy.ELIGIBLE_CONTINUATION
)


class ProtocolResolutionStatus(str, Enum):
    EXACT = "exact"
    FAIL_OPEN_UNSUPPORTED = "fail_open_unsupported"
    FAIL_OPEN_ZERO_SUPPORT = "fail_open_zero_support"


@dataclass(frozen=True, slots=True)
class ResolvedCardGroup:
    """Protocol-equivalent products and their replayed disclosure state."""

    card: DisclosureCard
    parent_asins: tuple[str, ...]
    disclosed_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProtocolResolution:
    """Complete support for one recognized transcript within its category."""

    status: ProtocolResolutionStatus
    groups: tuple[ResolvedCardGroup, ...]
    candidate_ids: tuple[str, ...]
    initial_candidate_count: int
    refuted_count: int

    @property
    def exact(self) -> bool:
        return self.status is ProtocolResolutionStatus.EXACT

    @property
    def support_count(self) -> int:
        return len(self.candidate_ids)


@dataclass(frozen=True, slots=True)
class _StrictReplay:
    disclosed_values: tuple[str, ...]


def resolve_protocol_transcript(
    evidence: Sequence[ProductProtocolEvidence],
    protocol_events: Sequence[ObservedProtocolEvent],
    *,
    observed_turn_count: int,
    refuted_ids: Set[str] = frozenset(),
) -> ProtocolResolution:
    """Resolve every product whose card can emit the exact event prefix.

    Replay comparisons deliberately use the visible strings exactly.  The
    official simulator performs exact membership and rendering, so casefolding
    or punctuation folding here would merge distinguishable protocol worlds.
    """

    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence")
    candidates = tuple(evidence)
    if len(candidates) > MAX_PROTOCOL_CATEGORY_PRODUCTS:
        raise ValueError("protocol category exceeds the bounded product limit")
    if any(not isinstance(item, ProductProtocolEvidence) for item in candidates):
        raise TypeError("evidence must contain ProductProtocolEvidence values")
    if len({item.parent_asin for item in candidates}) != len(candidates):
        raise ValueError("protocol evidence IDs must be unique")
    if isinstance(protocol_events, (str, bytes)) or not isinstance(
        protocol_events,
        Sequence,
    ):
        raise TypeError("protocol_events must be a sequence")
    events = tuple(protocol_events)
    if any(not isinstance(event, ObservedProtocolEvent) for event in events):
        raise TypeError("protocol_events contain an invalid value")
    if (
        isinstance(observed_turn_count, bool)
        or not isinstance(observed_turn_count, int)
        or not 1 <= observed_turn_count <= 10
    ):
        raise ValueError("observed_turn_count must be from one through ten")
    if tuple(event.turn for event in events) != tuple(
        range(1, observed_turn_count + 1)
    ):
        return _failed_resolution(
            ProtocolResolutionStatus.FAIL_OPEN_UNSUPPORTED,
            len(candidates),
        )
    if not isinstance(refuted_ids, Set):
        raise TypeError("refuted_ids must be a set-like collection")
    refuted = frozenset(refuted_ids)
    if any(not isinstance(value, str) or not value for value in refuted):
        raise ValueError("refuted IDs must be non-empty strings")

    grouped: dict[
        tuple[str, str, tuple[str, ...], tuple[str, ...]],
        list[ProductProtocolEvidence],
    ] = defaultdict(list)
    for item in candidates:
        grouped[
            (
                item.coarse_category,
                item.card.target_category,
                item.card.hard_constraints,
                item.card.soft_preferences,
            )
        ].append(item)

    replay_by_id: dict[str, _StrictReplay] = {}
    group_by_id: dict[str, tuple[DisclosureCard, tuple[str, ...]]] = {}
    for items in grouped.values():
        replay = _strict_replay(items[0].card, events)
        if replay is None:
            continue
        unrefuted_ids = tuple(
            item.parent_asin
            for item in items
            if item.parent_asin not in refuted
        )
        if not unrefuted_ids:
            continue
        for parent_asin in unrefuted_ids:
            replay_by_id[parent_asin] = replay
            group_by_id[parent_asin] = (items[0].card, unrefuted_ids)

    candidate_ids = tuple(
        item.parent_asin
        for item in candidates
        if item.parent_asin in replay_by_id
    )
    if not candidate_ids:
        return _failed_resolution(
            ProtocolResolutionStatus.FAIL_OPEN_ZERO_SUPPORT,
            len(candidates),
            refuted_count=sum(item.parent_asin in refuted for item in candidates),
        )

    ordered_groups: list[ResolvedCardGroup] = []
    seen_group_ids: set[tuple[str, ...]] = set()
    for parent_asin in candidate_ids:
        card, group_ids = group_by_id[parent_asin]
        if group_ids in seen_group_ids:
            continue
        seen_group_ids.add(group_ids)
        replay = replay_by_id[parent_asin]
        ordered_groups.append(
            ResolvedCardGroup(card, group_ids, replay.disclosed_values)
        )
    return ProtocolResolution(
        ProtocolResolutionStatus.EXACT,
        tuple(ordered_groups),
        candidate_ids,
        len(candidates),
        sum(item.parent_asin in refuted for item in candidates),
    )


def fuse_protocol_candidates(
    resolution: ProtocolResolution,
    preferred_ids: Sequence[str],
    *,
    limit: int = MAX_PROTOCOL_OUTPUT_CANDIDATES,
) -> tuple[str, ...]:
    """Fuse complete protocol support with the existing BM25+BGE prior.

    Protocol order is supplied by catalog popularity and stable catalog order.
    A second ordinary RRF vote preserves the active hybrid system's semantic
    intelligence without allowing its bounded recall to hide an exact survivor.
    """

    if not isinstance(resolution, ProtocolResolution):
        raise TypeError("resolution must be a ProtocolResolution")
    if not resolution.exact:
        return ()
    if isinstance(preferred_ids, (str, bytes)) or not isinstance(
        preferred_ids,
        Sequence,
    ):
        raise TypeError("preferred_ids must be a sequence")
    preferred = tuple(dict.fromkeys(preferred_ids))
    if any(not isinstance(value, str) or not value for value in preferred):
        raise ValueError("preferred IDs must be non-empty strings")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 0 <= limit <= MAX_PROTOCOL_OUTPUT_CANDIDATES
    ):
        raise ValueError("limit is outside the protocol candidate bound")
    if limit == 0:
        return ()

    protocol_rank = {
        parent_asin: rank
        for rank, parent_asin in enumerate(resolution.candidate_ids, start=1)
    }
    preferred_rank = {
        parent_asin: rank
        for rank, parent_asin in enumerate(preferred, start=1)
        if parent_asin in protocol_rank
    }
    ranked = sorted(
        resolution.candidate_ids,
        key=lambda parent_asin: (
            -(
                1.0 / (PROTOCOL_RRF_K + protocol_rank[parent_asin])
                + (
                    1.0
                    / (PROTOCOL_RRF_K + preferred_rank[parent_asin])
                    if parent_asin in preferred_rank
                    else 0.0
                )
            ),
            protocol_rank[parent_asin],
        ),
    )
    return tuple(ranked[:limit])


def protocol_probe_question(resolution: ProtocolResolution) -> str | None:
    """Return the repeatable highest-information protocol probe, if any.

    ``other`` is the simulator's only superset query and reveals at most two
    still-undisclosed card values.  Repeating it is therefore useful until all
    surviving groups are exhausted, including after a boundary decline.
    """

    if not isinstance(resolution, ProtocolResolution):
        raise TypeError("resolution must be a ProtocolResolution")
    if not resolution.exact:
        return None
    return (
        "other"
        if any(
            remaining_reply(
                group.card,
                "other",
                group.disclosed_values,
            ).status
            is CandidateReplyStatus.DISCLOSURE
            for group in resolution.groups
        )
        else None
    )


def _strict_replay(
    card: DisclosureCard,
    events: tuple[ObservedProtocolEvent, ...],
) -> _StrictReplay | None:
    if not events:
        return None
    initial = events[0]
    hard = card.hard_constraints
    soft = card.soft_preferences
    disclosed: set[str] = set()
    override_session = False
    override_applied = False
    boundary_possible = False
    boundary_consumed = False

    if initial.kind is ProtocolEventKind.INITIAL_EXPLICIT:
        if not hard or initial.values != (hard[0],):
            return None
        disclosed.add(hard[0])
    elif initial.kind is ProtocolEventKind.INITIAL_TENTATIVE:
        if not soft or initial.values != (soft[-1],):
            return None
        override_session = True
    elif initial.kind is ProtocolEventKind.INITIAL_BROWSING:
        boundary_possible = True
    else:
        return None

    for event in events[1:]:
        if event.kind is ProtocolEventKind.OVERRIDE:
            if (
                not override_session
                or override_applied
                or not hard
                or event.values != (hard[0],)
            ):
                return None
            override_applied = True
            disclosed.add(hard[0])
            continue
        if event.kind in {
            ProtocolEventKind.INITIAL_EXPLICIT,
            ProtocolEventKind.INITIAL_TENTATIVE,
            ProtocolEventKind.INITIAL_BROWSING,
        }:
            return None
        if event.kind is ProtocolEventKind.BOUNDARY_DECLINE:
            if (
                not boundary_possible
                or boundary_consumed
                or event.attribute is None
            ):
                return None
            boundary_consumed = True
            boundary_possible = False
            continue
        if event.kind is ProtocolEventKind.NEED_ATTRIBUTE:
            # The event carries no action.  It is candidate-independent and
            # leaves a possible boundary world's one-time decline unconsumed.
            continue

        signature = remaining_reply(card, event.attribute, disclosed)
        if event.kind is ProtocolEventKind.DISCLOSURE:
            if (
                signature.status is not CandidateReplyStatus.DISCLOSURE
                or event.serialized_reply_values != "; ".join(signature.values)
            ):
                return None
            disclosed.update(signature.values)
        elif event.kind is ProtocolEventKind.NO_ADDITIONAL:
            if signature.status is not CandidateReplyStatus.NO_ADDITIONAL:
                return None
        else:
            return None
        boundary_possible = False

    return _StrictReplay(tuple(sorted(disclosed)))


def _failed_resolution(
    status: ProtocolResolutionStatus,
    initial_candidate_count: int,
    *,
    refuted_count: int = 0,
) -> ProtocolResolution:
    return ProtocolResolution(
        status,
        (),
        (),
        initial_candidate_count,
        refuted_count,
    )
