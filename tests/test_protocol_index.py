from __future__ import annotations

import unittest

from conversational_search.protocol import (
    DisclosureCard,
    ObservedProtocolEvent,
    ProductProtocolEvidence,
    ProtocolEventKind,
)
from conversational_search.protocol_index import (
    ProtocolResolutionStatus,
    fuse_protocol_candidates,
    protocol_probe_question,
    resolve_protocol_transcript,
)


def evidence(
    parent_asin: str,
    hard: tuple[str, ...],
    soft: tuple[str, ...],
    *,
    popularity: int,
) -> ProductProtocolEvidence:
    return ProductProtocolEvidence(
        parent_asin,
        "Women Boots",
        DisclosureCard(f"title {parent_asin}", hard, soft),
        popularity=popularity,
    )


class ProtocolIndexTest(unittest.TestCase):
    def test_complete_replay_filters_on_exact_visible_replies(self) -> None:
        candidates = (
            evidence("A", ("leather", "color: red"), ("warm",), popularity=9),
            evidence("B", ("leather", "color: blue"), ("warm",), popularity=8),
            evidence("C", ("cotton", "color: red"), ("warm",), popularity=7),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("leather",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "other",
                reply_payload="color: red; warm",
            ),
        )
        result = resolve_protocol_transcript(
            candidates,
            events,
            observed_turn_count=2,
        )
        self.assertIs(result.status, ProtocolResolutionStatus.EXACT)
        self.assertEqual(result.candidate_ids, ("A",))
        self.assertEqual(result.groups[0].disclosed_values, ("color: red", "leather", "warm"))

    def test_case_differences_are_not_folded(self) -> None:
        candidates = (
            evidence("LOWER", ("leather",), ("warm",), popularity=2),
            evidence("UPPER", ("Leather",), ("warm",), popularity=1),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("leather",),
            ),
        )
        result = resolve_protocol_transcript(
            candidates,
            events,
            observed_turn_count=1,
        )
        self.assertEqual(result.candidate_ids, ("LOWER",))

    def test_boundary_decline_preserves_cards_for_repeated_other(self) -> None:
        candidates = (
            evidence("A", ("leather", "wide"), ("warm",), popularity=1),
        )
        declined = resolve_protocol_transcript(
            candidates,
            (
                ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.BOUNDARY_DECLINE,
                    "other",
                ),
            ),
            observed_turn_count=2,
        )
        self.assertEqual(declined.candidate_ids, ("A",))
        self.assertEqual(declined.groups[0].disclosed_values, ())
        self.assertEqual(protocol_probe_question(declined), "other")

        disclosed = resolve_protocol_transcript(
            candidates,
            (
                ObservedProtocolEvent(1, ProtocolEventKind.INITIAL_BROWSING),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.BOUNDARY_DECLINE,
                    "other",
                ),
                ObservedProtocolEvent(
                    3,
                    ProtocolEventKind.DISCLOSURE,
                    "other",
                    reply_payload="leather; wide",
                ),
            ),
            observed_turn_count=3,
        )
        self.assertEqual(disclosed.candidate_ids, ("A",))
        self.assertEqual(protocol_probe_question(disclosed), "other")

    def test_tentative_value_remains_validated_after_override(self) -> None:
        candidates = (
            evidence("A", ("leather", "wide"), ("warm",), popularity=2),
            evidence("B", ("leather", "wide"), ("cool",), popularity=1),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_TENTATIVE,
                values=("warm",),
            ),
            ObservedProtocolEvent(2, ProtocolEventKind.NEED_ATTRIBUTE),
            ObservedProtocolEvent(
                3,
                ProtocolEventKind.OVERRIDE,
                values=("leather",),
            ),
        )
        result = resolve_protocol_transcript(
            candidates,
            events,
            observed_turn_count=3,
        )
        self.assertEqual(result.candidate_ids, ("A",))

    def test_refutation_removes_only_the_exposed_sibling(self) -> None:
        candidates = (
            evidence("A", ("leather",), ("warm",), popularity=2),
            evidence("B", ("leather",), ("warm",), popularity=1),
        )
        result = resolve_protocol_transcript(
            candidates,
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_EXPLICIT,
                    values=("leather",),
                ),
            ),
            observed_turn_count=1,
            refuted_ids=frozenset({"A"}),
        )
        self.assertEqual(result.candidate_ids, ("B",))
        self.assertEqual(result.refuted_count, 1)

    def test_rrf_keeps_protocol_only_candidates_reachable(self) -> None:
        candidates = tuple(
            evidence(str(index), ("leather",), ("warm",), popularity=10 - index)
            for index in range(5)
        )
        resolution = resolve_protocol_transcript(
            candidates,
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_EXPLICIT,
                    values=("leather",),
                ),
            ),
            observed_turn_count=1,
        )
        fused = fuse_protocol_candidates(resolution, ("4", "3"), limit=5)
        self.assertEqual(set(fused), {"0", "1", "2", "3", "4"})
        self.assertEqual(fused[0], "4")

    def test_zero_support_is_typed_fail_open(self) -> None:
        result = resolve_protocol_transcript(
            (evidence("A", ("leather",), ("warm",), popularity=1),),
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_EXPLICIT,
                    values=("cotton",),
                ),
            ),
            observed_turn_count=1,
        )
        self.assertIs(
            result.status,
            ProtocolResolutionStatus.FAIL_OPEN_ZERO_SUPPORT,
        )
        self.assertEqual(result.candidate_ids, ())


if __name__ == "__main__":
    unittest.main()
