"""Competition API adapter selecting the released agent configuration.

``Agent`` subclasses ``ConversationalSearchAgent`` and pins the policy set
evaluated for submission: smart hybrid retrieval routing, lexicographic
exact-evidence ranking, full-transcript protocol resolution with eligible
continuation refutation, the wildcard ``other`` question policy,
metric-aware exposure, exact-ranking reuse orchestration, and
intent-epoch novelty slates.
"""

from __future__ import annotations

from pathlib import Path

from conversational_search.exposure_policy import (
    PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
)
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.questions import WILDCARD_OTHER_POLICY
from conversational_search.protocol_index import (
    ELIGIBLE_CONTINUATION_REFUTATION_POLICY,
    FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval_routing import (
    SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.jsonl"


class Agent(ConversationalSearchAgent):
    """Competition API adapter for the offline conversational-search core."""

    def __init__(self, catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> None:
        super().__init__(
            catalog_path,
            evidence_exposure_policy=PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            protocol_catalog_policy=FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
            protocol_refutation_policy=(
                ELIGIBLE_CONTINUATION_REFUTATION_POLICY
            ),
            question_policy=WILDCARD_OTHER_POLICY,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            retrieval_routing_policy=(
                SMART_HYBRID_RETRIEVAL_ROUTING_POLICY
            ),
            slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        )
