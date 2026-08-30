from __future__ import annotations

from pathlib import Path

from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
)
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.jsonl"


class Agent(ConversationalSearchAgent):
    """Competition API adapter for the offline conversational-search core."""

    def __init__(self, catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> None:
        super().__init__(
            catalog_path,
            evidence_exposure_policy=BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        )
