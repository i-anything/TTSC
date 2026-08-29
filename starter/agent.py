from __future__ import annotations

from pathlib import Path

from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.service import ConversationalSearchAgent


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.jsonl"


class Agent(ConversationalSearchAgent):
    """Competition API adapter for the offline conversational-search core."""

    def __init__(self, catalog_path: str | Path = DEFAULT_CATALOG_PATH) -> None:
        super().__init__(
            catalog_path,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        )
