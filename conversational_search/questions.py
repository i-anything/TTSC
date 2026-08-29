from __future__ import annotations

from dataclasses import dataclass

from conversational_search.intent import IntentState, active_attributes


QUESTION_TEXT = {
    "feature": "Which product feature matters most to you?",
    "material": "Do you have a material preference?",
    "color": "Do you have a color preference?",
    "style": "What style or fit would you prefer?",
    "size": "Do you have a size or width requirement?",
    "use_case": "What will you mainly use it for?",
    "budget": "What budget range should I use?",
    "other": "Is there another requirement that would help narrow the choice?",
}


@dataclass(frozen=True, slots=True)
class QuestionPolicy:
    """Deterministic clarification order with optional interruption recovery."""

    name: str
    priority: tuple[str, ...]
    requeue_interrupted: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("question policy name must not be empty")
        if not self.priority or len(set(self.priority)) != len(self.priority):
            raise ValueError("question policy priority must be non-empty and unique")
        unsupported = [item for item in self.priority if item not in QUESTION_TEXT]
        if unsupported:
            raise ValueError(f"unsupported question attributes: {unsupported}")

    def choose(self, state: IntentState) -> str | None:
        resolved = active_attributes(state) | state.no_preference
        pending = state.last_asked_attribute
        if (
            self.requeue_interrupted
            and pending is not None
            and pending not in resolved
        ):
            return pending
        for attribute in self.priority:
            if attribute not in resolved and attribute not in state.asked_attributes:
                return attribute
        return None


PHASE1_QUESTION_POLICY = QuestionPolicy(
    name="phase1",
    priority=(
        "feature",
        "material",
        "color",
        "style",
        "size",
        "use_case",
        "budget",
        "other",
    ),
)

CONSERVATIVE_EARLY_OTHER_POLICY = QuestionPolicy(
    name="conservative_early_other",
    priority=(
        "feature",
        "material",
        "color",
        "other",
        "style",
        "size",
        "use_case",
        "budget",
    ),
    requeue_interrupted=True,
)

QUESTION_POLICIES = {
    PHASE1_QUESTION_POLICY.name: PHASE1_QUESTION_POLICY,
    CONSERVATIVE_EARLY_OTHER_POLICY.name: CONSERVATIVE_EARLY_OTHER_POLICY,
}
