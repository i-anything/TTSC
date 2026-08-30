"""Fail-open bridge from exact catalog evidence to turn-level utility actions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from math import ceil

from conversational_search.decision_policy import (
    EXPECTED_UTILITY_DECISION_POLICY,
    PROTECTED_DECISION_POLICY,
    PROTOCOL_UTILITY_DECISION_POLICY,
    DecisionPolicy,
)

from conversational_search.exact_evidence import (
    ExactEvidenceResult,
    ExactEvidenceStatus,
    rank_exact_evidence,
)
from conversational_search.intent import (
    ROBUST_INTENT_POLICY,
    IntentParsingPolicy,
    IntentState,
    active_attributes,
    apply_user_message,
    record_question,
)
from conversational_search.protocol import (
    ALLOWED_ATTRIBUTES,
    MAX_CONSTRAINT_CHARACTERS,
    MAX_REPLY_PAYLOAD_CHARACTERS,
    PROTOCOL_ACTIONS,
    CandidateReplySignature,
    ObservedProtocolEvent,
    ProductProtocolEvidence,
    ProtocolEventKind,
    ProtocolMode,
    QuestionReplyModel,
    build_protocol_world_model,
    eligible_protocol_actions,
    project_protocol_world_model,
    remaining_reply,
)
from conversational_search.slates import (
    SlateSelection,
    SlateState,
    select_slate_with_intent_epoch_novelty,
)
from conversational_search.utility_planner import (
    CandidateHypothesis,
    ExpectedUtilityCandidate,
    ExpectedUtilityPlan,
    RetrievalChoice,
    SimulatedQuestion,
    hit_utility,
    plan_expected_utility,
    plan_one_step_action,
)


MAX_DECISION_CANDIDATES = 200
MAX_SIMULATED_REPLY_PARTITIONS = 128
RERETRIEVAL_TIE_COST = 0.000001
LATENT_BOUNDARY_WORLD_PROBABILITY = 0.5
PROTOCOL_QUESTIONS = PROTOCOL_ACTIONS
RUNTIME_PROTOCOL_QUESTIONS = PROTOCOL_ACTIONS[:8]
_STRONG_SOURCES = frozenset({"initial_explicit", "answer", "override"})
_STRUCTURED_PRODUCT_ATTRIBUTES = frozenset(
    {
        "material",
        "color",
        "size",
        "feature",
        "use_case",
        "brand",
        "style",
        "budget",
    }
)
_SPACE_RE = re.compile(r"\s+")
_INITIAL_BUYING_RE = re.compile(
    r"^I'm looking for .+?\. A key requirement is: (?P<value>.+)\.$",
    re.DOTALL,
)
_INITIAL_BROWSING_RE = re.compile(
    r"^I'm looking for .+, but I'm still exploring\.$",
    re.DOTALL,
)
_INITIAL_TENTATIVE_RE = re.compile(
    r"^I'm looking for .+?\. (?P<value>.+)$",
    re.DOTALL,
)
_INITIAL_RES = (
    _INITIAL_BUYING_RE,
    _INITIAL_BROWSING_RE,
    _INITIAL_TENTATIVE_RE,
)
_ANSWER_RE = re.compile(
    r"^For that, what matters is: (?P<value>.+)\.$",
    re.DOTALL,
)
_OVERRIDE_RE = re.compile(
    r"^Actually, ignore my earlier preference\. What I need is: "
    r"(?P<value>.+)\.$",
    re.DOTALL,
)
_BOUNDARY_RE = re.compile(
    r"^I don't have a preference for (?P<attribute>[a-z_]+); "
    r"please use your judgment\.$"
)
_NO_ADDITIONAL_RE = re.compile(
    r"^I don't have an additional preference for "
    r"(?P<attribute>[a-z_]+)\.$"
)
_NEED_ATTRIBUTE = (
    "Those options are not quite right yet. "
    "Ask me about one specific attribute."
)


class ProtocolObservation(str, Enum):
    INITIAL = "initial"
    DISCLOSURE = "disclosure"
    OVERRIDE = "override"
    BOUNDARY_DECLINE = "boundary_decline"
    NO_ADDITIONAL = "no_additional"
    NEED_ATTRIBUTE = "need_attribute"
    UNSUPPORTED = "unsupported"


class ProtocolDecisionStatus(str, Enum):
    APPLIED = "applied"
    FAIL_OPEN_EVIDENCE = "fail_open_evidence"
    FAIL_OPEN_NO_CANDIDATES = "fail_open_no_candidates"
    FAIL_OPEN_NO_SUPPORT = "fail_open_no_support"
    FAIL_OPEN_VALIDATION = "fail_open_validation"


@dataclass(frozen=True, slots=True)
class ProtocolDecisionTrace:
    candidate_count: int
    available_candidate_count: int
    support_count: int
    question_count: int
    protocol_locked: bool
    protocol_mode: str = ProtocolMode.RECOVERY.value
    confidence: float = 0.0
    out_of_pool_probability: float = 1.0
    simulated_partition_count: int = 0
    pruned_question_count: int = 0
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ProtocolDecision:
    status: ProtocolDecisionStatus
    question: str | None
    width: int
    value: float
    ordered_ids: tuple[str, ...]
    trace: ProtocolDecisionTrace
    retrieval: RetrievalChoice = RetrievalChoice.RERETRIEVE
    immediate_value: float = 0.0
    continuation_value: float = 0.0
    runner_up_question: str | None = None
    runner_up_width: int | None = None
    runner_up_value: float | None = None


@dataclass(frozen=True, slots=True)
class Bm25OnlyConditions:
    """Auditable Boolean gate for the conservative BM25-first route."""

    message_is_exact_protocol: bool
    session_state_is_consistent: bool
    category_is_exactly_recognized: bool
    exact_product_constraints: int
    no_unparsed_or_free_text_requirement: bool
    no_tentative_override_or_contradiction: bool
    bm25_contains_structurally_valid_candidate: bool = False

    def __post_init__(self) -> None:
        boolean_values = (
            self.message_is_exact_protocol,
            self.session_state_is_consistent,
            self.category_is_exactly_recognized,
            self.no_unparsed_or_free_text_requirement,
            self.no_tentative_override_or_contradiction,
            self.bm25_contains_structurally_valid_candidate,
        )
        if any(type(value) is not bool for value in boolean_values):
            raise TypeError("BM25-only conditions must be booleans")
        if (
            isinstance(self.exact_product_constraints, bool)
            or not isinstance(self.exact_product_constraints, int)
            or self.exact_product_constraints < 0
        ):
            raise ValueError(
                "exact_product_constraints must be a non-negative integer"
            )

    @property
    def pre_bm25_eligible(self) -> bool:
        return all(
            (
                self.message_is_exact_protocol,
                self.session_state_is_consistent,
                self.category_is_exactly_recognized,
                self.exact_product_constraints >= 1,
                self.no_unparsed_or_free_text_requirement,
                self.no_tentative_override_or_contradiction,
            )
        )

    @property
    def bm25_only(self) -> bool:
        return (
            self.pre_bm25_eligible
            and self.bm25_contains_structurally_valid_candidate
        )

    def with_bm25_support(self, supported: bool) -> Bm25OnlyConditions:
        if type(supported) is not bool:
            raise TypeError("BM25 support must be a boolean")
        return replace(
            self,
            bm25_contains_structurally_valid_candidate=supported,
        )

    def as_dict(self) -> dict[str, bool | int]:
        return {
            "message_is_exact_protocol": self.message_is_exact_protocol,
            "session_state_is_consistent": self.session_state_is_consistent,
            "category_is_exactly_recognized": (
                self.category_is_exactly_recognized
            ),
            "exact_product_constraints": self.exact_product_constraints,
            "no_unparsed_or_free_text_requirement": (
                self.no_unparsed_or_free_text_requirement
            ),
            "no_tentative_override_or_contradiction": (
                self.no_tentative_override_or_contradiction
            ),
            "bm25_contains_structurally_valid_candidate": (
                self.bm25_contains_structurally_valid_candidate
            ),
        }


def protocol_state_is_consistent(
    state: IntentState,
    protocol_events: Sequence[ObservedProtocolEvent],
    current_turn: int,
) -> bool:
    """Check bounded state/event agreement without interpreting free prose."""

    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")
    if isinstance(current_turn, bool) or not isinstance(current_turn, int):
        raise TypeError("current_turn must be an integer")
    if isinstance(protocol_events, (str, bytes)) or not isinstance(
        protocol_events,
        Sequence,
    ):
        raise TypeError("protocol_events must be a sequence")
    events = tuple(protocol_events)
    if (
        not state.category
        or state.last_turn != current_turn
        or not events
        or events[-1].turn != current_turn
        or any(not isinstance(event, ObservedProtocolEvent) for event in events)
    ):
        return False
    turns = tuple(event.turn for event in events)
    if turns != tuple(sorted(set(turns))) or turns[0] != 1:
        return False
    if any(
        not requirement.value
        or not 1 <= requirement.turn <= state.last_turn
        or requirement.strength not in {"hard", "soft"}
        for requirement in state.requirements
    ):
        return False
    active_hard_attributes = tuple(
        requirement.attribute
        for requirement in state.requirements
        if requirement.strength == "hard" and requirement.attribute is not None
    )
    if (
        len(active_hard_attributes) != len(set(active_hard_attributes))
        or any(
            requirement.attribute in state.no_preference
            for requirement in state.requirements
            if requirement.attribute is not None
        )
        or (
            state.last_asked_attribute is not None
            and state.last_asked_attribute not in state.asked_attributes
        )
    ):
        return False
    events_by_turn = {event.turn: event for event in events}
    expected_kinds = {
        "initial_explicit": ProtocolEventKind.INITIAL_EXPLICIT,
        "initial_tentative": ProtocolEventKind.INITIAL_TENTATIVE,
        "answer": ProtocolEventKind.DISCLOSURE,
        "override": ProtocolEventKind.OVERRIDE,
    }
    for requirement in state.requirements:
        expected_kind = expected_kinds.get(requirement.source)
        event = events_by_turn.get(requirement.turn)
        if expected_kind is None or event is None or event.kind is not expected_kind:
            return False
        expected_value = _SPACE_RE.sub(" ", requirement.value).strip().casefold()
        if event.kind is ProtocolEventKind.DISCLOSURE:
            actual_value = _SPACE_RE.sub(
                " ", event.serialized_reply_values or ""
            ).strip().casefold()
            if event.attribute != requirement.attribute:
                return False
        else:
            if len(event.values) != 1:
                return False
            actual_value = _SPACE_RE.sub(" ", event.values[0]).strip().casefold()
        if not expected_value or actual_value != expected_value:
            return False
    positives = {
        _SPACE_RE.sub(" ", requirement.value).strip().casefold()
        for requirement in state.requirements
    }
    negatives = {
        _SPACE_RE.sub(" ", value).strip().casefold()
        for value in state.excluded
        if isinstance(value, str)
    }
    return not positives.intersection(negatives)


def protocol_events_are_structured_for_routing(
    protocol_events: Sequence[ObservedProtocolEvent],
) -> bool:
    """Reject opaque/multi-value disclosures from the BM25-only route."""

    if isinstance(protocol_events, (str, bytes)) or not isinstance(
        protocol_events,
        Sequence,
    ):
        raise TypeError("protocol_events must be a sequence")
    for event in protocol_events:
        if not isinstance(event, ObservedProtocolEvent):
            raise TypeError(
                "protocol_events must contain ObservedProtocolEvent values"
            )
        if any(";" in value for value in event.values):
            return False
        if event.kind is ProtocolEventKind.DISCLOSURE:
            payload = event.serialized_reply_values
            if (
                event.attribute not in _STRUCTURED_PRODUCT_ATTRIBUTES
                or not payload
                or ";" in payload
            ):
                return False
    return True


def derive_bm25_only_conditions(
    state: IntentState,
    *,
    message_is_exact_protocol: bool,
    session_state_is_consistent: bool,
    category_is_exactly_recognized: bool,
    exact_product_constraints: int,
    session_forces_hybrid: bool,
    protocol_values_are_structured: bool,
) -> Bm25OnlyConditions:
    """Derive the six pre-BM25 conditions from explicit bounded evidence."""

    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")
    for value in (
        message_is_exact_protocol,
        session_state_is_consistent,
        category_is_exactly_recognized,
        session_forces_hybrid,
        protocol_values_are_structured,
    ):
        if type(value) is not bool:
            raise TypeError("route condition inputs must be booleans")

    hard_requirements_are_typed = all(
        requirement.strength != "hard"
        or requirement.attribute in _STRUCTURED_PRODUCT_ATTRIBUTES
        for requirement in state.requirements
    )
    no_free_text = all(
        requirement.source != "free_text"
        for requirement in state.requirements
    )
    return Bm25OnlyConditions(
        message_is_exact_protocol=message_is_exact_protocol,
        session_state_is_consistent=session_state_is_consistent,
        category_is_exactly_recognized=category_is_exactly_recognized,
        exact_product_constraints=exact_product_constraints,
        no_unparsed_or_free_text_requirement=(
            hard_requirements_are_typed
            and no_free_text
            and protocol_values_are_structured
        ),
        no_tentative_override_or_contradiction=(
            not session_forces_hybrid and not state.excluded
        ),
    )


def protocol_route_dependency_digest(
    conditions: Bm25OnlyConditions,
    protocol_events: Sequence[ObservedProtocolEvent],
    structural_support_ids: Sequence[str],
) -> str:
    """Hash every candidate-only input that can change dense routing."""

    if not isinstance(conditions, Bm25OnlyConditions):
        raise TypeError("conditions must be Bm25OnlyConditions")
    if isinstance(protocol_events, (str, bytes)) or not isinstance(
        protocol_events,
        Sequence,
    ):
        raise TypeError("protocol_events must be a sequence")
    if isinstance(structural_support_ids, (str, bytes)) or not isinstance(
        structural_support_ids,
        Sequence,
    ):
        raise TypeError("structural_support_ids must be a sequence")
    events = tuple(protocol_events)
    support = tuple(sorted(set(structural_support_ids)))
    if any(not isinstance(event, ObservedProtocolEvent) for event in events):
        raise TypeError("protocol_events contain an invalid value")
    if any(not isinstance(value, str) or not value for value in support):
        raise ValueError("structural support IDs must be non-empty strings")
    payload = {
        "version": "protocol-bm25-first-route-dependencies-v1",
        "conditions": conditions.as_dict(),
        "events": [
            [
                event.turn,
                event.kind.value,
                event.attribute,
                list(event.values),
                event.reply_payload,
            ]
            for event in events
        ],
        "structural_support_ids": list(support),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def recognize_protocol_observation(message: str, turn: int) -> ProtocolObservation:
    """Classify only published simulator shapes; unfamiliar prose fails open."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= 10:
        raise ValueError("turn must be an integer from 1 through 10")
    cleaned = _SPACE_RE.sub(" ", message).strip()
    if turn == 1 and any(pattern.fullmatch(cleaned) for pattern in _INITIAL_RES):
        return ProtocolObservation.INITIAL
    if _ANSWER_RE.fullmatch(cleaned):
        return ProtocolObservation.DISCLOSURE
    if _OVERRIDE_RE.fullmatch(cleaned):
        return ProtocolObservation.OVERRIDE
    if _BOUNDARY_RE.fullmatch(cleaned):
        return ProtocolObservation.BOUNDARY_DECLINE
    if _NO_ADDITIONAL_RE.fullmatch(cleaned):
        return ProtocolObservation.NO_ADDITIONAL
    if cleaned == _NEED_ATTRIBUTE:
        return ProtocolObservation.NEED_ATTRIBUTE
    return ProtocolObservation.UNSUPPORTED


def protocol_observation_attribute(
    message: str,
    observation: ProtocolObservation,
) -> str | None:
    """Extract an exact reply attribute, or return ``None`` when inapplicable."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not isinstance(observation, ProtocolObservation):
        raise TypeError("observation must be a ProtocolObservation")
    pattern = {
        ProtocolObservation.BOUNDARY_DECLINE: _BOUNDARY_RE,
        ProtocolObservation.NO_ADDITIONAL: _NO_ADDITIONAL_RE,
    }.get(observation)
    if pattern is None:
        return None
    match = pattern.fullmatch(_SPACE_RE.sub(" ", message).strip())
    if match is None:
        return None
    attribute = match.group("attribute")
    return attribute if attribute in ALLOWED_ATTRIBUTES else None


def parse_protocol_event(
    message: str,
    observation: ProtocolObservation,
    turn: int,
    *,
    asked_attribute: str | None,
) -> ObservedProtocolEvent:
    """Convert a recognized official message into a bounded replay event."""

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    if not isinstance(observation, ProtocolObservation):
        raise TypeError("observation must be a ProtocolObservation")
    cleaned = _SPACE_RE.sub(" ", message).strip()
    if observation is ProtocolObservation.INITIAL:
        buying = _INITIAL_BUYING_RE.fullmatch(cleaned)
        if buying is not None:
            return ObservedProtocolEvent(
                turn,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=(_clean_event_value(buying.group("value")),),
            )
        if _INITIAL_BROWSING_RE.fullmatch(cleaned):
            return ObservedProtocolEvent(
                turn,
                ProtocolEventKind.INITIAL_BROWSING,
            )
        tentative = _INITIAL_TENTATIVE_RE.fullmatch(cleaned)
        if tentative is not None:
            return ObservedProtocolEvent(
                turn,
                ProtocolEventKind.INITIAL_TENTATIVE,
                values=(_clean_event_value(tentative.group("value")),),
            )
    elif observation is ProtocolObservation.DISCLOSURE:
        answer = _ANSWER_RE.fullmatch(cleaned)
        payload = (
            None
            if answer is None
            else _clean_reply_payload(answer.group("value"))
        )
        if asked_attribute in ALLOWED_ATTRIBUTES and payload:
            return ObservedProtocolEvent(
                turn,
                ProtocolEventKind.DISCLOSURE,
                asked_attribute,
                reply_payload=payload,
            )
    elif observation is ProtocolObservation.OVERRIDE:
        override = _OVERRIDE_RE.fullmatch(cleaned)
        if override is not None:
            return ObservedProtocolEvent(
                turn,
                ProtocolEventKind.OVERRIDE,
                values=(_clean_event_value(override.group("value")),),
            )
    elif observation in {
        ProtocolObservation.BOUNDARY_DECLINE,
        ProtocolObservation.NO_ADDITIONAL,
    }:
        attribute = protocol_observation_attribute(message, observation)
        if attribute == asked_attribute:
            kind = (
                ProtocolEventKind.BOUNDARY_DECLINE
                if observation is ProtocolObservation.BOUNDARY_DECLINE
                else ProtocolEventKind.NO_ADDITIONAL
            )
            return ObservedProtocolEvent(turn, kind, attribute)
    elif observation is ProtocolObservation.NEED_ATTRIBUTE:
        return ObservedProtocolEvent(turn, ProtocolEventKind.NEED_ATTRIBUTE)
    raise ValueError("message is not a valid event for the observed protocol shape")


def _clean_event_value(value: str) -> str:
    cleaned = _SPACE_RE.sub(" ", value).strip()
    if not cleaned:
        raise ValueError("protocol event values must not be empty")
    if len(cleaned) > MAX_CONSTRAINT_CHARACTERS:
        raise ValueError("protocol event values exceed the character bound")
    return cleaned


def _clean_reply_payload(value: str) -> str:
    cleaned = _SPACE_RE.sub(" ", value).strip()
    if not cleaned or len(cleaned) > MAX_REPLY_PAYLOAD_CHARACTERS:
        raise ValueError("serialized reply values are empty or over the bound")
    return cleaned


def exact_query_constraints(
    state: IntentState,
    protocol_events: Sequence[ObservedProtocolEvent] = (),
) -> tuple[str, ...]:
    """Return bounded strong card values for the union-only exact route."""

    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")
    if isinstance(protocol_events, (str, bytes)) or not isinstance(
        protocol_events,
        Sequence,
    ):
        raise TypeError("protocol_events must be a sequence")
    events = tuple(protocol_events)
    if any(not isinstance(event, ObservedProtocolEvent) for event in events):
        raise TypeError("protocol_events must contain ObservedProtocolEvent values")
    values: list[str] = []
    seen: set[str] = set()
    if events and not any(
        event.kind is ProtocolEventKind.OVERRIDE for event in events
    ):
        for event in events:
            if event.kind not in {
                ProtocolEventKind.INITIAL_EXPLICIT,
                ProtocolEventKind.DISCLOSURE,
                ProtocolEventKind.OVERRIDE,
            }:
                continue
            raw_values = event.values
            if event.kind is ProtocolEventKind.DISCLOSURE and not raw_values:
                payload = event.serialized_reply_values
                raw_values = (
                    (payload,)
                    if event.attribute in _STRUCTURED_PRODUCT_ATTRIBUTES
                    and payload
                    and ";" not in payload
                    else ()
                )
            for raw_value in raw_values:
                value = _SPACE_RE.sub(" ", raw_value).strip(" -;,.\t\n")
                key = value.casefold()
                if value and key not in seen:
                    seen.add(key)
                    values.append(value)
                    if len(values) >= 8:
                        return tuple(values)
        return tuple(values)

    for requirement in state.requirements:
        if (
            requirement.strength != "hard"
            or requirement.source == "free_text"
        ):
            continue
        for raw_value in requirement.value.split(";"):
            value = _SPACE_RE.sub(" ", raw_value).strip(" -;,.\t\n")
            key = value.casefold()
            if value and key not in seen:
                seen.add(key)
                values.append(value)
                if len(values) >= 8:
                    return tuple(values)
    return tuple(values)


def intent_override_is_locked(state: IntentState) -> bool:
    """Detect the official pre-override interval without scenario labels."""

    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")
    has_tentative = any(
        requirement.source == "initial_tentative"
        for requirement in state.requirements
    )
    has_override = any(
        requirement.source == "override" for requirement in state.requirements
    )
    return has_tentative and not has_override


def plan_protocol_decision(
    state: IntentState,
    exact_result: ExactEvidenceResult,
    evidence: Sequence[ProductProtocolEvidence],
    *,
    shown_ids: Sequence[str] = (),
    protocol_events: Sequence[ObservedProtocolEvent] = (),
    current_turn: int,
    requested_top_k: int,
    protocol_locked: bool | None = None,
) -> ProtocolDecision:
    """Plan over unexposed candidates, or return a typed fail-open result."""

    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")
    if not isinstance(exact_result, ExactEvidenceResult):
        raise TypeError("exact_result must be an ExactEvidenceResult")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence")
    candidates = tuple(evidence)
    if len(candidates) > MAX_DECISION_CANDIDATES:
        raise ValueError(f"at most {MAX_DECISION_CANDIDATES} candidates are supported")
    if any(not isinstance(item, ProductProtocolEvidence) for item in candidates):
        raise TypeError("evidence must contain ProductProtocolEvidence values")
    if isinstance(shown_ids, (str, bytes)) or not isinstance(shown_ids, Sequence):
        raise TypeError("shown_ids must be a sequence")
    shown = tuple(shown_ids)
    if any(not isinstance(value, str) or not value for value in shown):
        raise ValueError("shown IDs must be non-empty strings")
    if isinstance(protocol_events, (str, bytes)) or not isinstance(
        protocol_events,
        Sequence,
    ):
        raise TypeError("protocol_events must be a sequence")
    events = tuple(protocol_events)
    if len(events) > 10 or any(
        not isinstance(event, ObservedProtocolEvent) for event in events
    ):
        raise ValueError("protocol_events must contain at most ten events")
    if isinstance(current_turn, bool) or not isinstance(current_turn, int):
        raise ValueError("current_turn must be an integer")
    if not 1 <= current_turn <= 10:
        raise ValueError("current_turn must be from 1 through 10")
    if (
        isinstance(requested_top_k, bool)
        or not isinstance(requested_top_k, int)
        or not 0 <= requested_top_k <= 10
    ):
        raise ValueError("requested_top_k must be from 0 through 10")
    if protocol_locked is None:
        protocol_locked = intent_override_is_locked(state)
    elif not isinstance(protocol_locked, bool):
        raise TypeError("protocol_locked must be a boolean or None")

    base_trace = ProtocolDecisionTrace(
        candidate_count=len(candidates),
        available_candidate_count=0,
        support_count=0,
        question_count=0,
        protocol_locked=protocol_locked,
    )
    if exact_result.status is not ExactEvidenceStatus.APPLIED:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_EVIDENCE,
            exact_result.ranked_ids,
            requested_top_k,
            base_trace,
        )

    evidence_by_id = {item.parent_asin: item for item in candidates}
    if (
        len(evidence_by_id) != len(candidates)
        or set(evidence_by_id) != set(exact_result.ranked_ids)
    ):
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            exact_result.ranked_ids,
            requested_top_k,
            base_trace,
        )
    shown_set = frozenset(shown)
    available = tuple(
        parent_asin
        for parent_asin in exact_result.ranked_ids
        if parent_asin not in shown_set
    )
    trace = ProtocolDecisionTrace(
        candidate_count=len(candidates),
        available_candidate_count=len(available),
        support_count=0,
        question_count=0,
        protocol_locked=base_trace.protocol_locked,
    )
    if not available or requested_top_k == 0:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_NO_CANDIDATES,
            exact_result.ranked_ids,
            requested_top_k,
            trace,
        )

    belief_weights = {
        belief.parent_asin: belief.weight
        for belief in exact_result.beliefs
        if belief.parent_asin in available and belief.weight > 0.0
    }
    if not belief_weights:
        residual_support = tuple(
            parent_asin
            for parent_asin in exact_result.consistent_support_ids
            if parent_asin in available
        )
        if residual_support:
            weight = 1.0 / len(residual_support)
            belief_weights = {
                parent_asin: weight for parent_asin in residual_support
            }
    if not belief_weights:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_NO_SUPPORT,
            exact_result.ranked_ids,
            requested_top_k,
            trace,
        )

    disclosures = {
        item.parent_asin: item.disclosed_values
        for item in exact_result.disclosures
    }
    questions = PROTOCOL_QUESTIONS if current_turn < 10 else ()
    hypotheses: list[CandidateHypothesis] = []
    for rank, parent_asin in enumerate(available, start=1):
        candidate = evidence_by_id[parent_asin]
        answer_signatures = tuple(
            (
                question,
                _reply_key(
                    remaining_reply(
                        candidate.card,
                        question,
                        disclosures.get(parent_asin, ()),
                    )
                ),
            )
            for question in questions
        )
        hypotheses.append(
            CandidateHypothesis(
                candidate_id=parent_asin,
                rank=rank,
                weight=belief_weights.get(parent_asin, 0.0),
                answer_signatures=answer_signatures,
            )
        )

    planning_top_k = min(requested_top_k, len(hypotheses))
    boundary_ambiguous = (
        current_turn == 1
        and len(events) == 1
        and events[0].kind is ProtocolEventKind.INITIAL_BROWSING
    )
    try:
        action = plan_one_step_action(
            tuple(hypotheses),
            questions,
            current_turn=current_turn,
            top_k=planning_top_k,
            protocol_locked=trace.protocol_locked,
            shared_reply_probability=(
                LATENT_BOUNDARY_WORLD_PROBABILITY
                if boundary_ambiguous
                else 0.0
            ),
        )
    except Exception:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            exact_result.ranked_ids,
            requested_top_k,
            trace,
        )
    return ProtocolDecision(
        status=ProtocolDecisionStatus.APPLIED,
        question=action.question,
        width=action.width,
        value=action.value,
        ordered_ids=available,
        trace=ProtocolDecisionTrace(
            candidate_count=trace.candidate_count,
            available_candidate_count=trace.available_candidate_count,
            support_count=len(belief_weights),
            question_count=len(questions),
            protocol_locked=trace.protocol_locked,
        ),
    )


def plan_expected_utility_decision(
    state: IntentState,
    exact_result: ExactEvidenceResult,
    evidence: Sequence[ProductProtocolEvidence],
    *,
    slate_state: SlateState,
    ranking_signature: tuple[object, ...],
    shown_ids: Sequence[str] = (),
    protocol_events: Sequence[ObservedProtocolEvent] = (),
    current_turn: int,
    requested_top_k: int,
    protocol_locked: bool | None = None,
    intent_policy: IntentParsingPolicy = ROBUST_INTENT_POLICY,
    retrieval_was_reused: bool = False,
) -> ProtocolDecision:
    """Plan on the active exact order using simulated replies and real novelty.

    This is deliberately a bounded continuation model.  It reranks only the
    already retrieved candidate pool for known protocol replies; out-of-pool
    probability receives no invented reward.  Every current and next-turn rank
    is projected through the production intent-epoch novelty selector.
    """

    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")
    if not isinstance(exact_result, ExactEvidenceResult):
        raise TypeError("exact_result must be an ExactEvidenceResult")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence")
    candidates = tuple(evidence)
    if len(candidates) > MAX_DECISION_CANDIDATES:
        raise ValueError(f"at most {MAX_DECISION_CANDIDATES} candidates are supported")
    if any(not isinstance(item, ProductProtocolEvidence) for item in candidates):
        raise TypeError("evidence must contain ProductProtocolEvidence values")
    if not isinstance(slate_state, SlateState):
        raise TypeError("slate_state must be a SlateState")
    if not isinstance(ranking_signature, tuple) or not ranking_signature:
        raise ValueError("ranking_signature must be a non-empty tuple")
    if isinstance(shown_ids, (str, bytes)) or not isinstance(shown_ids, Sequence):
        raise TypeError("shown_ids must be a sequence")
    shown = tuple(shown_ids)
    if any(not isinstance(value, str) or not value for value in shown):
        raise ValueError("shown IDs must be non-empty strings")
    if isinstance(protocol_events, (str, bytes)) or not isinstance(
        protocol_events,
        Sequence,
    ):
        raise TypeError("protocol_events must be a sequence")
    events = tuple(protocol_events)
    if len(events) > 10 or any(
        not isinstance(event, ObservedProtocolEvent) for event in events
    ):
        raise ValueError("protocol_events must contain at most ten events")
    if isinstance(current_turn, bool) or not isinstance(current_turn, int):
        raise ValueError("current_turn must be an integer")
    if not 1 <= current_turn <= 10:
        raise ValueError("current_turn must be from 1 through 10")
    if (
        isinstance(requested_top_k, bool)
        or not isinstance(requested_top_k, int)
        or not 0 <= requested_top_k <= 10
    ):
        raise ValueError("requested_top_k must be from zero through ten")
    if not isinstance(intent_policy, IntentParsingPolicy):
        raise TypeError("intent_policy must be an IntentParsingPolicy")
    if not isinstance(retrieval_was_reused, bool):
        raise TypeError("retrieval_was_reused must be a boolean")
    if protocol_locked is None:
        protocol_locked = intent_override_is_locked(state)
    elif not isinstance(protocol_locked, bool):
        raise TypeError("protocol_locked must be a boolean or None")

    base_trace = ProtocolDecisionTrace(
        candidate_count=len(candidates),
        available_candidate_count=0,
        support_count=0,
        question_count=0,
        protocol_locked=protocol_locked,
    )
    if exact_result.status is not ExactEvidenceStatus.APPLIED:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_EVIDENCE,
            exact_result.ranked_ids,
            requested_top_k,
            base_trace,
        )

    evidence_by_id = {item.parent_asin: item for item in candidates}
    full_order = exact_result.ranked_ids
    if (
        len(evidence_by_id) != len(candidates)
        or set(evidence_by_id) != set(full_order)
        or len(full_order) != len(candidates)
    ):
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            full_order,
            requested_top_k,
            replace(base_trace, fallback_reason="evidence_order_mismatch"),
        )
    planning_top_k = min(requested_top_k, len(full_order))
    if not full_order or planning_top_k <= 0:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_NO_CANDIDATES,
            full_order,
            requested_top_k,
            base_trace,
        )

    shown_set = frozenset(shown)
    target_ids = tuple(
        parent_asin
        for parent_asin in full_order
        if protocol_locked or parent_asin not in shown_set
    )
    if not target_ids:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_NO_CANDIDATES,
            full_order,
            requested_top_k,
            replace(base_trace, fallback_reason="all_candidates_previously_exposed"),
        )

    belief_weights = {
        belief.parent_asin: float(belief.weight)
        for belief in exact_result.beliefs
        if belief.parent_asin in target_ids and belief.weight > 0.0
    }
    if not belief_weights:
        support = tuple(
            parent_asin
            for parent_asin in exact_result.consistent_support_ids
            if parent_asin in target_ids
        )
        belief_weights = {
            parent_asin: 1.0 / rank
            for rank, parent_asin in enumerate(support, start=1)
        }
    if not belief_weights:
        belief_weights = {
            parent_asin: 1.0 / rank
            for rank, parent_asin in enumerate(target_ids, start=1)
        }

    try:
        world = build_protocol_world_model(
            tuple(evidence_by_id[parent_asin] for parent_asin in target_ids),
            protocol_events=events,
            observed_turn_count=current_turn,
            candidate_weights=belief_weights,
            actions=RUNTIME_PROTOCOL_QUESTIONS,
        )
    except Exception:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            full_order,
            requested_top_k,
            replace(base_trace, fallback_reason="world_model_validation"),
        )

    trace = ProtocolDecisionTrace(
        candidate_count=len(candidates),
        available_candidate_count=len(target_ids),
        support_count=len(world.candidate_probabilities),
        question_count=0,
        protocol_locked=protocol_locked,
        protocol_mode=world.assessment.mode.value,
        confidence=float(world.assessment.confidence),
        out_of_pool_probability=float(world.out_of_pool_probability),
        fallback_reason=world.assessment.fallback_reason,
    )
    if world.assessment.mode in {ProtocolMode.FREE_FORM, ProtocolMode.RECOVERY}:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_NO_SUPPORT,
            full_order,
            requested_top_k,
            trace,
        )
    if not world.candidate_probabilities:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_NO_SUPPORT,
            full_order,
            requested_top_k,
            replace(trace, fallback_reason="empty_candidate_belief"),
        )

    try:
        current_novelty_order = select_slate_with_intent_epoch_novelty(
            slate_state,
            ranking_signature,
            full_order,
            len(full_order),
        ).selection.selected_ids
    except Exception:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            full_order,
            requested_top_k,
            replace(trace, fallback_reason="current_novelty_projection"),
        )

    resolved = active_attributes(state) | state.no_preference
    eligible_actions = eligible_protocol_actions(
        world,
        asked_actions=state.asked_attributes,
        resolved_actions=resolved,
    )
    eligible_actions = tuple(
        action for action in eligible_actions if action in RUNTIME_PROTOCOL_QUESTIONS
    )
    if eligible_actions:
        projection_widths = _expected_utility_widths(
            current_turn=current_turn,
            top_k=planning_top_k,
            protocol_mode=world.assessment.mode,
            protocol_confidence=float(world.assessment.confidence),
            protocol_locked=protocol_locked,
        )
        projection_rank_by_id = {
            parent_asin: rank
            for rank, parent_asin in enumerate(current_novelty_order, start=1)
        }
        projection_probabilities = tuple(
            sorted(
                world.candidate_probabilities,
                key=lambda item: projection_rank_by_id[item.parent_asin],
            )
        )
        projection_hypotheses = tuple(
            ExpectedUtilityCandidate(
                item.parent_asin,
                projection_rank_by_id[item.parent_asin],
                float(item.probability),
            )
            for item in projection_probabilities
        )
        projection_ids = tuple(
            item.candidate_id for item in projection_hypotheses
        )
        projection_no_question = []
        try:
            for width in projection_widths:
                current_selection = select_slate_with_intent_epoch_novelty(
                    slate_state,
                    ranking_signature,
                    full_order,
                    width,
                ).selection
                next_selection = select_slate_with_intent_epoch_novelty(
                    current_selection.state,
                    ranking_signature,
                    full_order,
                    planning_top_k,
                ).selection
                projection_no_question.append(
                    (
                        width,
                        _selected_rank_map(
                            projection_ids,
                            next_selection.selected_ids,
                        ),
                    )
                )
            projection_retrievals = (
                (RetrievalChoice.REUSE,)
                if retrieval_was_reused
                else (RetrievalChoice.RERETRIEVE,)
            )
            projection_plan = plan_expected_utility(
                projection_hypotheses,
                (),
                current_turn=current_turn,
                top_k=planning_top_k,
                widths=projection_widths,
                retrieval_choices=projection_retrievals,
                out_of_pool_probability=float(world.out_of_pool_probability),
                protocol_confidence=float(world.assessment.confidence),
                protocol_locked=protocol_locked,
                allow_zero_width=(world.assessment.mode is ProtocolMode.EXACT),
                reretrieve_computation_cost=RERETRIEVAL_TIE_COST,
                no_question_post_ranks_by_width=tuple(
                    projection_no_question
                ),
                fallback_reason=world.assessment.fallback_reason,
            )
        except Exception:
            return _fail_open(
                ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
                full_order,
                requested_top_k,
                replace(trace, fallback_reason="world_projection_planner"),
            )

        computation_cost = (
            0.0
            if projection_retrievals[0] is RetrievalChoice.REUSE
            else RERETRIEVAL_TIE_COST
        )
        competitive_questions = sum(
            _question_utility_upper_bound(
                projection_hypotheses,
                question,
                widths=projection_widths,
                current_turn=current_turn,
                top_k=planning_top_k,
                protocol_locked=protocol_locked,
                computation_cost=computation_cost,
            )
            > projection_plan.selected.value
            for action in eligible_actions
            if (question := world.question(action)) is not None
        )
        shared_partition = int(
            any(
                question is not None and question.shared_reply is not None
                for action in eligible_actions
                if (question := world.question(action)) is not None
            )
        )
        hypothesis_limit = len(world.candidate_probabilities)
        if competitive_questions:
            hypothesis_limit = max(
                planning_top_k,
                (
                    MAX_SIMULATED_REPLY_PARTITIONS
                    // competitive_questions
                )
                - shared_partition,
            )
        hypothesis_limit = min(
            hypothesis_limit,
            len(world.candidate_probabilities),
        )
        if len(world.candidate_probabilities) > hypothesis_limit:
            world_ids = frozenset(
                item.parent_asin for item in world.candidate_probabilities
            )
            preferred = tuple(
                parent_asin
                for parent_asin in current_novelty_order
                if parent_asin in world_ids
            )[:hypothesis_limit]
            retained_set = frozenset(preferred)
            retained = tuple(
                item.parent_asin
                for item in world.candidate_probabilities
                if item.parent_asin in retained_set
            )
            try:
                world = project_protocol_world_model(world, retained)
            except Exception:
                return _fail_open(
                    ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
                    full_order,
                    requested_top_k,
                    replace(trace, fallback_reason="world_projection_validation"),
                )
            eligible_actions = eligible_protocol_actions(
                world,
                asked_actions=state.asked_attributes,
                resolved_actions=resolved,
            )
            eligible_actions = tuple(
                action
                for action in eligible_actions
                if action in RUNTIME_PROTOCOL_QUESTIONS
            )
            trace = replace(
                trace,
                support_count=len(world.candidate_probabilities),
                out_of_pool_probability=float(world.out_of_pool_probability),
            )
    widths = _expected_utility_widths(
        current_turn=current_turn,
        top_k=planning_top_k,
        protocol_mode=world.assessment.mode,
        protocol_confidence=float(world.assessment.confidence),
        protocol_locked=protocol_locked,
    )
    current_rank_by_id = {
        parent_asin: rank
        for rank, parent_asin in enumerate(current_novelty_order, start=1)
    }
    ordered_probabilities = tuple(
        sorted(
            world.candidate_probabilities,
            key=lambda item: current_rank_by_id[item.parent_asin],
        )
    )
    modeled_ids = tuple(item.parent_asin for item in ordered_probabilities)
    hypotheses = tuple(
        ExpectedUtilityCandidate(
            probability.parent_asin,
            current_rank_by_id[probability.parent_asin],
            float(probability.probability),
        )
        for probability in ordered_probabilities
    )

    current_selection_by_width = {}
    no_question_ranks_by_width = []
    try:
        for width in widths:
            current_selection = select_slate_with_intent_epoch_novelty(
                slate_state,
                ranking_signature,
                full_order,
                width,
            ).selection
            current_selection_by_width[width] = current_selection
            next_selection = select_slate_with_intent_epoch_novelty(
                current_selection.state,
                ranking_signature,
                full_order,
                planning_top_k,
            ).selection
            no_question_ranks_by_width.append(
                (
                    width,
                    _selected_rank_map(modeled_ids, next_selection.selected_ids),
                )
            )
    except Exception:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            full_order,
            requested_top_k,
            replace(trace, fallback_reason="no_question_projection"),
        )

    # Phase 4 scores the exact orchestration action that actually produced the
    # active pool. A counterfactual reretrieval arm is not fabricated with the
    # same ranks; Phase 5 supplies real route-specific retrieval projections.
    retrieval_choices = (
        (RetrievalChoice.REUSE,)
        if retrieval_was_reused
        else (RetrievalChoice.RERETRIEVE,)
    )

    def evaluate_plans(
        question_models: tuple[SimulatedQuestion, ...],
    ) -> ExpectedUtilityPlan:
        return plan_expected_utility(
            hypotheses,
            question_models,
            current_turn=current_turn,
            top_k=planning_top_k,
            widths=widths,
            retrieval_choices=retrieval_choices,
            out_of_pool_probability=float(world.out_of_pool_probability),
            protocol_confidence=float(world.assessment.confidence),
            protocol_locked=protocol_locked,
            allow_zero_width=(world.assessment.mode is ProtocolMode.EXACT),
            reretrieve_computation_cost=RERETRIEVAL_TIE_COST,
            no_question_post_ranks_by_width=tuple(
                no_question_ranks_by_width
            ),
            fallback_reason=world.assessment.fallback_reason,
        )

    try:
        incumbent = evaluate_plans(())
    except Exception:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            full_order,
            requested_top_k,
            replace(trace, fallback_reason="utility_planner_validation"),
        )

    candidate_questions = []
    if current_turn < 10:
        computation_cost = (
            0.0
            if retrieval_choices[0] is RetrievalChoice.REUSE
            else RERETRIEVAL_TIE_COST
        )
        for index, action in enumerate(eligible_actions):
            question = world.question(action)
            if question is None:
                continue
            upper_bound = _question_utility_upper_bound(
                hypotheses,
                question,
                widths=widths,
                current_turn=current_turn,
                top_k=planning_top_k,
                protocol_locked=protocol_locked,
                computation_cost=computation_cost,
            )
            candidate_questions.append((upper_bound, index, action, question))
    candidate_questions.sort(key=lambda item: (-item[0], item[1]))

    evidence_in_order = tuple(
        evidence_by_id[parent_asin] for parent_asin in full_order
    )
    simulations: list[SimulatedQuestion] = []
    simulation_count = 0
    pruned_question_count = 0
    try:
        for upper_bound, _, action, question in candidate_questions:
            if upper_bound <= incumbent.selected.value:
                pruned_question_count += 1
                continue
            required_simulations = len(question.partitions) + int(
                question.shared_reply is not None
            )
            if simulation_count + required_simulations > MAX_SIMULATED_REPLY_PARTITIONS:
                return _fail_open(
                    ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
                    full_order,
                    requested_top_k,
                    replace(
                        trace,
                        question_count=len(candidate_questions),
                        simulated_partition_count=simulation_count,
                        pruned_question_count=pruned_question_count,
                        fallback_reason="reply_simulation_budget",
                    ),
                )
            simulation, simulated = _simulate_question_model(
                state,
                action,
                question,
                current_turn=current_turn,
                intent_policy=intent_policy,
                protocol_events=events,
                candidate_ids=full_order,
                evidence=evidence_in_order,
                modeled_ids=modeled_ids,
                widths=widths,
                current_selection_by_width=current_selection_by_width,
                top_k=planning_top_k,
            )
            simulations.append(simulation)
            simulation_count += simulated
            contender = evaluate_plans((simulation,))
            if contender.selected.value > incumbent.selected.value:
                incumbent = contender
        plan = evaluate_plans(tuple(simulations))
    except Exception:
        return _fail_open(
            ProtocolDecisionStatus.FAIL_OPEN_VALIDATION,
            full_order,
            requested_top_k,
            replace(
                trace,
                question_count=len(candidate_questions),
                simulated_partition_count=simulation_count,
                pruned_question_count=pruned_question_count,
                fallback_reason="counterfactual_rerank_validation",
            ),
        )

    selected = plan.selected
    runner_up = plan.runner_up
    return ProtocolDecision(
        status=ProtocolDecisionStatus.APPLIED,
        question=selected.question,
        width=selected.width,
        value=selected.value,
        ordered_ids=full_order,
        trace=replace(
            trace,
            question_count=len(candidate_questions),
            simulated_partition_count=simulation_count,
            pruned_question_count=pruned_question_count,
            fallback_reason=plan.fallback_reason,
        ),
        retrieval=selected.retrieval,
        immediate_value=selected.immediate_reward,
        continuation_value=selected.continuation_reward,
        runner_up_question=(None if runner_up is None else runner_up.question),
        runner_up_width=(None if runner_up is None else runner_up.width),
        runner_up_value=(None if runner_up is None else runner_up.value),
    )


def _question_utility_upper_bound(
    candidates: tuple[ExpectedUtilityCandidate, ...],
    question: QuestionReplyModel,
    *,
    widths: tuple[int, ...],
    current_turn: int,
    top_k: int,
    protocol_locked: bool,
    computation_cost: float,
) -> float:
    """Return an optimistic bound used only to prune dominated questions.

    Each answer partition is allowed its best possible probability-sorted
    ranking on the next turn.  The shared-answer world receives the same
    relaxation over all surviving candidates.  Therefore no realizable exact
    rerank can exceed this value, while computing it requires no state update
    or catalog scoring.
    """

    probability_by_id = {
        candidate.candidate_id: float(candidate.probability)
        for candidate in candidates
    }
    shared_probability = float(question.shared_reply_probability)

    def optimistic_reward(candidate_ids: Sequence[str], exposed: frozenset[str]) -> float:
        probabilities = sorted(
            (
                probability_by_id[parent_asin]
                for parent_asin in candidate_ids
                if parent_asin not in exposed
            ),
            reverse=True,
        )
        return sum(
            probability * hit_utility(current_turn + 1, rank)
            for rank, probability in enumerate(probabilities[:top_k], start=1)
        )

    best = float("-inf")
    all_ids = tuple(candidate.candidate_id for candidate in candidates)
    for width in widths:
        exposed = frozenset(
            candidate.candidate_id
            for candidate in candidates
            if not protocol_locked and candidate.current_rank <= width
        )
        immediate = sum(
            float(candidate.probability)
            * hit_utility(current_turn, candidate.current_rank)
            for candidate in candidates
            if candidate.candidate_id in exposed
        )
        ordinary = sum(
            optimistic_reward(partition.candidate_ids, exposed)
            for partition in question.partitions
        )
        shared = optimistic_reward(all_ids, exposed)
        value = (
            immediate
            + ((1.0 - shared_probability) * ordinary)
            + (shared_probability * shared)
            - computation_cost
        )
        best = max(best, value)
    return best


def _simulate_question_model(
    state: IntentState,
    action: str,
    question: QuestionReplyModel,
    *,
    current_turn: int,
    intent_policy: IntentParsingPolicy,
    protocol_events: tuple[ObservedProtocolEvent, ...],
    candidate_ids: tuple[str, ...],
    evidence: tuple[ProductProtocolEvidence, ...],
    modeled_ids: tuple[str, ...],
    widths: tuple[int, ...],
    current_selection_by_width: dict[int, SlateSelection],
    top_k: int,
) -> tuple[SimulatedQuestion, int]:
    """Simulate every reply partition for one non-dominated question."""

    ordinary_scenarios: list[
        tuple[tuple[str, ...], IntentState, tuple[str, ...]]
    ] = []
    for partition in question.partitions:
        hypothetical_state, hypothetical_events, post_order = (
            _simulate_protocol_reply_rerank(
                state,
                action,
                partition.observable_reply,
                current_turn=current_turn,
                intent_policy=intent_policy,
                protocol_events=protocol_events,
                candidate_ids=candidate_ids,
                evidence=evidence,
            )
        )
        del hypothetical_events
        ordinary_scenarios.append(
            (partition.candidate_ids, hypothetical_state, post_order)
        )

    shared_scenario: tuple[IntentState, tuple[str, ...]] | None = None
    if question.shared_reply is not None:
        shared_state, shared_events, shared_order = (
            _simulate_protocol_reply_rerank(
                state,
                action,
                question.shared_reply.reply_text,
                current_turn=current_turn,
                intent_policy=intent_policy,
                protocol_events=protocol_events,
                candidate_ids=candidate_ids,
                evidence=evidence,
            )
        )
        del shared_events
        shared_scenario = (shared_state, shared_order)

    ordinary_by_width = []
    shared_by_width = []
    for width in widths:
        ordinary_ranks: dict[str, int | None] = {
            parent_asin: None for parent_asin in modeled_ids
        }
        current_selection = current_selection_by_width[width]
        for partition_ids, hypothetical_state, post_order in ordinary_scenarios:
            next_signature = _counterfactual_ranking_signature(
                hypothetical_state,
                action,
                post_order,
            )
            next_selection = select_slate_with_intent_epoch_novelty(
                current_selection.state,
                next_signature,
                post_order,
                top_k,
            ).selection
            projected = dict(
                _selected_rank_map(modeled_ids, next_selection.selected_ids)
            )
            for parent_asin in partition_ids:
                if parent_asin in ordinary_ranks:
                    ordinary_ranks[parent_asin] = projected[parent_asin]
        ordinary_by_width.append((width, tuple(ordinary_ranks.items())))

        if shared_scenario is not None:
            shared_state, shared_order = shared_scenario
            shared_signature = _counterfactual_ranking_signature(
                shared_state,
                action,
                shared_order,
            )
            shared_selection = select_slate_with_intent_epoch_novelty(
                current_selection.state,
                shared_signature,
                shared_order,
                top_k,
            ).selection
            shared_by_width.append(
                (
                    width,
                    _selected_rank_map(
                        modeled_ids,
                        shared_selection.selected_ids,
                    ),
                )
            )

    default_ordinary = ordinary_by_width[0][1]
    default_shared = shared_by_width[0][1] if shared_by_width else ()
    return (
        SimulatedQuestion(
            action,
            ordinary_post_ranks=default_ordinary,
            shared_post_ranks=default_shared,
            shared_reply_probability=float(question.shared_reply_probability),
            ordinary_post_ranks_by_width=tuple(ordinary_by_width),
            shared_post_ranks_by_width=tuple(shared_by_width),
        ),
        len(question.partitions) + int(question.shared_reply is not None),
    )


def _expected_utility_widths(
    *,
    current_turn: int,
    top_k: int,
    protocol_mode: ProtocolMode,
    protocol_confidence: float,
    protocol_locked: bool,
) -> tuple[int, ...]:
    if current_turn == 10:
        return (top_k,)
    if protocol_locked:
        return (0,)
    minimum = 1
    if protocol_mode is ProtocolMode.AMBIGUOUS:
        minimum = max(1, ceil(top_k * (1.0 - protocol_confidence)))
    widths = tuple(range(minimum, top_k + 1))
    if protocol_mode is ProtocolMode.EXACT and protocol_confidence == 1.0:
        return (0, *widths)
    return widths


def _simulate_protocol_reply_rerank(
    state: IntentState,
    action: str,
    reply_text: str,
    *,
    current_turn: int,
    intent_policy: IntentParsingPolicy,
    protocol_events: tuple[ObservedProtocolEvent, ...],
    candidate_ids: tuple[str, ...],
    evidence: tuple[ProductProtocolEvidence, ...],
) -> tuple[IntentState, tuple[ObservedProtocolEvent, ...], tuple[str, ...]]:
    next_turn = current_turn + 1
    asked_state = record_question(state, action)
    hypothetical_state = apply_user_message(
        asked_state,
        reply_text,
        next_turn,
        policy=intent_policy,
    )
    observation = recognize_protocol_observation(reply_text, next_turn)
    if observation is ProtocolObservation.UNSUPPORTED:
        raise ValueError("simulated protocol reply was not recognized")
    event = parse_protocol_event(
        reply_text,
        observation,
        next_turn,
        asked_attribute=action,
    )
    hypothetical_events = (*protocol_events, event)
    result = rank_exact_evidence(
        candidate_ids,
        evidence,
        hypothetical_state,
        protocol_events=hypothetical_events,
    )
    return hypothetical_state, hypothetical_events, result.ranked_ids


def _selected_rank_map(
    candidate_ids: Sequence[str],
    selected_ids: Sequence[str],
) -> tuple[tuple[str, int | None], ...]:
    rank_by_id = {
        parent_asin: rank
        for rank, parent_asin in enumerate(selected_ids, start=1)
    }
    return tuple(
        (parent_asin, rank_by_id.get(parent_asin))
        for parent_asin in candidate_ids
    )


def _counterfactual_ranking_signature(
    state: IntentState,
    action: str,
    ranked_ids: tuple[str, ...],
) -> tuple[object, ...]:
    return (
        state.intent_version,
        EXPECTED_UTILITY_DECISION_POLICY.value,
        action,
        ranked_ids,
    )


def _reply_key(reply: CandidateReplySignature) -> str:
    return json.dumps(
        [reply.status.value, reply.attribute, *reply.values],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _fail_open(
    status: ProtocolDecisionStatus,
    ranked_ids: tuple[str, ...],
    requested_top_k: int,
    trace: ProtocolDecisionTrace,
) -> ProtocolDecision:
    return ProtocolDecision(
        status=status,
        question=None,
        width=requested_top_k,
        value=0.0,
        ordered_ids=ranked_ids,
        trace=trace,
    )
