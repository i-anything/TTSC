"""Bounded, label-free model of the official evaluator's dialog protocol."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite


ALLOWED_ATTRIBUTES = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)
# Stable action order for world-model construction.  The first eight preserve
# the existing agent's clarification tie order; category and brand are still
# modeled because the official API accepts them, even though the current
# simulator's constraint classifier makes them candidate-independent.
PROTOCOL_ACTIONS = (
    "other",
    "feature",
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "budget",
    "category",
    "brand",
)
MAX_CONSTRAINT_CHARACTERS = 180
MAX_DISCLOSED_VALUES = 8
MAX_EVIDENCE_TEXT_CHARACTERS = 32_768
MAX_PRICE_CHARACTERS = 64
MAX_PROTOCOL_EVENTS = 10
MAX_REPLY_PAYLOAD_CHARACTERS = (2 * MAX_CONSTRAINT_CHARACTERS) + 2

_SEARCH_FIELDS = (
    "title",
    "features",
    "details",
    "description",
    "categories",
    "store",
)
_MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DisclosureCard:
    """The evaluator-visible target card reconstructed from one product."""

    target_category: str
    hard_constraints: tuple[str, ...]
    soft_preferences: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_bounded_text(self.target_category, "target_category")
        _validate_constraints(self.hard_constraints, "hard_constraints")
        _validate_constraints(self.soft_preferences, "soft_preferences")


@dataclass(frozen=True, slots=True)
class ProductProtocolEvidence:
    """Bounded catalog evidence required by the protocol-aware planner."""

    parent_asin: str
    coarse_category: str
    card: DisclosureCard
    text: str = ""
    price: str | None = None
    popularity: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent_asin, str)
            or not self.parent_asin
            or self.parent_asin != self.parent_asin.strip()
        ):
            raise ValueError("parent_asin must be a non-empty normalized string")
        if not isinstance(self.coarse_category, str):
            raise TypeError("coarse_category must be a string")
        if not isinstance(self.card, DisclosureCard):
            raise TypeError("card must be a DisclosureCard")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if len(self.text) > MAX_EVIDENCE_TEXT_CHARACTERS:
            raise ValueError("text exceeds the protocol evidence character limit")
        if self.price is not None and (
            not isinstance(self.price, str)
            or len(self.price) > MAX_PRICE_CHARACTERS
        ):
            raise ValueError("price must be bounded text or None")
        if self.popularity is not None and (
            isinstance(self.popularity, bool)
            or not isinstance(self.popularity, int)
            or self.popularity < 0
        ):
            raise ValueError("popularity must be a non-negative integer or None")


class CandidateReplyStatus(str, Enum):
    """Mutually exclusive reply classes in the official simulator."""

    DISCLOSURE = "disclosure"
    BOUNDARY_DECLINE = "boundary_decline"
    NEED_ATTRIBUTE = "need_attribute"
    NO_ADDITIONAL = "no_additional"


class ProtocolMode(str, Enum):
    """Observable operating modes for the protocol/free-form world model."""

    EXACT = "exact"
    AMBIGUOUS = "ambiguous"
    FREE_FORM = "free_form"
    RECOVERY = "recovery"


class CandidateReplayStatus(str, Enum):
    """Why a candidate transcript replay did or did not remain possible."""

    CONSISTENT = "consistent"
    CARD_MISMATCH = "card_mismatch"
    REPLY_MISMATCH = "reply_mismatch"


class ReplyMatchStatus(str, Enum):
    """Result of matching one observed reply against a question world."""

    KNOWN = "known"
    SHARED = "shared"
    UNSEEN = "unseen"


@dataclass(frozen=True, slots=True)
class CandidateReplySignature:
    """One candidate target's deterministic answer to an agent action."""

    status: CandidateReplyStatus
    attribute: str | None
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, CandidateReplyStatus):
            raise TypeError("status must be a CandidateReplyStatus")
        if self.attribute is not None and not isinstance(self.attribute, str):
            raise TypeError("attribute must be a string or None")
        _validate_constraints(self.values, "values")
        if self.status is CandidateReplyStatus.DISCLOSURE:
            if not self.attribute or not self.values:
                raise ValueError("a disclosure requires an attribute and values")
        elif self.values:
            raise ValueError("only a disclosure may contain values")

    @property
    def boundary_consumed(self) -> bool:
        return self.status is CandidateReplyStatus.BOUNDARY_DECLINE

    @property
    def reply_text(self) -> str:
        if self.status is CandidateReplyStatus.DISCLOSURE:
            return "For that, what matters is: " + "; ".join(self.values) + "."
        if self.status is CandidateReplyStatus.BOUNDARY_DECLINE:
            return (
                f"I don't have a preference for {self.attribute}; "
                "please use your judgment."
            )
        if self.status is CandidateReplyStatus.NO_ADDITIONAL:
            return f"I don't have an additional preference for {self.attribute}."
        return (
            "Those options are not quite right yet. "
            "Ask me about one specific attribute."
        )


@dataclass(frozen=True, slots=True)
class CandidateProbability:
    """One label-free target hypothesis and its retained probability mass."""

    parent_asin: str
    probability: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent_asin, str)
            or not self.parent_asin
            or self.parent_asin != self.parent_asin.strip()
        ):
            raise ValueError("parent_asin must be a non-empty normalized string")
        if isinstance(self.probability, bool) or not isinstance(
            self.probability,
            (int, float),
        ):
            raise TypeError("probability must be numeric")
        probability = float(self.probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and between zero and one")


@dataclass(frozen=True, slots=True)
class ReplyPartition:
    """Candidates producing one user-visible reply to a question."""

    observable_reply: str
    status: CandidateReplyStatus
    candidate_ids: tuple[str, ...]
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.observable_reply, str) or not self.observable_reply:
            raise ValueError("observable_reply must be a non-empty string")
        if not isinstance(self.status, CandidateReplyStatus):
            raise TypeError("status must be a CandidateReplyStatus")
        if not isinstance(self.candidate_ids, tuple) or not self.candidate_ids:
            raise ValueError("candidate_ids must be a non-empty tuple")
        if any(
            not isinstance(parent_asin, str)
            or not parent_asin
            or parent_asin != parent_asin.strip()
            for parent_asin in self.candidate_ids
        ):
            raise ValueError("candidate_ids must contain normalized strings")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique within a partition")
        if isinstance(self.probability, bool) or not isinstance(
            self.probability,
            (int, float),
        ):
            raise TypeError("probability must be numeric")
        probability = float(self.probability)
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and between zero and one")


@dataclass(frozen=True, slots=True)
class QuestionReplyModel:
    """Answer partitions and candidate-independent outcomes for one action."""

    action: str
    partitions: tuple[ReplyPartition, ...]
    shared_reply: CandidateReplySignature | None = None
    shared_reply_probability: float = 0.0
    unknown_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.action not in ALLOWED_ATTRIBUTES:
            raise ValueError("action must be an allowed protocol attribute")
        if not isinstance(self.partitions, tuple):
            raise TypeError("partitions must be an immutable tuple")
        if any(not isinstance(item, ReplyPartition) for item in self.partitions):
            raise TypeError("partitions must contain ReplyPartition values")
        observed = [
            _observable_reply_key(item.observable_reply)
            for item in self.partitions
        ]
        if len(set(observed)) != len(observed):
            raise ValueError("observable replies must be unique within a question")
        assigned = [
            parent_asin
            for item in self.partitions
            for parent_asin in item.candidate_ids
        ]
        if len(set(assigned)) != len(assigned):
            raise ValueError("candidate partitions must be disjoint")
        if self.shared_reply is not None and not isinstance(
            self.shared_reply,
            CandidateReplySignature,
        ):
            raise TypeError("shared_reply must be a CandidateReplySignature or None")
        for name, value in (
            ("shared_reply_probability", self.shared_reply_probability),
            ("unknown_probability", self.unknown_probability),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
        if (self.shared_reply is None) != (self.shared_reply_probability == 0.0):
            raise ValueError(
                "a shared reply and a positive shared probability must appear together"
            )
        total_outcome_probability = (
            sum(float(item.probability) for item in self.partitions)
            + float(self.unknown_probability)
            + float(self.shared_reply_probability)
        )
        if not isclose(total_outcome_probability, 1.0, abs_tol=1e-12):
            raise ValueError("question reply outcomes must sum to one")

    @property
    def informative(self) -> bool:
        """Whether the ordinary candidate world has multiple visible outcomes."""

        return len(self.partitions) > 1


@dataclass(frozen=True, slots=True)
class ProtocolAssessment:
    """Aggregate-only confidence derived from observable transcript behavior."""

    mode: ProtocolMode
    confidence: float
    observed_turn_count: int
    recognized_turn_count: int
    consistent_candidate_count: int
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ProtocolMode):
            raise TypeError("mode must be a ProtocolMode")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence,
            (int, float),
        ):
            raise TypeError("confidence must be numeric")
        if not isfinite(float(self.confidence)) or not 0.0 <= float(
            self.confidence
        ) <= 1.0:
            raise ValueError("confidence must be finite and between zero and one")
        for name, value in (
            ("observed_turn_count", self.observed_turn_count),
            ("recognized_turn_count", self.recognized_turn_count),
            ("consistent_candidate_count", self.consistent_candidate_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.recognized_turn_count > self.observed_turn_count:
            raise ValueError("recognized turns cannot exceed observed turns")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason
        ):
            raise ValueError("fallback_reason must be non-empty text or None")


@dataclass(frozen=True, slots=True)
class ProtocolWorldModel:
    """Bounded candidate beliefs plus counterfactual reply worlds."""

    assessment: ProtocolAssessment
    candidate_probabilities: tuple[CandidateProbability, ...]
    out_of_pool_probability: float
    questions: tuple[QuestionReplyModel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.assessment, ProtocolAssessment):
            raise TypeError("assessment must be a ProtocolAssessment")
        if not isinstance(self.candidate_probabilities, tuple) or any(
            not isinstance(item, CandidateProbability)
            for item in self.candidate_probabilities
        ):
            raise TypeError(
                "candidate_probabilities must contain CandidateProbability values"
            )
        candidate_ids = tuple(
            item.parent_asin for item in self.candidate_probabilities
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate probabilities must have unique IDs")
        if isinstance(self.out_of_pool_probability, bool) or not isinstance(
            self.out_of_pool_probability,
            (int, float),
        ):
            raise TypeError("out_of_pool_probability must be numeric")
        residual = float(self.out_of_pool_probability)
        if not isfinite(residual) or not 0.0 <= residual <= 1.0:
            raise ValueError(
                "out_of_pool_probability must be finite and between zero and one"
            )
        total = residual + sum(
            float(item.probability) for item in self.candidate_probabilities
        )
        if not isclose(total, 1.0, abs_tol=1e-12):
            raise ValueError("candidate and out-of-pool probabilities must sum to one")
        if not isinstance(self.questions, tuple) or any(
            not isinstance(item, QuestionReplyModel) for item in self.questions
        ):
            raise TypeError("questions must contain QuestionReplyModel values")
        actions = tuple(item.action for item in self.questions)
        if len(set(actions)) != len(actions):
            raise ValueError("world-model actions must be unique")
        expected_mass = 1.0 - residual
        probability_by_id = {
            item.parent_asin: float(item.probability)
            for item in self.candidate_probabilities
        }
        for question in self.questions:
            ordinary_world_probability = 1.0 - float(
                question.shared_reply_probability
            )
            partition_mass = sum(
                float(item.probability) for item in question.partitions
            )
            if not isclose(
                partition_mass,
                expected_mass * ordinary_world_probability,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "every question must partition all in-pool candidate mass"
                )
            partition_ids = tuple(
                parent_asin
                for partition in question.partitions
                for parent_asin in partition.candidate_ids
            )
            if set(partition_ids) != set(candidate_ids):
                raise ValueError(
                    "every question must cover exactly the modeled candidates"
                )
            for partition in question.partitions:
                expected_partition_probability = ordinary_world_probability * sum(
                    probability_by_id[parent_asin]
                    for parent_asin in partition.candidate_ids
                )
                if not isclose(
                    float(partition.probability),
                    expected_partition_probability,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "partition probability must equal its candidate mass"
                    )
            if not isclose(
                float(question.unknown_probability),
                residual * ordinary_world_probability,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "every question must retain scaled out-of-pool mass"
                )

    def question(self, action: str) -> QuestionReplyModel | None:
        """Return one modeled action without constructing a mutable index."""

        return next(
            (question for question in self.questions if question.action == action),
            None,
        )

    @property
    def informative_actions(self) -> tuple[str, ...]:
        return tuple(
            question.action for question in self.questions if question.informative
        )


@dataclass(frozen=True, slots=True)
class ReplyMatch:
    """Safe transition result for one observed customer reply."""

    status: ReplyMatchStatus
    action: str
    candidate_ids: tuple[str, ...]
    probability: float
    next_mode: ProtocolMode
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReplyMatchStatus):
            raise TypeError("status must be a ReplyMatchStatus")
        if self.action not in ALLOWED_ATTRIBUTES:
            raise ValueError("action must be an allowed protocol attribute")
        if not isinstance(self.candidate_ids, tuple) or any(
            not isinstance(parent_asin, str) or not parent_asin
            for parent_asin in self.candidate_ids
        ):
            raise TypeError("candidate_ids must be a tuple of non-empty strings")
        _validated_probability(self.probability, "probability")
        if not isinstance(self.next_mode, ProtocolMode):
            raise TypeError("next_mode must be a ProtocolMode")
        if self.fallback_reason is not None and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason
        ):
            raise ValueError("fallback_reason must be non-empty text or None")

    @property
    def requires_broad_retrieval(self) -> bool:
        return self.status is ReplyMatchStatus.UNSEEN


class ProtocolEventKind(str, Enum):
    """Observed official-protocol events retained independently of intent slots."""

    INITIAL_BROWSING = "initial_browsing"
    INITIAL_EXPLICIT = "initial_explicit"
    INITIAL_TENTATIVE = "initial_tentative"
    DISCLOSURE = "disclosure"
    OVERRIDE = "override"
    NO_ADDITIONAL = "no_additional"
    BOUNDARY_DECLINE = "boundary_decline"
    NEED_ATTRIBUTE = "need_attribute"


@dataclass(frozen=True, slots=True)
class ObservedProtocolEvent:
    """One bounded message event needed to replay candidate disclosures."""

    turn: int
    kind: ProtocolEventKind
    attribute: str | None = None
    values: tuple[str, ...] = ()
    reply_payload: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.turn, bool)
            or not isinstance(self.turn, int)
            or not 1 <= self.turn <= MAX_PROTOCOL_EVENTS
        ):
            raise ValueError("turn must be an integer from 1 through 10")
        if not isinstance(self.kind, ProtocolEventKind):
            raise TypeError("kind must be a ProtocolEventKind")
        if self.attribute is not None and self.attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError("attribute must be an allowed protocol attribute")
        _validate_constraints(self.values, "values")
        if self.reply_payload is not None and (
            not isinstance(self.reply_payload, str)
            or not self.reply_payload
            or len(self.reply_payload) > MAX_REPLY_PAYLOAD_CHARACTERS
        ):
            raise ValueError("reply_payload must be bounded non-empty text or None")

        initial = self.kind in {
            ProtocolEventKind.INITIAL_BROWSING,
            ProtocolEventKind.INITIAL_EXPLICIT,
            ProtocolEventKind.INITIAL_TENTATIVE,
        }
        if initial != (self.turn == 1):
            raise ValueError("only an initial event may occur on turn one")
        if self.kind in {
            ProtocolEventKind.INITIAL_EXPLICIT,
            ProtocolEventKind.INITIAL_TENTATIVE,
            ProtocolEventKind.OVERRIDE,
        }:
            if self.attribute is not None or len(self.values) != 1:
                raise ValueError("value events require exactly one unlabeled value")
        elif self.kind is ProtocolEventKind.DISCLOSURE:
            if self.attribute is None or (not self.values and not self.reply_payload):
                raise ValueError(
                    "a disclosure requires an attribute and serialized values"
                )
        elif self.kind in {
            ProtocolEventKind.NO_ADDITIONAL,
            ProtocolEventKind.BOUNDARY_DECLINE,
        }:
            if self.attribute is None or self.values:
                raise ValueError("a negative reply requires only an attribute")
        elif self.attribute is not None or self.values:
            raise ValueError("this event kind cannot carry an attribute or values")
        if (
            self.kind is not ProtocolEventKind.DISCLOSURE
            and self.reply_payload is not None
        ):
            raise ValueError("only a disclosure may carry a reply_payload")

    @property
    def serialized_reply_values(self) -> str | None:
        """Return the evaluator-visible payload without guessing semicolon splits."""

        if self.kind is not ProtocolEventKind.DISCLOSURE:
            return None
        if self.reply_payload is not None:
            return self.reply_payload
        return "; ".join(self.values)


@dataclass(frozen=True, slots=True)
class CandidateTranscriptReplay:
    """Candidate-specific state after replaying recognized protocol events."""

    status: CandidateReplayStatus
    disclosed_values: tuple[str, ...]
    boundary_consumed: bool
    matched_event_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, CandidateReplayStatus):
            raise TypeError("status must be a CandidateReplayStatus")
        _validate_disclosed_values(self.disclosed_values)
        if not isinstance(self.boundary_consumed, bool):
            raise TypeError("boundary_consumed must be a boolean")
        if (
            isinstance(self.matched_event_count, bool)
            or not isinstance(self.matched_event_count, int)
            or self.matched_event_count < 0
        ):
            raise ValueError("matched_event_count must be a non-negative integer")

    @property
    def consistent(self) -> bool:
        return self.status is CandidateReplayStatus.CONSISTENT


def build_disclosure_card(product: Mapping[str, object]) -> DisclosureCard:
    """Reconstruct the hidden card using the official evaluator's exact order."""

    title = _clean_constraint(str(product.get("title") or "product"))
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = _searchable_text(product)
    material = _MATERIAL_RE.search(corpus)
    color = _COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")

    cleaned = list(
        dict.fromkeys(
            value
            for item in candidates
            if (value := _clean_constraint(item))
        )
    )
    if not cleaned:
        cleaned = [title]
    hard = tuple(cleaned[:2])
    soft = tuple(cleaned[2:4] or cleaned[:1])
    return DisclosureCard(title, hard, soft)


def build_product_protocol_evidence(
    product: Mapping[str, object],
    *,
    include_text: bool = True,
) -> ProductProtocolEvidence:
    """Create one immutable, bounded protocol hypothesis from a catalog row."""

    if not isinstance(include_text, bool):
        raise TypeError("include_text must be a boolean")

    parent_asin = str(product.get("parent_asin") or "").strip()
    category_value = product.get("categories")
    category_values = (
        [str(value) for value in category_value]
        if isinstance(category_value, list)
        else []
    )
    raw_price = product.get("price")
    price = None if raw_price in (None, "") else str(raw_price)[:MAX_PRICE_CHARACTERS]
    raw_popularity = product.get("rating_number")
    popularity = (
        raw_popularity
        if isinstance(raw_popularity, int)
        and not isinstance(raw_popularity, bool)
        and raw_popularity >= 0
        else None
    )
    return ProductProtocolEvidence(
        parent_asin=parent_asin,
        coarse_category=coarse_category(category_values),
        card=build_disclosure_card(product),
        text=(
            _searchable_text(product)[:MAX_EVIDENCE_TEXT_CHARACTERS]
            if include_text
            else ""
        ),
        price=price,
        popularity=popularity,
    )


def coarse_category(values: Sequence[str]) -> str:
    """Mirror the evaluator's lossy category phrase exposed to the agent."""

    excluded = {
        "clothing",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    """Return the attribute bucket selected by the official evaluator."""

    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in _MATERIALS):
        return "material"
    if any(
        word in lowered
        for word in ("color", "black", "white", "blue", "red", "pink", "green")
    ):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(
        word in lowered
        for word in ("department", "style", "fit", "sleeve", "neck")
    ):
        return "style"
    if any(
        word in lowered
        for word in ("hiking", "running", "gym", "winter", "outdoor", "work")
    ):
        return "use_case"
    return "feature"


def remaining_reply(
    card: DisclosureCard,
    ask_attribute: object,
    disclosed_values: Iterable[str],
    *,
    boundary_pending: bool = False,
) -> CandidateReplySignature:
    """Predict the evaluator reply for one candidate target and agent action."""

    if not isinstance(card, DisclosureCard):
        raise TypeError("card must be a DisclosureCard")
    if not isinstance(boundary_pending, bool):
        raise TypeError("boundary_pending must be a boolean")
    disclosed = _bounded_disclosed_values(disclosed_values)
    attribute = ask_attribute if isinstance(ask_attribute, str) else None

    if boundary_pending and attribute:
        return CandidateReplySignature(
            CandidateReplyStatus.BOUNDARY_DECLINE,
            attribute,
        )
    if not attribute:
        return CandidateReplySignature(
            CandidateReplyStatus.NEED_ATTRIBUTE,
            None,
        )
    if attribute not in ALLOWED_ATTRIBUTES:
        attribute = "other"

    matches = tuple(
        value
        for value in (*card.hard_constraints, *card.soft_preferences)
        if value not in disclosed
        and (attribute == "other" or classify_constraint(value) == attribute)
    )[:2]
    if not matches:
        return CandidateReplySignature(
            CandidateReplyStatus.NO_ADDITIONAL,
            attribute,
        )
    return CandidateReplySignature(
        CandidateReplyStatus.DISCLOSURE,
        attribute,
        matches,
    )


def replay_protocol_transcript(
    card: DisclosureCard,
    protocol_events: Sequence[ObservedProtocolEvent],
) -> CandidateTranscriptReplay:
    """Replay recognized messages against one candidate without guessing labels.

    Disclosure payloads are compared in their serialized, user-visible form.
    This deliberately treats ``("x; y",)`` and ``("x", "y")`` as the same
    observation when the simulator would render the same text.
    """

    if not isinstance(card, DisclosureCard):
        raise TypeError("card must be a DisclosureCard")
    events = _validated_protocol_events(protocol_events)
    hard = card.hard_constraints
    disclosed: list[str] = []
    boundary_consumed = False
    matched = 0
    for event in events:
        if event.kind in {
            ProtocolEventKind.INITIAL_BROWSING,
            ProtocolEventKind.NEED_ATTRIBUTE,
        }:
            matched += 1
            continue
        if event.kind is ProtocolEventKind.INITIAL_TENTATIVE:
            soft = card.soft_preferences
            expected = (_normalized_protocol_value(soft[-1]),) if soft else ()
            actual = tuple(
                _normalized_protocol_value(value) for value in event.values
            )
            if not soft or actual != expected:
                return CandidateTranscriptReplay(
                    CandidateReplayStatus.CARD_MISMATCH,
                    tuple(disclosed),
                    boundary_consumed,
                    matched,
                )
            matched += 1
            continue
        if event.kind is ProtocolEventKind.BOUNDARY_DECLINE:
            boundary_consumed = True
            matched += 1
            continue
        if event.kind in {
            ProtocolEventKind.INITIAL_EXPLICIT,
            ProtocolEventKind.OVERRIDE,
        }:
            expected = (_normalized_protocol_value(hard[0]),) if hard else ()
            actual = tuple(
                _normalized_protocol_value(value) for value in event.values
            )
            if not hard or actual != expected:
                return CandidateTranscriptReplay(
                    CandidateReplayStatus.CARD_MISMATCH,
                    tuple(disclosed),
                    boundary_consumed,
                    matched,
                )
            if hard[0] not in disclosed:
                disclosed.append(hard[0])
            matched += 1
            continue

        signature = remaining_reply(card, event.attribute, disclosed)
        if event.kind is ProtocolEventKind.DISCLOSURE:
            actual = _normalized_protocol_value(
                event.serialized_reply_values or ""
            )
            expected = _normalized_protocol_value("; ".join(signature.values))
            if (
                signature.status is not CandidateReplyStatus.DISCLOSURE
                or actual != expected
            ):
                return CandidateTranscriptReplay(
                    CandidateReplayStatus.REPLY_MISMATCH,
                    tuple(disclosed),
                    boundary_consumed,
                    matched,
                )
            for value in signature.values:
                if value not in disclosed:
                    disclosed.append(value)
        elif (
            event.kind is ProtocolEventKind.NO_ADDITIONAL
            and signature.status is not CandidateReplyStatus.NO_ADDITIONAL
        ):
            return CandidateTranscriptReplay(
                CandidateReplayStatus.REPLY_MISMATCH,
                tuple(disclosed),
                boundary_consumed,
                matched,
            )
        matched += 1
    return CandidateTranscriptReplay(
        CandidateReplayStatus.CONSISTENT,
        tuple(disclosed),
        boundary_consumed,
        matched,
    )


def build_protocol_world_model(
    evidence: Sequence[ProductProtocolEvidence],
    *,
    protocol_events: Sequence[ObservedProtocolEvent] = (),
    observed_turn_count: int | None = None,
    candidate_weights: Mapping[str, float] | None = None,
    out_of_pool_probability: float | None = None,
    actions: Sequence[str] = PROTOCOL_ACTIONS,
    boundary_pending: bool | None = None,
) -> ProtocolWorldModel:
    """Build bounded candidate beliefs and observable answer partitions.

    No target labels, session identifiers, or public examples enter this model.
    If any observed turn is not represented by a recognized protocol event, or
    if no weighted in-pool candidate can explain the transcript, it returns a
    recovery-safe model whose entire probability is out of pool.
    """

    candidates = _validated_world_evidence(evidence)
    events = _validated_protocol_events(protocol_events)
    action_tuple = _validated_actions(actions)
    latest_event_turn = events[-1].turn if events else 0
    if observed_turn_count is None:
        observed_turns = latest_event_turn
    else:
        if (
            isinstance(observed_turn_count, bool)
            or not isinstance(observed_turn_count, int)
            or not 0 <= observed_turn_count <= MAX_PROTOCOL_EVENTS
        ):
            raise ValueError("observed_turn_count must be an integer from zero to ten")
        observed_turns = observed_turn_count
    if latest_event_turn > observed_turns:
        raise ValueError("observed_turn_count cannot precede the latest event")

    if boundary_pending is None:
        pending_boundary = _boundary_world_still_possible(events)
    elif isinstance(boundary_pending, bool):
        pending_boundary = boundary_pending
    else:
        raise TypeError("boundary_pending must be a boolean or None")

    raw_weights = _validated_candidate_weights(candidates, candidate_weights)
    explicit_residual = (
        None
        if out_of_pool_probability is None
        else _validated_probability(
            out_of_pool_probability,
            "out_of_pool_probability",
        )
    )
    replays = tuple(
        (candidate, replay_protocol_transcript(candidate.card, events))
        for candidate in candidates
    )
    supported = tuple(
        (candidate, replay, raw_weights[candidate.parent_asin])
        for candidate, replay in replays
        if replay.consistent and raw_weights[candidate.parent_asin] > 0.0
    )
    recognized_turns = len(events)
    if observed_turns == 0:
        mode = ProtocolMode.AMBIGUOUS
        confidence = 0.5
        fallback_reason = "no_observed_turn"
    elif recognized_turns != observed_turns:
        mode = ProtocolMode.FREE_FORM
        confidence = 0.0
        fallback_reason = "unrecognized_or_free_form_turn"
    elif not supported:
        mode = ProtocolMode.RECOVERY
        confidence = 0.0
        fallback_reason = "no_in_pool_protocol_support"
    elif explicit_residual == 1.0:
        mode = ProtocolMode.RECOVERY
        confidence = 0.0
        fallback_reason = "all_probability_out_of_pool"
    elif pending_boundary:
        mode = ProtocolMode.AMBIGUOUS
        confidence = 0.5
        fallback_reason = "initial_browsing_boundary_ambiguity"
    else:
        mode = ProtocolMode.EXACT
        confidence = 1.0
        fallback_reason = None

    assessment = ProtocolAssessment(
        mode=mode,
        confidence=confidence,
        observed_turn_count=observed_turns,
        recognized_turn_count=recognized_turns,
        consistent_candidate_count=len(supported),
        fallback_reason=fallback_reason,
    )
    if mode in {ProtocolMode.FREE_FORM, ProtocolMode.RECOVERY}:
        residual = 1.0
    elif explicit_residual is None:
        laplace_residual = 1.0 / (len(supported) + 1.0)
        residual = (
            max(0.5, laplace_residual)
            if mode is ProtocolMode.AMBIGUOUS
            else laplace_residual
        )
    else:
        residual = explicit_residual

    candidate_mass = 1.0 - residual
    total_raw_weight = sum(weight for _, _, weight in supported)
    probabilities = tuple(
        CandidateProbability(
            candidate.parent_asin,
            candidate_mass * (weight / total_raw_weight),
        )
        for candidate, _, weight in supported
        if candidate_mass > 0.0
    )
    replay_by_id = {
        candidate.parent_asin: replay for candidate, replay, _ in supported
    }
    evidence_by_id = {
        candidate.parent_asin: candidate for candidate, _, _ in supported
    }
    mass_by_id = {
        item.parent_asin: float(item.probability) for item in probabilities
    }
    questions: list[QuestionReplyModel] = []
    for action in action_tuple:
        shared_probability = (
            0.5 if pending_boundary and probabilities else 0.0
        )
        ordinary_world_probability = 1.0 - shared_probability
        grouped_ids: dict[str, list[str]] = {}
        grouped_reply: dict[str, str] = {}
        grouped_status: dict[str, CandidateReplyStatus] = {}
        grouped_mass: dict[str, float] = {}
        for probability in probabilities:
            parent_asin = probability.parent_asin
            signature = remaining_reply(
                evidence_by_id[parent_asin].card,
                action,
                replay_by_id[parent_asin].disclosed_values,
            )
            observable = _observable_reply_key(signature.reply_text)
            grouped_ids.setdefault(observable, []).append(parent_asin)
            grouped_reply.setdefault(observable, signature.reply_text)
            grouped_status.setdefault(observable, signature.status)
            grouped_mass[observable] = (
                grouped_mass.get(observable, 0.0)
                + (ordinary_world_probability * mass_by_id[parent_asin])
            )
        partitions = tuple(
            ReplyPartition(
                observable_reply=grouped_reply[observable],
                status=grouped_status[observable],
                candidate_ids=tuple(parent_asins),
                probability=grouped_mass[observable],
            )
            for observable, parent_asins in grouped_ids.items()
        )
        shared_reply = (
            CandidateReplySignature(
                CandidateReplyStatus.BOUNDARY_DECLINE,
                action,
            )
            if pending_boundary and probabilities
            else None
        )
        questions.append(
            QuestionReplyModel(
                action=action,
                partitions=partitions,
                shared_reply=shared_reply,
                shared_reply_probability=shared_probability,
                unknown_probability=(
                    ordinary_world_probability * residual
                ),
            )
        )
    return ProtocolWorldModel(
        assessment=assessment,
        candidate_probabilities=probabilities,
        out_of_pool_probability=residual,
        questions=tuple(questions),
    )


def eligible_protocol_actions(
    world: ProtocolWorldModel,
    *,
    asked_actions: Iterable[str] = (),
    resolved_actions: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return non-repeated, unresolved actions with candidate information gain."""

    if not isinstance(world, ProtocolWorldModel):
        raise TypeError("world must be a ProtocolWorldModel")
    asked = _validated_action_set(asked_actions, "asked_actions")
    resolved = _validated_action_set(resolved_actions, "resolved_actions")
    if world.assessment.mode in {ProtocolMode.FREE_FORM, ProtocolMode.RECOVERY}:
        return ()
    return tuple(
        question.action
        for question in world.questions
        if question.action not in asked
        and question.action not in resolved
        and question.informative
    )


def project_protocol_world_model(
    world: ProtocolWorldModel,
    retained_candidate_ids: Sequence[str],
) -> ProtocolWorldModel:
    """Move omitted in-pool belief mass into explicit unknown probability.

    This is a computational projection, not a target filter.  Candidate order
    and absolute probabilities are preserved for the retained hypotheses;
    every omitted hypothesis becomes uncertainty.  Reply partitions are
    filtered consistently and keep the original shared-world probability.
    """

    if not isinstance(world, ProtocolWorldModel):
        raise TypeError("world must be a ProtocolWorldModel")
    if isinstance(retained_candidate_ids, (str, bytes)) or not isinstance(
        retained_candidate_ids,
        Sequence,
    ):
        raise TypeError("retained_candidate_ids must be a sequence")
    retained = tuple(retained_candidate_ids)
    if not retained or any(
        not isinstance(parent_asin, str)
        or not parent_asin
        or parent_asin != parent_asin.strip()
        for parent_asin in retained
    ):
        raise ValueError("retained candidate IDs must be non-empty strings")
    if len(set(retained)) != len(retained):
        raise ValueError("retained candidate IDs must be unique")
    probability_by_id = {
        item.parent_asin: float(item.probability)
        for item in world.candidate_probabilities
    }
    if not set(retained).issubset(probability_by_id):
        raise ValueError("retained candidate IDs must belong to the world")
    retained_set = frozenset(retained)
    probabilities = tuple(
        item
        for item in world.candidate_probabilities
        if item.parent_asin in retained_set
    )
    if tuple(item.parent_asin for item in probabilities) != retained:
        raise ValueError("retained candidate IDs must preserve world order")
    retained_mass = sum(float(item.probability) for item in probabilities)
    residual = 1.0 - retained_mass

    questions = []
    for question in world.questions:
        ordinary_probability = 1.0 - float(
            question.shared_reply_probability
        )
        partitions = []
        for partition in question.partitions:
            candidate_ids = tuple(
                parent_asin
                for parent_asin in partition.candidate_ids
                if parent_asin in retained_set
            )
            if not candidate_ids:
                continue
            partitions.append(
                ReplyPartition(
                    observable_reply=partition.observable_reply,
                    status=partition.status,
                    candidate_ids=candidate_ids,
                    probability=(
                        ordinary_probability
                        * sum(probability_by_id[value] for value in candidate_ids)
                    ),
                )
            )
        questions.append(
            QuestionReplyModel(
                action=question.action,
                partitions=tuple(partitions),
                shared_reply=question.shared_reply,
                shared_reply_probability=float(
                    question.shared_reply_probability
                ),
                unknown_probability=ordinary_probability * residual,
            )
        )
    return ProtocolWorldModel(
        assessment=world.assessment,
        candidate_probabilities=probabilities,
        out_of_pool_probability=residual,
        questions=tuple(questions),
    )


def match_protocol_reply(
    world: ProtocolWorldModel,
    action: str,
    reply_text: str,
) -> ReplyMatch:
    """Match a visible reply or request broad recovery for an unseen outcome."""

    if not isinstance(world, ProtocolWorldModel):
        raise TypeError("world must be a ProtocolWorldModel")
    if action not in ALLOWED_ATTRIBUTES:
        raise ValueError("action must be an allowed protocol attribute")
    if not isinstance(reply_text, str):
        raise TypeError("reply_text must be a string")
    question = world.question(action)
    observed = _observable_reply_key(reply_text)
    if question is not None:
        for partition in question.partitions:
            if observed == _observable_reply_key(partition.observable_reply):
                return ReplyMatch(
                    ReplyMatchStatus.KNOWN,
                    action,
                    partition.candidate_ids,
                    float(partition.probability),
                    world.assessment.mode,
                )
        if (
            question.shared_reply is not None
            and observed == _observable_reply_key(question.shared_reply.reply_text)
        ):
            return ReplyMatch(
                ReplyMatchStatus.SHARED,
                action,
                tuple(
                    item.parent_asin for item in world.candidate_probabilities
                ),
                float(question.shared_reply_probability),
                ProtocolMode.EXACT,
            )
    return ReplyMatch(
        ReplyMatchStatus.UNSEEN,
        action,
        (),
        float(world.out_of_pool_probability),
        ProtocolMode.RECOVERY,
        "unrecognized_reply",
    )


def _searchable_text(product: Mapping[str, object]) -> str:
    parts: list[str] = []
    for field in _SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[
        :MAX_CONSTRAINT_CHARACTERS
    ].rstrip()


def _bounded_disclosed_values(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, str):
        raise TypeError("disclosed_values must be an iterable of strings")
    retained = tuple(values)
    if len(retained) > MAX_DISCLOSED_VALUES:
        raise ValueError(
            "disclosed_values must contain at most "
            f"{MAX_DISCLOSED_VALUES} items"
        )
    if any(not isinstance(value, str) for value in retained):
        raise TypeError("disclosed_values must contain only strings")
    return frozenset(retained)


def _validated_protocol_events(
    events: Sequence[ObservedProtocolEvent],
) -> tuple[ObservedProtocolEvent, ...]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("protocol_events must be a sequence")
    retained = tuple(events)
    if len(retained) > MAX_PROTOCOL_EVENTS:
        raise ValueError("at most ten protocol events are supported")
    if any(not isinstance(item, ObservedProtocolEvent) for item in retained):
        raise TypeError("protocol_events must contain ObservedProtocolEvent values")
    turns = tuple(item.turn for item in retained)
    if turns != tuple(sorted(set(turns))):
        raise ValueError("protocol event turns must be unique and increasing")
    if retained and retained[0].kind not in {
        ProtocolEventKind.INITIAL_BROWSING,
        ProtocolEventKind.INITIAL_EXPLICIT,
        ProtocolEventKind.INITIAL_TENTATIVE,
    }:
        raise ValueError("a protocol transcript must begin with an initial event")
    if not retained:
        return retained

    initial_kind = retained[0].kind
    boundary_indexes = tuple(
        index
        for index, event in enumerate(retained)
        if event.kind is ProtocolEventKind.BOUNDARY_DECLINE
    )
    if boundary_indexes:
        if initial_kind is not ProtocolEventKind.INITIAL_BROWSING:
            raise ValueError("only a browsing transcript may contain a boundary decline")
        if len(boundary_indexes) > 1:
            raise ValueError("a boundary decline may occur at most once")
        boundary_index = boundary_indexes[0]
        if any(
            event.kind is not ProtocolEventKind.NEED_ATTRIBUTE
            for event in retained[1:boundary_index]
        ):
            raise ValueError("a boundary decline must precede ordinary replies")

    override_events = tuple(
        event for event in retained if event.kind is ProtocolEventKind.OVERRIDE
    )
    if override_events:
        if initial_kind is not ProtocolEventKind.INITIAL_TENTATIVE:
            raise ValueError("only a tentative transcript may contain an override")
        if len(override_events) > 1:
            raise ValueError("an override may occur at most once")
        if override_events[0].turn not in {3, 4}:
            raise ValueError("the official override may occur only on turn three or four")
    return retained


def _validated_world_evidence(
    evidence: Sequence[ProductProtocolEvidence],
) -> tuple[ProductProtocolEvidence, ...]:
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence")
    candidates = tuple(evidence)
    if len(candidates) > 200:
        raise ValueError("the protocol world model supports at most 200 candidates")
    if any(not isinstance(item, ProductProtocolEvidence) for item in candidates):
        raise TypeError("evidence must contain ProductProtocolEvidence values")
    identifiers = tuple(item.parent_asin for item in candidates)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("protocol evidence candidate IDs must be unique")
    return candidates


def _validated_actions(actions: Sequence[str]) -> tuple[str, ...]:
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise TypeError("actions must be a sequence")
    retained = tuple(actions)
    if any(action not in ALLOWED_ATTRIBUTES for action in retained):
        raise ValueError("actions must contain only allowed protocol attributes")
    if len(set(retained)) != len(retained):
        raise ValueError("actions must be unique")
    return retained


def _validated_action_set(values: Iterable[str], name: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of actions")
    retained = tuple(values)
    if any(value not in ALLOWED_ATTRIBUTES for value in retained):
        raise ValueError(f"{name} must contain only allowed protocol attributes")
    return frozenset(retained)


def _validated_candidate_weights(
    candidates: tuple[ProductProtocolEvidence, ...],
    candidate_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    identifiers = tuple(item.parent_asin for item in candidates)
    if candidate_weights is None:
        return {parent_asin: 1.0 for parent_asin in identifiers}
    if not isinstance(candidate_weights, Mapping):
        raise TypeError("candidate_weights must be a mapping or None")
    unknown = set(candidate_weights).difference(identifiers)
    if unknown:
        raise ValueError("candidate_weights contains IDs outside the evidence pool")
    result: dict[str, float] = {}
    for parent_asin in identifiers:
        value = candidate_weights.get(parent_asin, 0.0)
        result[parent_asin] = _validated_nonnegative_weight(
            value,
            "candidate weight",
        )
    return result


def _validated_probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    probability = float(value)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return probability


def _validated_nonnegative_weight(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    weight = float(value)
    if not isfinite(weight) or weight < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return weight


def _boundary_world_still_possible(
    events: tuple[ObservedProtocolEvent, ...],
) -> bool:
    if not events or events[0].kind is not ProtocolEventKind.INITIAL_BROWSING:
        return False
    return not any(
        event.kind
        in {
            ProtocolEventKind.BOUNDARY_DECLINE,
            ProtocolEventKind.DISCLOSURE,
            ProtocolEventKind.NO_ADDITIONAL,
            ProtocolEventKind.OVERRIDE,
        }
        for event in events[1:]
    )


def _normalized_protocol_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n").casefold()


def _observable_reply_key(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _validate_disclosed_values(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError("disclosed_values must be a tuple")
    if len(values) > MAX_DISCLOSED_VALUES:
        raise ValueError(
            "disclosed_values must contain at most "
            f"{MAX_DISCLOSED_VALUES} items"
        )
    for value in values:
        _validate_bounded_text(value, "disclosed_values")


def _validate_constraints(values: tuple[str, ...], name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(values) > 2:
        raise ValueError(f"{name} must contain at most two values")
    for value in values:
        _validate_bounded_text(value, name)


def _validate_bounded_text(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} values must be strings")
    if len(value) > MAX_CONSTRAINT_CHARACTERS:
        raise ValueError(
            f"{name} values must not exceed {MAX_CONSTRAINT_CHARACTERS} characters"
        )
