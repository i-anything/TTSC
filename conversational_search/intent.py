"""Deterministic intent state and reduction of user messages.

``IntentState`` is the immutable, session-local record of the conversation:
the active category, positive requirements with value provenance and
ordinal importance, exclusions, no-preference attributes, and the last
asked attribute.  Each user message reduces to a new state through
``apply_user_message`` instead of mutating the old one; reductions are
classified by ``IntentReductionStatus`` and validated for turn order,
types, and bounded collections.  Unsupported prose is preserved as soft
free-text evidence rather than discarded.

The module also renders the per-route queries (``render_lexical_query``
for SQLite FTS5 BM25, ``render_dense_query`` for the BGE encoder) and the
bounded parsing behaviors exposed by ``IntentParsingPolicy``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal


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

RequirementSource = Literal[
    "initial_explicit",
    "initial_tentative",
    "answer",
    "override",
    "free_text",
]
RequirementStrength = Literal["hard", "soft"]


class RequirementImportance(str, Enum):
    """Ordinal user importance, independent of interpretation confidence."""

    MUST = "must"
    SHOULD = "should"
    PREFER = "prefer"

_DEFAULT_REQUIREMENT_STRENGTH: dict[RequirementSource, RequirementStrength] = {
    "initial_explicit": "hard",
    "initial_tentative": "soft",
    "answer": "hard",
    "override": "hard",
    "free_text": "soft",
}

_DEFAULT_REQUIREMENT_IMPORTANCE: dict[
    RequirementSource,
    RequirementImportance,
] = {
    "initial_explicit": RequirementImportance.MUST,
    "initial_tentative": RequirementImportance.PREFER,
    "answer": RequirementImportance.SHOULD,
    "override": RequirementImportance.MUST,
    "free_text": RequirementImportance.PREFER,
}
_NEGATED_IMPORTANCE_RE = re.compile(
    r"\b(?:not|no)\s+(?:(?:a|an)\s+)?(?:must(?:[- ]have)?|required|"
    r"requirement|needed?|important|preference|preferred)\b",
    re.IGNORECASE,
)
_IMPORTANCE_CUE_PATTERNS: tuple[
    tuple[RequirementImportance, re.Pattern[str]],
    ...,
] = (
    (
        RequirementImportance.MUST,
        re.compile(
            r"^\s*(?:i\s+)?(?:must|need|require)(?:\s+it)?(?:\s+to)?"
            r"(?:\s+be|\s+have)?\s+(?P<payload>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.MUST,
        re.compile(
            r"^\s*(?:must[- ]have|required|non[- ]?negotiable)"
            r"(?:\s*[:\u2014-]\s*|\s+)(?P<payload>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.MUST,
        re.compile(
            r"^\s*(?P<payload>.+?)\s+(?:is\s+)?(?:required|"
            r"non[- ]?negotiable|a\s+must(?:[- ]have)?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.SHOULD,
        re.compile(
            r"^\s*(?:i\s+)?strongly\s+prefer(?:\s+it)?(?:\s+to)?"
            r"(?:\s+be|\s+have)?\s+(?P<payload>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.SHOULD,
        re.compile(
            r"^\s*(?:important)(?:\s*[:\u2014-]\s*|\s+)"
            r"(?P<payload>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.SHOULD,
        re.compile(
            r"^\s*(?P<payload>.+?)\s+is\s+important\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.SHOULD,
        re.compile(
            r"^\s*(?:it\s+)?should\s+(?:be|have)\s+"
            r"(?P<payload>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.PREFER,
        re.compile(
            r"^\s*(?:ideally|maybe|perhaps|preferably|tentatively)"
            r"(?:\s*[:\u2014-]\s*|\s+)(?P<payload>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.PREFER,
        re.compile(
            r"^\s*(?:i\s+)?(?:prefer|would\s+like)(?:\s+it)?"
            r"(?:\s+to)?(?:\s+be|\s+have)?\s+(?P<payload>.+?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        RequirementImportance.PREFER,
        re.compile(
            r"^\s*(?P<payload>.+?)\s+would\s+be\s+"
            r"(?:nice|ideal)\s*$",
            re.IGNORECASE,
        ),
    ),
)
_MAXIMUM_BUDGET_RE = re.compile(
    r"(?:\bunder\b|\bless\s+than\b|\bat\s+most\b|\bmaximum\b|\bmax\b|"
    r"\bno\s+more\s+than\b|<=)\s*\$?\s*\d",
    re.IGNORECASE,
)


def _importance_signal(
    value: str,
) -> tuple[RequirementImportance | None, str]:
    stripped = value.strip()
    if _NEGATED_IMPORTANCE_RE.search(stripped):
        return None, stripped
    for importance, pattern in _IMPORTANCE_CUE_PATTERNS:
        match = pattern.fullmatch(stripped)
        if match is not None:
            payload = match.group("payload").strip()
            if payload:
                return importance, payload
    return None, stripped


def requirement_semantic_payload(value: str) -> str:
    """Remove only an anchored importance cue, retaining the requested value."""

    if not isinstance(value, str):
        raise TypeError("requirement value must be a string")
    return _importance_signal(value)[1]


def infer_requirement_importance(
    value: str,
    source: RequirementSource,
    attribute: str | None,
) -> RequirementImportance:
    """Infer one bounded ordinal level from explicit language and provenance."""

    if not isinstance(value, str):
        raise TypeError("requirement value must be a string")
    if source not in _DEFAULT_REQUIREMENT_IMPORTANCE:
        raise ValueError(f"unsupported requirement source: {source!r}")
    if attribute == "budget" and _MAXIMUM_BUDGET_RE.search(value):
        return RequirementImportance.MUST
    explicit, _payload = _importance_signal(value)
    if explicit is not None:
        return explicit
    return _DEFAULT_REQUIREMENT_IMPORTANCE[source]


class IntentParsingPolicy(str, Enum):
    """Reversible intent reducers used by the Phase 6 robustness ablation."""

    CANONICAL = "canonical"
    ROBUST = "robust"
    LOSSLESS_MULTI_SLOT = "lossless_multislot"


CANONICAL_INTENT_POLICY = IntentParsingPolicy.CANONICAL
ROBUST_INTENT_POLICY = IntentParsingPolicy.ROBUST
LOSSLESS_MULTI_SLOT_INTENT_POLICY = IntentParsingPolicy.LOSSLESS_MULTI_SLOT


class IntentReductionStatus(str, Enum):
    """Fixed-cardinality outcomes for the bounded multi-slot reducer."""

    BASELINE_POLICY = "baseline_policy"
    APPLIED = "applied"
    SINGLE_SLOT = "single_slot"
    AMBIGUOUS = "ambiguous"
    BOUNDS = "bounds"
    VALIDATION_FALLBACK = "validation_fallback"


@dataclass(frozen=True, slots=True)
class Requirement:
    """One active positive requirement with enough provenance to supersede safely."""

    value: str
    source: RequirementSource
    turn: int
    attribute: str | None = None
    strength: RequirementStrength | None = None
    importance: RequirementImportance | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, str)
            or self.source not in _DEFAULT_REQUIREMENT_STRENGTH
        ):
            raise ValueError(f"unsupported requirement source: {self.source!r}")
        if self.strength is None:
            strength = _DEFAULT_REQUIREMENT_STRENGTH[self.source]
            object.__setattr__(self, "strength", strength)
        elif self.strength not in {"hard", "soft"}:
            raise ValueError("strength must be 'hard' or 'soft'")
        if self.source == "free_text" and self.strength != "soft":
            raise ValueError("free_text requirements must remain soft")
        if self.importance is None:
            importance = infer_requirement_importance(
                self.value,
                self.source,
                self.attribute,
            )
            object.__setattr__(self, "importance", importance)
        elif not isinstance(self.importance, RequirementImportance):
            raise TypeError("importance must be a RequirementImportance")
        if (
            self.attribute == "budget"
            and _MAXIMUM_BUDGET_RE.search(self.value)
            and self.importance is not RequirementImportance.MUST
        ):
            object.__setattr__(self, "importance", RequirementImportance.MUST)


@dataclass(frozen=True, slots=True)
class IntentState:
    """Immutable, session-local retrieval state."""

    category: str | None = None
    requirements: tuple[Requirement, ...] = ()
    excluded: tuple[str, ...] = ()
    no_preference: frozenset[str] = frozenset()
    asked_attributes: tuple[str, ...] = ()
    last_asked_attribute: str | None = None
    intent_version: int = 0
    last_turn: int = 0


@dataclass(frozen=True, slots=True)
class IntentReduction:
    """One intent update plus aggregate-safe multi-slot outcome counts."""

    state: IntentState
    status: IntentReductionStatus
    positive_atoms: int = 0
    exclusion_atoms: int = 0
    clear_atoms: int = 0
    replacement_atoms: int = 0
    residual_atoms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.state, IntentState):
            raise TypeError("state must be IntentState")
        if not isinstance(self.status, IntentReductionStatus):
            raise TypeError("status must be IntentReductionStatus")
        counts = (
            self.positive_atoms,
            self.exclusion_atoms,
            self.clear_atoms,
            self.replacement_atoms,
            self.residual_atoms,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("intent reduction counts must be non-negative integers")
        if self.status is not IntentReductionStatus.APPLIED and any(counts):
            raise ValueError("only an applied reduction may report atom counts")


_SPACE_RE = re.compile(r"\s+")
_BUYING_MARKER = ". A key requirement is:"
_EXPLORING_MARKER = ", but I'm still exploring"
_INITIAL_PREFIX = "I'm looking for "
_ANSWER_RE = re.compile(
    r"^\s*For that, what matters is:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_OVERRIDE_RE = re.compile(
    r"^\s*Actually,\s*ignore my earlier preference\.\s*"
    r"What I need is:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_NO_PREFERENCE_RE = re.compile(
    r"^\s*I don't have (?:an additional|a) preference for "
    r"(?P<attribute>[a-z_ ]+?)"
    r"(?:;\s*please use your judgment)?\.\s*$",
    re.IGNORECASE,
)
_NOT_RIGHT_RE = re.compile(
    r"^\s*Those options are not quite right yet\.\s*"
    r"Ask me about one specific attribute\.\s*$",
    re.IGNORECASE,
)

_ROBUST_BUYING_RES = (
    re.compile(
        r"^(?:i['\u2019]?m\s+)?(?:looking|searching|shopping)\s+for\s+"
        r"(?P<category>.+?)\.\s*(?:(?:my|a)\s+)?(?:main|key)\s+requirement\s+"
        r"(?:is\s*)?[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^for\s+(?P<category>.+?),\s*the\s+(?:key|main)\s+requirement\s+"
        r"is\s*[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?help me find\s+(?P<category>.+?)\.\s*"
        r"it\s+must\s+(?:have|be)\s+(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i\s+need\s+(?P<category>.+?)[;,]\s*"
        r"(?:it\s+)?must\s+(?:have|be)\s+(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i\s+need\s+(?P<category>.+?)[;,]\s*(?:with\s+)?(?:this\s+)?"
        r"(?:main\s+|key\s+)?requirement\s*[:\u2014-]\s*"
        r"(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
)
_ROBUST_BROWSING_RES = (
    re.compile(
        r"^(?:i['\u2019]?m\s+)?(?:looking|browsing|shopping)\s+for\s+"
        r"(?P<category>.+?)(?:,\s*(?:but\s+)?|\s+and\s+)"
        r"(?:i['\u2019]?m\s+)?still\s+(?:exploring|deciding)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i['\u2019]?m\s+considering\s+(?P<category>.+?),\s*but\s+i\s+"
        r"have\s+not\s+narrowed\s+it\s+down\s+yet\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?show me\s+(?:some\s+)?(?P<category>.+?)[;,]\s*"
        r"(?:i['\u2019]?m\s+open\s+to\s+options|"
        r"i['\u2019]?m\s+keeping\s+my\s+options\s+open)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i\s+want\s+to\s+explore\s+(?P<category>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i['\u2019]?d\s+like\s+to\s+explore\s+(?P<category>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i['\u2019]?m\s+browsing\s+(?:for\s+)?(?P<category>.+?),?\s*"
        r"but\s+i\s+have(?:n['\u2019]?t|\s+not)\s+narrowed\s+"
        r"(?:things|it)\s+down\s+yet\.?$",
        re.IGNORECASE,
    ),
)
_ROBUST_TENTATIVE_RES = (
    re.compile(
        r"^i['\u2019]?m\s+considering\s+(?P<category>.+?)\.\s*"
        r"one\s+tentative\s+preference\s+is\s*[:\u2014-]?\s*"
        r"(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^for\s+(?P<category>.+?),\s*my\s+preference\s+for\s+now\s+is\s*"
        r"[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:i['\u2019]?m\s+)?looking\s+for\s+(?P<category>.+?),\s*"
        r"maybe\s+(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i['\u2019]?m\s+looking\s+for\s+(?P<category>.+?)\.\s*"
        r"(?P<value>.+?)$",
        re.IGNORECASE,
    ),
)
_ROBUST_CONTEXTUAL_ANSWER_RES = (
    re.compile(
        r"^for\s+that,\s*(?:the\s+important\s+detail|what\s+matters)\s+"
        r"is\s*[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^what\s+matters\s+to\s+me\s+there\s+is\s*[:\u2014-]?\s*"
        r"(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:my\s+preference\s+(?:there\s+)?is|please\s+prioritize)\s*"
        r"[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^for\s+that\s+attribute,?\s*i\s+(?:care\s+about|prefer|want)\s*"
        r"[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
)
_ROBUST_BARE_ANSWER_RES = (
    re.compile(
        r"^(?:i(?:['\u2019]?d|\s+would)?\s+)?(?:prefer|would\s+like)\s+"
        r"(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:make\s+it|go\s+with)\s+(?P<value>.+?)\.?$|"
        r"^(?P<polite_value>.+?),\s*please\.?$",
        re.IGNORECASE,
    ),
)
_ROBUST_OVERRIDE_RES = (
    re.compile(
        r"^actually,?\s*ignore\s+my\s+earlier\s+preference\.\s*"
        r"what\s+i\s+need\s+is\s*[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^actually,?\s*disregard\s+my\s+earlier\s+preference\.\s*"
        r"i\s+now\s+need\s*[:\u2014-]?\s*(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^change\s+of\s+plan\s*:\s*replace\s+my\s+earlier\s+preference\s+"
        r"with\s+(?P<value>.+?)\.?$",
        re.IGNORECASE,
    ),
)
_ROBUST_SCRATCH_OVERRIDE_RE = re.compile(
    r"^(?:actually,?\s*)?scratch\s+that\.\s*"
    r"(?:what\s+i\s+really\s+need\s+is\s*)?[:\u2014-]?\s*"
    r"(?P<value>.+?)\.?$",
    re.IGNORECASE,
)
_ROBUST_REPLACE_OVERRIDE_RE = re.compile(
    r"^i['\u2019]?ve\s+changed\s+my\s+mind\s*[:\u2014-]?\s*"
    r"(?:replace\s+(?:that|my\s+(?:earlier|last)\s+preference)\s+with\s*"
    r"[:\u2014-]?\s*)?(?P<value>.+?)\.?$",
    re.IGNORECASE,
)
_ROBUST_NO_PREFERENCE_RES = (
    re.compile(
        r"^i\s+don['\u2019]?t\s+have\s+(?:an\s+additional|a)\s+preference\s+"
        r"for\s+(?P<attribute>[a-z_ ]+?)"
        r"(?:;\s*please\s+use\s+your\s+judgment)?\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i\s+have\s+no\s+(?:additional\s+)?preference\s+for\s+"
        r"(?P<attribute>[a-z_ ]+?)(?:;\s*use\s+your\s+judgment)?\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^for\s+(?P<attribute>[a-z_ ]+?),\s*i\s+do\s+not\s+have\s+"
        r"(?:a\s+preference|another\s+requirement)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:any\s+(?P<any_attribute>[a-z_ ]+?)\s+is\s+fine|"
        r"(?P<matter_attribute>[a-z_ ]+?)\s+does(?:n['\u2019]?t|\s+not)\s+"
        r"matter\s+to\s+me)\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^no\s+preference\s+(?:on|for|about)\s+"
        r"(?P<attribute>[a-z_ ]+?)(?:;\s*(?:you\s+decide|use\s+your\s+"
        r"judgment))?\.?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i\s+have\s+(?:nothing\s+else|no\s+other\s+requirement)\s+"
        r"(?:to\s+add\s+)?(?:for|on)\s+(?P<attribute>[a-z_ ]+?)\.?$",
        re.IGNORECASE,
    ),
)
_ROBUST_NO_PREFERENCE_LAST_RE = re.compile(
    r"^(?:i\s+(?:have\s+)?no\s+preference|anything\s+is\s+fine|"
    r"i\s+do\s+not\s+care|you\s+decide)(?:;?\s*(?:use\s+your\s+"
    r"judgment|please))?\.?$",
    re.IGNORECASE,
)
_ROBUST_NOT_RIGHT_RE = re.compile(
    r"^(?=.*\b(?:ask|question|follow[ -]?up|clarif\w*)\b)"
    r"(?=.*\b(?:not\s+right|not\s+quite|narrow\s+this\s+down|no\s+luck|"
    r"not\s+working|do(?:n['\u2019]?t|\s+not)\s+fit)\b).+$",
    re.IGNORECASE,
)

_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)
_ROBUST_EXTRA_COLOR_RE = re.compile(
    r"\b(navy|teal|tan|beige|ivory|cream|khaki|charcoal|cerulean|"
    r"magenta|cyan|turquoise|indigo|violet|maroon|burgundy|coral|"
    r"lavender|olive|mustard|gold|silver)\b",
    re.IGNORECASE,
)
_ROBUST_EXTRA_MATERIAL_RE = re.compile(
    r"\b(suede|canvas|linen|rubber|mesh|fleece|denim|velvet|cashmere|"
    r"acrylic|microfiber|synthetic|down)\b",
    re.IGNORECASE,
)
_ROBUST_BRAND_VALUE_RE = re.compile(
    r"[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*){0,3}\Z"
)
_SIZE_RE = re.compile(r"\b(size|sizing|width|wide|narrow)\b", re.IGNORECASE)
_STYLE_RE = re.compile(
    r"\b(department|style|fit|sleeve|neck)\b", re.IGNORECASE
)
_USE_CASE_RE = re.compile(
    r"\b(hiking|running|gym|winter|outdoor|work)\b", re.IGNORECASE
)
_BUDGET_RE = re.compile(r"(?:\bbudget\b|\bunder\s+\$?\d|\$\s*\d|<=\s*\d)", re.IGNORECASE)
_BRAND_RE = re.compile(r"(?:^|\b)brand\s*:", re.IGNORECASE)
_BARE_ANSWER_COMMAND_RE = re.compile(
    r"^(?:show|find|give|suggest|recommend|help|search|look)\b",
    re.IGNORECASE,
)

_MULTI_MAX_MESSAGE_CHARACTERS = 2048
_MULTI_MAX_ATOMS_PER_TURN = 8
_MULTI_MAX_ATOM_CHARACTERS = 256
_MULTI_MAX_ACTIVE_REQUIREMENTS = 24
_MULTI_MAX_EXCLUSIONS = 16
_MULTI_MAX_SOFT_BOUNDARIES = 7

_MULTI_HARD_SEPARATOR_RE = re.compile(r";|\n+")
_MULTI_SOFT_SEPARATOR_RE = re.compile(
    r"(?<!\d),(?!\d)|\b(?:and|but|plus|also|while)\b",
    re.IGNORECASE,
)
_MULTI_EXPLICIT_LABEL_RE = re.compile(
    r"^(?P<attribute>material|color|size|style|brand|budget|feature|"
    r"use[ _]case|other)\s*[:=]\s*(?P<value>.+?)$",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_POSITIVE_SCAFFOLD_RE = re.compile(
    r"^(?:(?:i\s+(?:also\s+)?(?:want|need|prefer|would\s+like)(?:\s+it)?)|"
    r"(?:(?:please\s+)?(?:make|keep)\s+it)|"
    r"(?:it\s+(?:should|must)\s+(?:be|have))|with)\s+",
    re.IGNORECASE,
)
_MULTI_REPLACE_RE = re.compile(
    r"^replace\s+(?P<old>.+?)\s+with\s+(?P<new>.+?)$",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_EXCLUSION_RE = re.compile(
    r"^(?:(?:i\s+(?:do\s+not|don['\u2019]?t)\s+(?:want|prefer))|"
    r"no\s+longer|not|no|without|avoid|exclude)\s+(?P<value>.+?)$",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_NEGATION_GUARD_RE = re.compile(
    r"^(?:not\s+only|not\s+sure|no\s+less\s+than|without\s+a\s+doubt)\b",
    re.IGNORECASE,
)
_MULTI_NOW_RE = re.compile(r"^now\s+(?P<value>.+?)$", re.IGNORECASE | re.DOTALL)
_MULTI_INSTEAD_RE = re.compile(
    r"^(?P<value>.+?)\s+instead$",
    re.IGNORECASE | re.DOTALL,
)


class _AtomOperation(str, Enum):
    ADD = "add"
    EXCLUDE = "exclude"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class _IntentAtom:
    operation: _AtomOperation
    attribute: str | None
    value: str
    replacement: bool = False


@dataclass(frozen=True, slots=True)
class _CandidateEnvelope:
    payload: str
    source: RequirementSource
    category: str | None = None
    mode: Literal["add", "full_override", "replace_last", "replace_slot"] = "add"


class _CandidateBoundsError(ValueError):
    pass


class _CandidateAmbiguityError(ValueError):
    pass


def _clean(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def _generated_value(value: str) -> str:
    """Remove the simulator's one terminal period, preserving internal punctuation."""

    cleaned = _clean(value)
    return cleaned[:-1].rstrip() if cleaned.endswith(".") else cleaned


def _normalize_attribute(value: str) -> str | None:
    attribute = _clean(value).lower().replace(" ", "_")
    return attribute if attribute in ALLOWED_ATTRIBUTES else None


def classify_requirement(value: str) -> str:
    """Conservatively map evaluator-shaped text to a query section."""

    if _BUDGET_RE.search(value):
        return "budget"
    if _MATERIAL_RE.search(value):
        return "material"
    if _COLOR_RE.search(value) or "color" in value.lower():
        return "color"
    if _SIZE_RE.search(value):
        return "size"
    if _STYLE_RE.search(value):
        return "style"
    if _USE_CASE_RE.search(value):
        return "use_case"
    if _BRAND_RE.search(value):
        return "brand"
    return "feature"


def _is_slot_like_bare_answer(value: str, asked_attribute: str) -> bool:
    if _BARE_ANSWER_COMMAND_RE.search(value):
        return False
    inferred = classify_requirement(value)
    return inferred != "feature" and inferred == asked_attribute


def _append_requirement(
    requirements: tuple[Requirement, ...], requirement: Requirement
) -> tuple[Requirement, ...]:
    if not requirement.value:
        return requirements
    key = requirement.value.casefold()
    without_duplicate = tuple(
        current for current in requirements if current.value.casefold() != key
    )
    return (*without_duplicate, requirement)


def _apply_strong_override(
    state: IntentState,
    raw_value: str,
    turn: int,
    *,
    replace_last: bool,
) -> IntentState:
    """Replace the most recent contradicted preference for explicit reset cues."""

    value = _generated_value(raw_value)
    requirements = list(state.requirements)
    attribute = classify_requirement(value) if value else None
    if attribute == "feature":
        if _ROBUST_EXTRA_COLOR_RE.search(value):
            attribute = "color"
        elif _ROBUST_EXTRA_MATERIAL_RE.search(value):
            attribute = "material"
        elif (
            requirements
            and requirements[-1].attribute == "brand"
            and _ROBUST_BRAND_VALUE_RE.fullmatch(value)
        ):
            attribute = "brand"
    replaced: Requirement | None = None
    if replace_last and requirements:
        replaced = requirements.pop()
    elif not replace_last:
        replacement_index: int | None = None
        if attribute not in {None, "feature"}:
            replacement_index = next(
                (
                    index
                    for index in range(len(requirements) - 1, -1, -1)
                    if requirements[index].attribute == attribute
                ),
                None,
            )
        if replacement_index is None and len(requirements) == 1:
            replacement_index = len(requirements) - 1
        if replacement_index is not None:
            replaced = requirements.pop(replacement_index)
    if (
        attribute == "feature"
        and replaced is not None
        and replaced.attribute not in {None, "feature", "use_case", "other"}
    ):
        attribute = replaced.attribute
    if attribute not in {None, "feature"}:
        requirements = [
            requirement
            for requirement in requirements
            if requirement.attribute != attribute
        ]
    if value:
        requirements = list(
            _append_requirement(
                tuple(requirements),
                Requirement(
                    value=value,
                    source="override",
                    turn=turn,
                    attribute=attribute,
                ),
            )
        )
    no_preference = state.no_preference
    if attribute is not None:
        no_preference = frozenset(
            item for item in no_preference if item != attribute
        )
    return replace(
        state,
        requirements=tuple(requirements),
        no_preference=no_preference,
        intent_version=state.intent_version + 1,
        last_turn=turn,
    )


def _parse_initial_message(message: str, turn: int) -> tuple[str | None, Requirement | None]:
    if not message.startswith(_INITIAL_PREFIX):
        return None, None

    body = message[len(_INITIAL_PREFIX) :]
    if _BUYING_MARKER in body:
        category, _, raw_value = body.partition(_BUYING_MARKER)
        value = _generated_value(raw_value)
        requirement = Requirement(
            value=value,
            source="initial_explicit",
            turn=turn,
            attribute=classify_requirement(value),
        ) if value else None
        return _clean(category).rstrip("."), requirement

    marker_position = body.casefold().find(_EXPLORING_MARKER.casefold())
    if marker_position >= 0:
        return _clean(body[:marker_position]).rstrip("."), None

    category, separator, raw_value = body.partition(". ")
    category = _clean(category).rstrip(".")
    if not separator:
        return category, None
    value = _generated_value(raw_value)
    requirement = Requirement(
        value=value,
        source="initial_tentative",
        turn=turn,
        attribute=classify_requirement(value),
    ) if value else None
    return category, requirement


def _first_match(
    patterns: tuple[re.Pattern[str], ...], message: str
) -> re.Match[str] | None:
    for pattern in patterns:
        match = pattern.fullmatch(message)
        if match is not None:
            return match
    return None


def _canonicalize_robust_message(
    state: IntentState,
    message: str,
    turn: int,
) -> str:
    """Map common natural scaffolds onto the frozen canonical reducer grammar."""

    if turn == 1:
        buying = _first_match(_ROBUST_BUYING_RES, message)
        if buying is not None:
            category = _clean(buying.group("category")).rstrip(".")
            value = _clean(buying.group("value"))
            return f"I'm looking for {category}. A key requirement is: {value}."

        browsing = _first_match(_ROBUST_BROWSING_RES, message)
        if browsing is not None:
            category = _clean(browsing.group("category")).rstrip(".")
            return f"I'm looking for {category}, but I'm still exploring"

        tentative = _first_match(_ROBUST_TENTATIVE_RES, message)
        if tentative is not None:
            category = _clean(tentative.group("category")).rstrip(".")
            value = _generated_value(tentative.group("value"))
            return f"I'm looking for {category}. {value}"

    override = _first_match(_ROBUST_OVERRIDE_RES, message)
    if override is not None:
        value = _clean(override.group("value"))
        return (
            "Actually, ignore my earlier preference. "
            f"What I need is: {value}."
        )

    no_preference = _first_match(_ROBUST_NO_PREFERENCE_RES, message)
    if no_preference is not None:
        groups = no_preference.groupdict()
        raw_attribute = (
            groups.get("attribute")
            or groups.get("any_attribute")
            or groups.get("matter_attribute")
            or ""
        )
        attribute = _normalize_attribute(raw_attribute)
        if attribute is not None:
            return f"I don't have an additional preference for {attribute}."

    if (
        state.last_asked_attribute is not None
        and _ROBUST_NO_PREFERENCE_LAST_RE.fullmatch(message)
    ):
        return (
            "I don't have an additional preference for "
            f"{state.last_asked_attribute}."
        )

    if _ROBUST_NOT_RIGHT_RE.fullmatch(message):
        return "Those options are not quite right yet. Ask me about one specific attribute."

    answer = _first_match(_ROBUST_CONTEXTUAL_ANSWER_RES, message)
    if answer is not None and state.last_asked_attribute is not None:
        groups = answer.groupdict()
        value = _clean(groups.get("value") or groups.get("polite_value") or "")
        return f"For that, what matters is: {value}."

    bare_answer = _first_match(_ROBUST_BARE_ANSWER_RES, message)
    if bare_answer is not None and state.last_asked_attribute is not None:
        groups = bare_answer.groupdict()
        value = _clean(groups.get("value") or groups.get("polite_value") or "")
        if _is_slot_like_bare_answer(value, state.last_asked_attribute):
            return f"For that, what matters is: {value}."

    return message


def _candidate_attribute(value: str) -> tuple[str | None, bool]:
    """Return a singleton high-confidence attribute and ambiguity flag."""

    matches: set[str] = set()
    if _BUDGET_RE.search(value):
        matches.add("budget")
    if _MATERIAL_RE.search(value) or _ROBUST_EXTRA_MATERIAL_RE.search(value):
        matches.add("material")
    if (
        _COLOR_RE.search(value)
        or _ROBUST_EXTRA_COLOR_RE.search(value)
        or "color" in value.casefold()
    ):
        matches.add("color")
    if _SIZE_RE.search(value):
        matches.add("size")
    if _STYLE_RE.search(value):
        matches.add("style")
    if _USE_CASE_RE.search(value):
        matches.add("use_case")
    if _BRAND_RE.search(value):
        matches.add("brand")
    if len(matches) > 1:
        return None, True
    return (next(iter(matches)) if matches else None), False


def _candidate_value_and_attribute(
    raw_value: str,
) -> tuple[str, str | None]:
    value = _generated_value(raw_value)
    label = _MULTI_EXPLICIT_LABEL_RE.fullmatch(value)
    if label is not None:
        attribute = _normalize_attribute(label.group("attribute"))
        if attribute is None or attribute == "category":
            raise _CandidateAmbiguityError("invalid explicit attribute")
        value = _generated_value(label.group("value"))
        if not value:
            raise _CandidateAmbiguityError("empty explicit value")
        return value, attribute
    attribute, ambiguous = _candidate_attribute(value)
    if ambiguous:
        raise _CandidateAmbiguityError("multiple attributes in one span")
    return value, attribute


def _candidate_no_preference_attribute(
    state: IntentState,
    value: str,
) -> str | None:
    match = _first_match(_ROBUST_NO_PREFERENCE_RES, value)
    if match is not None:
        groups = match.groupdict()
        raw_attribute = (
            groups.get("attribute")
            or groups.get("any_attribute")
            or groups.get("matter_attribute")
            or ""
        )
        return _normalize_attribute(raw_attribute)
    if (
        state.last_asked_attribute is not None
        and _ROBUST_NO_PREFERENCE_LAST_RE.fullmatch(value)
    ):
        return state.last_asked_attribute
    return None


def _bounded_candidate_value(raw_value: str) -> str:
    value = _generated_value(raw_value)
    if not value or re.search(r"\w", value) is None:
        raise _CandidateAmbiguityError("candidate value has no meaningful text")
    if len(value) > _MULTI_MAX_ATOM_CHARACTERS:
        raise _CandidateBoundsError("candidate value is too long")
    return value


def _parse_candidate_segment(
    state: IntentState,
    raw_segment: str,
) -> tuple[_IntentAtom, ...]:
    segment = _bounded_candidate_value(raw_segment)
    no_preference = _candidate_no_preference_attribute(state, segment)
    if no_preference is not None:
        return (
            _IntentAtom(
                operation=_AtomOperation.CLEAR,
                attribute=no_preference,
                value="",
            ),
        )

    if _MULTI_NEGATION_GUARD_RE.match(segment):
        raise _CandidateAmbiguityError("unsafe negation scope")

    replacement = _MULTI_REPLACE_RE.fullmatch(segment)
    if replacement is not None:
        old_value, old_attribute = _candidate_value_and_attribute(
            _bounded_candidate_value(replacement.group("old"))
        )
        new_value, new_attribute = _candidate_value_and_attribute(
            _bounded_candidate_value(replacement.group("new"))
        )
        if (
            old_attribute is None
            or new_attribute is None
            or old_attribute != new_attribute
        ):
            raise _CandidateAmbiguityError("replacement slot is ambiguous")
        return (
            _IntentAtom(
                operation=_AtomOperation.EXCLUDE,
                attribute=old_attribute,
                value=old_value,
            ),
            _IntentAtom(
                operation=_AtomOperation.ADD,
                attribute=new_attribute,
                value=new_value,
                replacement=True,
            ),
        )

    exclusion = _MULTI_EXCLUSION_RE.fullmatch(segment)
    if exclusion is not None:
        value, attribute = _candidate_value_and_attribute(
            _bounded_candidate_value(exclusion.group("value"))
        )
        return (
            _IntentAtom(
                operation=_AtomOperation.EXCLUDE,
                attribute=attribute,
                value=value,
            ),
        )

    replace_value = _MULTI_NOW_RE.fullmatch(segment)
    if replace_value is None:
        replace_value = _MULTI_INSTEAD_RE.fullmatch(segment)
    is_replacement = replace_value is not None
    if replace_value is not None:
        segment = _bounded_candidate_value(replace_value.group("value"))

    segment = _MULTI_POSITIVE_SCAFFOLD_RE.sub("", segment, count=1)
    value, attribute = _candidate_value_and_attribute(
        _bounded_candidate_value(segment)
    )
    if is_replacement and attribute is None:
        raise _CandidateAmbiguityError("untyped replacement")
    return (
        _IntentAtom(
            operation=_AtomOperation.ADD,
            attribute=attribute,
            value=value,
            replacement=is_replacement,
        ),
    )


def _candidate_segments(
    payload: str,
    soft_mask: int,
    hard_matches: tuple[re.Match[str], ...],
    soft_matches: tuple[re.Match[str], ...],
) -> tuple[str, ...]:
    selected = [*hard_matches]
    selected.extend(
        match
        for index, match in enumerate(soft_matches)
        if soft_mask & (1 << index)
    )
    selected.sort(key=lambda match: (match.start(), match.end()))
    segments: list[str] = []
    cursor = 0
    for match in selected:
        if match.start() < cursor:
            raise _CandidateAmbiguityError("overlapping separators")
        segment = _clean(payload[cursor : match.start()])
        if not segment:
            raise _CandidateAmbiguityError("empty candidate segment")
        segments.append(segment)
        cursor = match.end()
    segment = _clean(payload[cursor:])
    if not segment:
        raise _CandidateAmbiguityError("empty candidate segment")
    segments.append(segment)
    return tuple(segments)


def _candidate_atoms_are_multislot(atoms: tuple[_IntentAtom, ...]) -> bool:
    if not 2 <= len(atoms) <= _MULTI_MAX_ATOMS_PER_TURN:
        return False
    if any(atom.operation is not _AtomOperation.ADD for atom in atoms):
        return True
    semantic_slots = {atom.attribute for atom in atoms}
    return len(semantic_slots) >= 2


def _parse_candidate_atoms(
    state: IntentState,
    payload: str,
) -> tuple[tuple[_IntentAtom, ...] | None, IntentReductionStatus]:
    hard_matches = tuple(_MULTI_HARD_SEPARATOR_RE.finditer(payload))
    soft_matches = tuple(
        match
        for match in _MULTI_SOFT_SEPARATOR_RE.finditer(payload)
        if all(
            match.end() <= hard.start() or match.start() >= hard.end()
            for hard in hard_matches
        )
    )
    if len(soft_matches) > _MULTI_MAX_SOFT_BOUNDARIES:
        return None, IntentReductionStatus.BOUNDS
    if len(hard_matches) + 1 > _MULTI_MAX_ATOMS_PER_TURN:
        return None, IntentReductionStatus.BOUNDS

    masks = sorted(
        range(1 << len(soft_matches)),
        key=lambda value: (value.bit_count(), value),
    )
    saw_bounds = False
    saw_parseable = False
    for mask in masks:
        try:
            segments = _candidate_segments(
                payload,
                mask,
                hard_matches,
                soft_matches,
            )
            atoms = tuple(
                atom
                for segment in segments
                for atom in _parse_candidate_segment(state, segment)
            )
        except _CandidateBoundsError:
            saw_bounds = True
            continue
        except _CandidateAmbiguityError:
            continue
        saw_parseable = True
        if len(atoms) > _MULTI_MAX_ATOMS_PER_TURN:
            saw_bounds = True
            continue
        if _candidate_atoms_are_multislot(atoms):
            return atoms, IntentReductionStatus.APPLIED
    if saw_bounds:
        return None, IntentReductionStatus.BOUNDS
    if saw_parseable:
        return None, IntentReductionStatus.SINGLE_SLOT
    return None, IntentReductionStatus.AMBIGUOUS


def _candidate_envelope(
    state: IntentState,
    message: str,
    turn: int,
) -> _CandidateEnvelope | None:
    cleaned = _clean(message)
    scratch = _ROBUST_SCRATCH_OVERRIDE_RE.fullmatch(cleaned)
    if scratch is not None:
        return _CandidateEnvelope(
            payload=_generated_value(scratch.group("value")),
            source="override",
            mode="replace_last",
        )
    replacement = _ROBUST_REPLACE_OVERRIDE_RE.fullmatch(cleaned)
    if replacement is not None:
        return _CandidateEnvelope(
            payload=_generated_value(replacement.group("value")),
            source="override",
            mode="replace_slot",
        )

    canonical = _canonicalize_robust_message(state, cleaned, turn)
    if turn == 1:
        category, requirement = _parse_initial_message(canonical, turn)
        if category is not None:
            if requirement is None:
                return None
            return _CandidateEnvelope(
                payload=requirement.value,
                source=requirement.source,
                category=category,
            )

    override = _OVERRIDE_RE.fullmatch(canonical)
    if override is not None:
        return _CandidateEnvelope(
            payload=_generated_value(override.group("value")),
            source="override",
            mode="full_override",
        )
    answer = _ANSWER_RE.fullmatch(canonical)
    if answer is not None:
        return _CandidateEnvelope(
            payload=_generated_value(answer.group("value")),
            source="answer",
        )
    if (
        _NO_PREFERENCE_RE.fullmatch(canonical)
        or _NOT_RIGHT_RE.fullmatch(canonical)
        or not canonical
    ):
        return None
    return _CandidateEnvelope(payload=canonical, source="free_text")


def _same_value(left: str, right: str) -> bool:
    def normalized(value: str) -> str:
        cleaned = _generated_value(value)
        label = _MULTI_EXPLICIT_LABEL_RE.fullmatch(cleaned)
        if label is not None:
            cleaned = _generated_value(label.group("value"))
        return cleaned.casefold()

    return normalized(left) == normalized(right)


def _exclusion_matches_attribute(value: str, attribute: str) -> bool:
    inferred, ambiguous = _candidate_attribute(value)
    return not ambiguous and inferred == attribute


def _apply_candidate_atoms(
    state: IntentState,
    envelope: _CandidateEnvelope,
    atoms: tuple[_IntentAtom, ...],
    turn: int,
) -> IntentState:
    if (
        len(state.requirements) > _MULTI_MAX_ACTIVE_REQUIREMENTS
        or len(state.excluded) > _MULTI_MAX_EXCLUSIONS
    ):
        raise _CandidateBoundsError("existing state exceeds candidate bounds")

    requirements = list(state.requirements)
    exclusions = list(state.excluded)
    no_preference = set(state.no_preference)
    destructive = envelope.mode != "add"
    if envelope.mode == "full_override":
        requirements = [
            requirement
            for requirement in requirements
            if requirement.source
            not in {"initial_explicit", "initial_tentative", "override"}
        ]
    elif envelope.mode == "replace_last":
        if requirements:
            requirements.pop()

    for atom in atoms:
        attribute = atom.attribute
        if atom.operation is _AtomOperation.ADD:
            replacement = atom.replacement or envelope.mode == "replace_slot"
            if replacement:
                if attribute is None:
                    raise _CandidateAmbiguityError("replacement is untyped")
                requirements = [
                    requirement
                    for requirement in requirements
                    if requirement.attribute != attribute
                ]
                destructive = True
            exclusions = [
                value for value in exclusions if not _same_value(value, atom.value)
            ]
            if attribute is not None:
                no_preference.discard(attribute)
            requirements = [
                requirement
                for requirement in requirements
                if not _same_value(requirement.value, atom.value)
            ]
            requirements.append(
                Requirement(
                    value=atom.value,
                    source=envelope.source,
                    turn=turn,
                    attribute=attribute,
                )
            )
        elif atom.operation is _AtomOperation.EXCLUDE:
            requirements = [
                requirement
                for requirement in requirements
                if not _same_value(requirement.value, atom.value)
            ]
            exclusions = [
                value for value in exclusions if not _same_value(value, atom.value)
            ]
            exclusions.append(atom.value)
            destructive = True
        elif atom.operation is _AtomOperation.CLEAR:
            if attribute is None:
                raise _CandidateAmbiguityError("clear operation is untyped")
            requirements = [
                requirement
                for requirement in requirements
                if requirement.attribute != attribute
            ]
            exclusions = [
                value
                for value in exclusions
                if not _exclusion_matches_attribute(value, attribute)
            ]
            no_preference.add(attribute)
            destructive = True
        else:
            raise _CandidateAmbiguityError("unsupported candidate operation")

    if (
        len(requirements) > _MULTI_MAX_ACTIVE_REQUIREMENTS
        or len(exclusions) > _MULTI_MAX_EXCLUSIONS
    ):
        raise _CandidateBoundsError("candidate result exceeds state bounds")
    if any(
        not requirement.value
        or len(requirement.value) > _MULTI_MAX_ATOM_CHARACTERS
        or (
            requirement.attribute is not None
            and requirement.attribute not in ALLOWED_ATTRIBUTES
        )
        for requirement in requirements
    ):
        raise _CandidateAmbiguityError("candidate requirement is invalid")
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > _MULTI_MAX_ATOM_CHARACTERS
        for value in exclusions
    ):
        raise _CandidateAmbiguityError("candidate exclusion is invalid")
    if len({_clean(value).casefold() for value in exclusions}) != len(exclusions):
        raise _CandidateAmbiguityError("candidate exclusions are duplicated")
    if any(
        _same_value(requirement.value, exclusion)
        for requirement in requirements
        for exclusion in exclusions
    ):
        raise _CandidateAmbiguityError("positive and excluded values conflict")

    return replace(
        state,
        category=envelope.category or state.category,
        requirements=tuple(requirements),
        excluded=tuple(exclusions),
        no_preference=frozenset(no_preference),
        intent_version=state.intent_version + int(destructive),
        last_turn=turn,
    )


def apply_user_message_with_trace(
    state: IntentState,
    message: str,
    turn: int,
    *,
    policy: IntentParsingPolicy = ROBUST_INTENT_POLICY,
) -> IntentReduction:
    """Reduce one message and expose only fixed aggregate-safe parse facts."""

    if policy is not LOSSLESS_MULTI_SLOT_INTENT_POLICY:
        return IntentReduction(
            state=apply_user_message(state, message, turn, policy=policy),
            status=IntentReductionStatus.BASELINE_POLICY,
        )

    baseline = apply_user_message(
        state,
        message,
        turn,
        policy=ROBUST_INTENT_POLICY,
    )
    try:
        cleaned = _clean(message)
        if len(cleaned) > _MULTI_MAX_MESSAGE_CHARACTERS:
            return IntentReduction(baseline, IntentReductionStatus.BOUNDS)
        envelope = _candidate_envelope(state, cleaned, turn)
        if envelope is None or not envelope.payload:
            return IntentReduction(baseline, IntentReductionStatus.SINGLE_SLOT)
        atoms, status = _parse_candidate_atoms(state, envelope.payload)
        if atoms is None:
            return IntentReduction(baseline, status)
        candidate = _apply_candidate_atoms(state, envelope, atoms, turn)
        if candidate == baseline:
            return IntentReduction(baseline, IntentReductionStatus.SINGLE_SLOT)
        return IntentReduction(
            state=candidate,
            status=IntentReductionStatus.APPLIED,
            positive_atoms=sum(
                atom.operation is _AtomOperation.ADD for atom in atoms
            ),
            exclusion_atoms=sum(
                atom.operation is _AtomOperation.EXCLUDE for atom in atoms
            ),
            clear_atoms=sum(
                atom.operation is _AtomOperation.CLEAR for atom in atoms
            ),
            replacement_atoms=sum(atom.replacement for atom in atoms),
            residual_atoms=sum(
                atom.operation is _AtomOperation.ADD and atom.attribute is None
                for atom in atoms
            ),
        )
    except (_CandidateBoundsError,):
        return IntentReduction(baseline, IntentReductionStatus.BOUNDS)
    except _CandidateAmbiguityError:
        return IntentReduction(baseline, IntentReductionStatus.AMBIGUOUS)
    except Exception:
        return IntentReduction(
            baseline,
            IntentReductionStatus.VALIDATION_FALLBACK,
        )


def apply_user_message(
    state: IntentState,
    message: str,
    turn: int,
    *,
    policy: IntentParsingPolicy = ROBUST_INTENT_POLICY,
) -> IntentState:
    """Reduce one latest-message delta into a new active intent state."""

    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= 10:
        raise ValueError("turn must be an integer from 1 through 10")
    if turn <= state.last_turn:
        raise ValueError("turns must be strictly increasing within a session")
    if not isinstance(policy, IntentParsingPolicy):
        raise TypeError("policy must be an IntentParsingPolicy")
    if policy is LOSSLESS_MULTI_SLOT_INTENT_POLICY:
        return apply_user_message_with_trace(
            state,
            message,
            turn,
            policy=policy,
        ).state

    cleaned_message = _clean(message)
    if policy is ROBUST_INTENT_POLICY:
        scratch_override = _ROBUST_SCRATCH_OVERRIDE_RE.fullmatch(cleaned_message)
        if scratch_override is not None:
            return _apply_strong_override(
                state,
                scratch_override.group("value"),
                turn,
                replace_last=True,
            )
        replace_override = _ROBUST_REPLACE_OVERRIDE_RE.fullmatch(cleaned_message)
        if replace_override is not None:
            return _apply_strong_override(
                state,
                replace_override.group("value"),
                turn,
                replace_last=False,
            )
        cleaned_message = _canonicalize_robust_message(
            state,
            cleaned_message,
            turn,
        )
    next_state = replace(state, last_turn=turn)

    if turn == 1:
        category, requirement = _parse_initial_message(cleaned_message, turn)
        if category:
            next_state = replace(next_state, category=category)
        if requirement is not None:
            next_state = replace(
                next_state,
                requirements=_append_requirement(next_state.requirements, requirement),
            )
        if category is not None:
            return next_state

    override_match = _OVERRIDE_RE.fullmatch(cleaned_message)
    if override_match:
        value = _generated_value(override_match.group("value"))
        retained = tuple(
            requirement
            for requirement in next_state.requirements
            if requirement.source
            not in {"initial_explicit", "initial_tentative", "override"}
        )
        if value:
            retained = _append_requirement(
                retained,
                Requirement(
                    value=value,
                    source="override",
                    turn=turn,
                    attribute=classify_requirement(value),
                ),
            )
        replacement_attribute = classify_requirement(value) if value else None
        no_preference = next_state.no_preference
        if replacement_attribute is not None:
            no_preference = frozenset(
                item for item in no_preference if item != replacement_attribute
            )
        return replace(
            next_state,
            requirements=retained,
            no_preference=no_preference,
            intent_version=next_state.intent_version + 1,
        )

    answer_match = _ANSWER_RE.fullmatch(cleaned_message)
    if answer_match:
        value = _generated_value(answer_match.group("value"))
        attribute = next_state.last_asked_attribute
        requirements = next_state.requirements
        if value:
            requirements = _append_requirement(
                requirements,
                Requirement(
                    value=value,
                    source="answer",
                    turn=turn,
                    attribute=attribute,
                ),
            )
        no_preference = next_state.no_preference
        if attribute is not None:
            no_preference = frozenset(item for item in no_preference if item != attribute)
        return replace(
            next_state,
            requirements=requirements,
            no_preference=no_preference,
        )

    no_preference_match = _NO_PREFERENCE_RE.fullmatch(cleaned_message)
    if no_preference_match:
        attribute = _normalize_attribute(no_preference_match.group("attribute"))
        if attribute is None:
            return next_state
        requirements = tuple(
            requirement
            for requirement in next_state.requirements
            if requirement.attribute != attribute
        )
        return replace(
            next_state,
            requirements=requirements,
            no_preference=next_state.no_preference | {attribute},
        )

    if not cleaned_message or _NOT_RIGHT_RE.fullmatch(cleaned_message):
        return next_state

    return replace(
        next_state,
        requirements=_append_requirement(
            next_state.requirements,
            Requirement(
                value=cleaned_message,
                source="free_text",
                turn=turn,
                attribute=None,
            ),
        ),
    )


def record_question(state: IntentState, attribute: str) -> IntentState:
    normalized = _normalize_attribute(attribute)
    if normalized is None:
        raise ValueError(f"unsupported clarification attribute: {attribute!r}")
    asked = state.asked_attributes
    if normalized not in asked:
        asked = (*asked, normalized)
    return replace(
        state,
        asked_attributes=asked,
        last_asked_attribute=normalized,
    )


def active_attributes(state: IntentState) -> frozenset[str]:
    return frozenset(
        requirement.attribute
        for requirement in state.requirements
        if requirement.attribute is not None
    )


def _deduplicate(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _without_label(value: str, label: str) -> str:
    return re.sub(
        rf"^\s*{re.escape(label)}\s*:\s*",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )


def render_dense_query(state: IntentState) -> str:
    """Render `query-text-v1` from active state in its frozen section order."""

    lines: list[str] = []
    if state.category:
        lines.append(f"Category: {state.category}")

    search_clues = _deduplicate(
        [
            requirement.value
            for requirement in state.requirements
            if requirement.attribute in {None, "feature", "use_case", "other"}
        ]
    )
    if search_clues:
        lines.append("Search Clues: " + " | ".join(search_clues))

    brands = _deduplicate(
        [
            _without_label(requirement.value, "brand")
            for requirement in state.requirements
            if requirement.attribute == "brand"
        ]
    )
    if brands:
        lines.append("Brand: " + " | ".join(brands))

    labels = {
        "material": "Material",
        "color": "Color",
        "size": "Size",
        "style": "Style",
    }
    attributes: list[str] = []
    for key, label in labels.items():
        for requirement in state.requirements:
            if requirement.attribute == key:
                attributes.append(
                    f"{label}: {_without_label(requirement.value, key)}"
                )
    if attributes:
        lines.append("Attributes: " + " | ".join(_deduplicate(attributes)))

    budgets = _deduplicate(
        [
            requirement.value
            for requirement in state.requirements
            if requirement.attribute == "budget"
        ]
    )
    if budgets:
        lines.append("Price: " + " | ".join(budgets))
    return "\n".join(lines)


def render_lexical_query(state: IntentState) -> str:
    """Render label-free positive terms for SQLite FTS."""

    values = [state.category or "", *(item.value for item in state.requirements)]
    return " ".join(_deduplicate([value for value in values if value]))


def render_requirement_probe_candidates(state: IntentState) -> tuple[str, ...]:
    """Return bounded strong positive clauses without catalog-dependent ranking.

    Catalog document frequency and the final two-probe selection belong to the
    retriever.  This renderer only preserves active requirement boundaries and
    provenance that would be lost in the blended lexical query.
    """

    if not isinstance(state, IntentState):
        raise TypeError("state must be IntentState")
    if len(state.requirements) > _MULTI_MAX_ACTIVE_REQUIREMENTS:
        return ()

    values: list[str] = []
    seen: set[str] = set()
    total_characters = 0
    for requirement in state.requirements:
        if requirement.strength != "hard" or requirement.attribute == "budget":
            continue
        value = requirement.value
        if requirement.attribute:
            value = _without_label(value, requirement.attribute)
        value = _clean(value)
        if not value or len(value) > _MULTI_MAX_ATOM_CHARACTERS:
            continue
        total_characters += len(value)
        if total_characters > 1024:
            return ()
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return tuple(values)
