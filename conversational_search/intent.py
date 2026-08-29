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


class IntentParsingPolicy(str, Enum):
    """Reversible intent reducers used by the Phase 6 robustness ablation."""

    CANONICAL = "canonical"
    ROBUST = "robust"


CANONICAL_INTENT_POLICY = IntentParsingPolicy.CANONICAL
ROBUST_INTENT_POLICY = IntentParsingPolicy.ROBUST


@dataclass(frozen=True, slots=True)
class Requirement:
    """One active positive requirement with enough provenance to supersede safely."""

    value: str
    source: RequirementSource
    turn: int
    attribute: str | None = None


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
            if requirement.source not in {"initial_explicit", "initial_tentative"}
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
