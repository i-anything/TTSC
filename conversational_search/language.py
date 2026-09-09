"""Bounded, lossless normalization of explicit shopping dialogue envelopes.

Only the surrounding speech act is normalized. Product values, their order,
and internal punctuation remain evidence for the existing transcript replay.
Matching a phrase never establishes catalog support on its own. Unrecognized
or ambiguous prose is returned intact for ordinary hybrid retrieval.
"""

from __future__ import annotations

import re

from conversational_search.intent import ALLOWED_ATTRIBUTES


MAX_MESSAGE_CHARACTERS = 2048
_SPACE = re.compile(r"\s+")
_QUERY = re.compile(
    r"^(?:(?:i(?:['’]m| am) )?(?:looking|shopping|searching|browsing) for|"
    r"(?:please )?(?:help me find|find me|show me)|"
    r"i (?:need|want)(?: to (?:find|buy))?|"
    r"i(?:['’]d| would) like(?: to (?:find|buy))?) (?P<body>.+)$",
    re.IGNORECASE,
)
_BUYING = re.compile(
    r"^(?P<category>.+?)[.;] "
    r"(?:(?:(?:a|my|the) )?(?:(?:key|main|essential) )?requirement(?: is)?\s*:|"
    r"it (?:must|needs to) (?:satisfy|meet) (?:this|the) requirement\s*:|"
    r"it (?:must|needs to) (?:have|be)\s*:?|(?:must-have|essential)\s*:) "
    r"(?P<value>.+)$",
    re.IGNORECASE,
)
_BROWSING = re.compile(
    r"^(?P<category>.+?)[,;.] (?:but )?(?:i(?:['’]m| am) )?"
    r"(?:still |just )?(?:exploring|browsing|deciding)"
    r"(?: for now| my options)?[.!]?$",
    re.IGNORECASE,
)
_TENTATIVE = re.compile(
    r"^(?P<category>.+?)[.;] "
    r"(?:one tentative preference is|my preference for now is|maybe|ideally)"
    r"\s*:? (?P<value>.+)$",
    re.IGNORECASE,
)
_PLAIN_INITIAL = re.compile(r"^(?P<category>.+?)\. (?P<value>.+)$")
_ANSWER = re.compile(
    r"^(?:(?:for that(?: attribute)?[,;] )?"
    r"(?:what matters(?: to me)?|the important detail|my preference) is|"
    r"my (?:priorities|requirements) are|"
    r"for that(?: attribute)?[,;] i (?:prefer|want|care about))"
    r"\s*:? (?P<value>.+)$",
    re.IGNORECASE,
)
_OVERRIDE = re.compile(
    r"^(?:(?:actually[,;] )?(?:ignore|disregard) my (?:earlier|previous) preference|"
    r"i(?: have|['’]ve)? changed my mind|change of plan)[.;:] "
    r"(?:what i (?:need|want) is|i now (?:need|want)|please prioritize|"
    r"replace my (?:earlier|previous) preference with)\s*:? (?P<value>.+)$",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"^(?P<kind>"
    r"i (?:do not|don['’]t) have (?:a|an additional) preference|"
    r"no (?:particular |additional )?preference|"
    r"nothing (?:else|more) to add|i have nothing (?:else|more) to add) "
    r"(?:for|on|about|regarding) (?P<attribute>[a-z_]+)"
    r"(?:; (?:please use your judg(?:e)?ment|you can decide|you decide))?\.$",
    re.IGNORECASE,
)
_HARD_CUE = re.compile(r"\b(?:must|requirement|required|need|essential)\b", re.I)


def _payload(value: str, limit: int, *, sentence: bool = True) -> str | None:
    # Remove at most the envelope's final full stop, never internal punctuation.
    payload = value[:-1] if sentence and value.endswith(".") else value
    return payload if payload and len(payload) <= limit else None


def normalize_dialogue_envelope(
    message: str,
    turn: int,
    asked_attribute: str | None,
) -> str:
    """Share one interpretation between intent reduction and catalog replay.

    Answers and negative replies require a matching question context. In
    particular, declining an attribute is distinct from exhausting additional
    preferences for it. No values are inferred, reordered, split, or corrected.
    """

    if len(message) > MAX_MESSAGE_CHARACTERS:
        return message
    cleaned = _SPACE.sub(" ", message).strip()
    # Preserve every already recognized official envelope, including tentative
    # values whose final punctuation belongs to the value rather than a wrapper.
    from conversational_search.decision import (
        ProtocolObservation,
        recognize_protocol_observation,
    )

    if recognize_protocol_observation(cleaned, turn) is not ProtocolObservation.UNSUPPORTED:
        return message

    if turn == 1:
        query = _QUERY.fullmatch(cleaned)
        if query is None:
            return message
        body = query.group("body")
        for pattern, kind in ((_BUYING, "buying"), (_BROWSING, "browsing"),
                              (_TENTATIVE, "tentative"), (_PLAIN_INITIAL, "plain")):
            match = pattern.fullmatch(body)
            if match is None:
                continue
            category = match.group("category")
            if len(category) > 256:
                return message
            if kind == "browsing":
                return f"I'm looking for {category}, but I'm still exploring."
            value = _payload(match.group("value"), 180, sentence=kind != "plain")
            if value is None or (kind == "plain" and _HARD_CUE.search(value)):
                return message
            if kind == "buying":
                return f"I'm looking for {category}. A key requirement is: {value}."
            return f"I'm looking for {category}. {value}"
        return message

    override = _OVERRIDE.fullmatch(cleaned)
    if override is not None:
        value = _payload(override.group("value"), 180)
        if value:
            return f"Actually, ignore my earlier preference. What I need is: {value}."
        return message

    if asked_attribute not in ALLOWED_ATTRIBUTES:
        return message
    negative = _NEGATIVE.fullmatch(cleaned)
    if negative is not None:
        attribute = negative.group("attribute").lower()
        if attribute != asked_attribute:
            return message
        kind = negative.group("kind").lower()
        if any(cue in kind for cue in ("additional", "else", "more")):
            return f"I don't have an additional preference for {attribute}."
        return f"I don't have a preference for {attribute}; please use your judgment."
    answer = _ANSWER.fullmatch(cleaned)
    if answer is not None:
        value = _payload(answer.group("value"), 362)
        if value:
            return f"For that, what matters is: {value}."
    return message
