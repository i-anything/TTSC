"""Bounded, privacy-preserving profile priors for candidate reranking.

Only the controlled ``preference_tags`` field is observed.  A parsed prior
retains a fixed theme bitmask and exposes a derived ranking digest; it never
retains input text.
Candidate evidence must already be expressed as a :class:`ProductTheme`
bitmask, so this module has no dependency on intent state or product records.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum, IntFlag
from types import MappingProxyType


MAX_PROFILE_TAGS = 16
MAX_PROFILE_TAG_CHARACTERS = 64
PROFILE_THEME_MASK_BYTES = 2
DEFAULT_PROFILE_RESIDUAL_WEIGHT = 0.05


class ProductTheme(IntFlag):
    """Fixed, generic themes that can safely form anonymous profile evidence."""

    NONE = 0
    COMFORT = 1 << 0
    DURABILITY = 1 << 1
    PERFORMANCE = 1 << 2
    WARMTH = 1 << 3
    WEATHER_PROTECTION = 1 << 4
    LIGHTWEIGHT = 1 << 5
    BREATHABILITY = 1 << 6
    EASY_CARE = 1 << 7
    VERSATILITY = 1 << 8
    SUSTAINABILITY = 1 << 9


_ALL_THEME_BITS = sum(theme.value for theme in ProductTheme)
_BOUNDED_POLICY_VERSION = "phase9-bounded-profile-residual-v1"
_DISABLED_POLICY_VERSION = "phase7-profile-disabled-v1"
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_DIMENSION_PREFIX_RE = re.compile(
    r"^(?:material|color|size|style|fit|brand|category|budget|price|"
    r"features?|use\s+case)(?:$|\s)"
)


def _normalized_tag(tag: str) -> str | None:
    folded = (
        unicodedata.normalize("NFKD", tag)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    normalized = _NON_ALNUM_RE.sub(" ", folded).strip()
    if not normalized:
        return None
    return normalized


_THEME_ALIASES = MappingProxyType(
    {
        "comfort": ProductTheme.COMFORT,
        "comfortable": ProductTheme.COMFORT,
        "comfort focused": ProductTheme.COMFORT,
        "comfort first": ProductTheme.COMFORT,
        "durability": ProductTheme.DURABILITY,
        "durable": ProductTheme.DURABILITY,
        "long lasting": ProductTheme.DURABILITY,
        "hard wearing": ProductTheme.DURABILITY,
        "rugged": ProductTheme.DURABILITY,
        "performance": ProductTheme.PERFORMANCE,
        "high performance": ProductTheme.PERFORMANCE,
        "performance focused": ProductTheme.PERFORMANCE,
        "technical performance": ProductTheme.PERFORMANCE,
        "warmth": ProductTheme.WARMTH,
        "warm": ProductTheme.WARMTH,
        "insulated": ProductTheme.WARMTH,
        "thermal": ProductTheme.WARMTH,
        "weather protection": ProductTheme.WEATHER_PROTECTION,
        "weather resistant": ProductTheme.WEATHER_PROTECTION,
        "weatherproof": ProductTheme.WEATHER_PROTECTION,
        "water resistant": ProductTheme.WEATHER_PROTECTION,
        "waterproof": ProductTheme.WEATHER_PROTECTION,
        "wind resistant": ProductTheme.WEATHER_PROTECTION,
        "windproof": ProductTheme.WEATHER_PROTECTION,
        "rain protection": ProductTheme.WEATHER_PROTECTION,
        "lightweight": ProductTheme.LIGHTWEIGHT,
        "light weight": ProductTheme.LIGHTWEIGHT,
        "ultralight": ProductTheme.LIGHTWEIGHT,
        "featherweight": ProductTheme.LIGHTWEIGHT,
        "breathability": ProductTheme.BREATHABILITY,
        "breathable": ProductTheme.BREATHABILITY,
        "ventilated": ProductTheme.BREATHABILITY,
        "airflow": ProductTheme.BREATHABILITY,
        "easy care": ProductTheme.EASY_CARE,
        "low maintenance": ProductTheme.EASY_CARE,
        "machine washable": ProductTheme.EASY_CARE,
        "washable": ProductTheme.EASY_CARE,
        "versatility": ProductTheme.VERSATILITY,
        "versatile": ProductTheme.VERSATILITY,
        "multi purpose": ProductTheme.VERSATILITY,
        "multipurpose": ProductTheme.VERSATILITY,
        "all purpose": ProductTheme.VERSATILITY,
        "sustainability": ProductTheme.SUSTAINABILITY,
        "sustainable": ProductTheme.SUSTAINABILITY,
        "eco friendly": ProductTheme.SUSTAINABILITY,
        "environmentally friendly": ProductTheme.SUSTAINABILITY,
        "recycled": ProductTheme.SUSTAINABILITY,
    }
)


def _ranking_digest(policy_version: str, theme_mask: ProductTheme) -> bytes:
    return hashlib.sha256(
        policy_version.encode("utf-8")
        + b"\0"
        + int(theme_mask).to_bytes(PROFILE_THEME_MASK_BYTES, "big")
    ).digest()


def _valid_theme_mask(value: object) -> ProductTheme | None:
    if not isinstance(value, ProductTheme):
        return None
    numeric = int(value)
    if numeric < 0 or numeric & ~_ALL_THEME_BITS:
        return None
    return value


@dataclass(frozen=True, slots=True, init=False)
class ProfilePrior:
    """Immutable ranking prior containing no raw or normalized profile text."""

    theme_mask: ProductTheme

    def __init__(self, theme_mask: ProductTheme = ProductTheme.NONE) -> None:
        validated = _valid_theme_mask(theme_mask)
        if validated is None:
            raise ValueError("theme_mask must contain only supported ProductTheme bits")
        object.__setattr__(self, "theme_mask", validated)

    @property
    def ranking_digest(self) -> bytes:
        """Derive the 32-byte bounded-policy dependency without retaining it."""

        return _ranking_digest(_BOUNDED_POLICY_VERSION, self.theme_mask)

    @property
    def active_theme_count(self) -> int:
        return int(self.theme_mask).bit_count()

    @property
    def is_neutral(self) -> bool:
        return self.theme_mask == ProductTheme.NONE


NEUTRAL_PROFILE_PRIOR = ProfilePrior()
EMPTY_PROFILE_PRIOR = NEUTRAL_PROFILE_PRIOR


def parse_profile_prior(user_profile: object) -> ProfilePrior:
    """Parse controlled tags, failing closed to a neutral prior.

    At most the first sixteen entries are inspected.  Unknown, dimension,
    malformed, and overlength tags are discarded independently, so they cannot
    become inferred theme evidence.  Fields other than ``preference_tags`` are
    never read.
    """

    if type(user_profile) is not dict:
        return NEUTRAL_PROFILE_PRIOR
    tags = user_profile.get("preference_tags")
    if type(tags) is not list or not tags:
        return NEUTRAL_PROFILE_PRIOR

    theme_mask = ProductTheme.NONE
    for tag in tags[:MAX_PROFILE_TAGS]:
        if type(tag) is not str or not 1 <= len(tag) <= MAX_PROFILE_TAG_CHARACTERS:
            continue
        normalized = _normalized_tag(tag)
        if normalized is None or _DIMENSION_PREFIX_RE.match(normalized):
            continue
        theme = _THEME_ALIASES.get(normalized)
        if theme is not None:
            theme_mask |= theme

    return (
        NEUTRAL_PROFILE_PRIOR
        if theme_mask == ProductTheme.NONE
        else ProfilePrior(theme_mask)
    )


class ProfilePolicy(Enum):
    """Reversible profile policies with no access to intent or product data."""

    DISABLED = _DISABLED_POLICY_VERSION
    BOUNDED_RESIDUAL = _BOUNDED_POLICY_VERSION

    def residual(
        self,
        prior: ProfilePrior,
        candidate_themes: ProductTheme,
    ) -> float:
        """Return the matched fraction of active prior themes in ``[0, 1]``.

        The ranking layer owns the explicit-intent gate and multiplies this
        fraction by :data:`DEFAULT_PROFILE_RESIDUAL_WEIGHT`.
        """

        if not isinstance(prior, ProfilePrior):
            return 0.0
        candidate_mask = _valid_theme_mask(candidate_themes)
        if (
            self is ProfilePolicy.DISABLED
            or prior.is_neutral
            or candidate_mask is None
        ):
            return 0.0
        overlap_count = int(prior.theme_mask & candidate_mask).bit_count()
        return overlap_count / prior.active_theme_count

    def ranking_digest(self, prior: ProfilePrior) -> bytes:
        """Return a deterministic cache dependency without profile text."""

        if not isinstance(prior, ProfilePrior):
            prior = NEUTRAL_PROFILE_PRIOR
        effective_mask = (
            ProductTheme.NONE
            if self is ProfilePolicy.DISABLED
            else prior.theme_mask
        )
        return _ranking_digest(self.value, effective_mask)


DISABLED_PROFILE_POLICY = ProfilePolicy.DISABLED
BOUNDED_RESIDUAL_PROFILE_POLICY = ProfilePolicy.BOUNDED_RESIDUAL
