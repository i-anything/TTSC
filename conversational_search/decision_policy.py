"""Lightweight opt-in policy identities for turn-level decision control."""

from __future__ import annotations

from enum import Enum


class DecisionPolicy(str, Enum):
    PROTECTED = "protected_phase13"
    PROTOCOL_UTILITY = "phase15-protocol-utility-v2"
    EXPECTED_UTILITY = "phase4-protocol-expected-utility-v1"


PROTECTED_DECISION_POLICY = DecisionPolicy.PROTECTED
PROTOCOL_UTILITY_DECISION_POLICY = DecisionPolicy.PROTOCOL_UTILITY
EXPECTED_UTILITY_DECISION_POLICY = DecisionPolicy.EXPECTED_UTILITY
PROTOCOL_DECISION_POLICIES = frozenset(
    {
        PROTOCOL_UTILITY_DECISION_POLICY,
        EXPECTED_UTILITY_DECISION_POLICY,
    }
)
