from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from conversational_search.intent import (
    CANONICAL_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
    IntentParsingPolicy,
    IntentState,
    apply_user_message,
    record_question,
    render_dense_query,
    render_lexical_query,
)
from conversational_search.service import ConversationalSearchAgent
from evaluator.local_evaluator import ALLOWED_ATTRIBUTES, catalog_index, evaluate, load_jsonl
from scripts.run_fusion_ablations import RouteHealthRetriever, _sha256
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_reranking_ablations import (
    METRIC_KEYS,
    RespondLatencyAgent,
    _expected_turns,
    _official_summary,
)


SCHEMA_VERSION = 1
SUITE_VERSION = "ttsc-message-robustness-v1"
MASTER_SEED = 20260829
REPLICATES = (1, 2, 3, 4, 5)
UNCHANGED_CUTOFF = 0.15
SURFACE_CUTOFF = 0.40
CHOICE_MAPPING_VERSION = "sha256-two-word-mode-variant-v1"
EXPECTED_PERTURBATION_BANK_SHA256 = (
    "be52c101df1194b517c7be136da1d67b616fa878d5e3c221d6026a6774db7d4a"
)
EXPECTED_COORDINATE_PLAN_SUITE_SHA256 = (
    "7e8987bfd92c113849d89888238b87a6495d7f1c3864e9dc7511953a0340289b"
)
EXPECTED_DECISION_PLAN_SUITE_SHA256 = (
    "f0879f8fc57a170a158cdbcf0f708e5b4cc215ac7d9d4603a2a1f30c0c7e223f"
)
EXPECTED_PERTURBATION_SPEC_SHA256 = (
    "a965e2da01c15bc4cfd768146169b12073bb13f51a27c56cb3464223f720175c"
)
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase6_implementation_lock.json"
PHASE5_METRICS = {
    "sample_count": 200,
    "hit_rate_at_10": 0.99,
    "mrr": 0.52223,
    "mttc": 3.07,
    "efficiency": 0.793,
    "recommended_technical_score": 0.810269,
}
PHASE5_SCENARIO_METRICS = {
    "boundary": {
        "sample_count": 10,
        "hit_rate_at_10": 0.9,
        "mrr": 0.385952,
        "mttc": 4.3,
    },
    "browsing": {
        "sample_count": 80,
        "hit_rate_at_10": 1.0,
        "mrr": 0.513219,
        "mttc": 2.5625,
    },
    "buying": {
        "sample_count": 80,
        "hit_rate_at_10": 0.9875,
        "mrr": 0.46,
        "mttc": 2.9,
    },
    "intent_override": {
        "sample_count": 30,
        "hit_rate_at_10": 1.0,
        "mrr": 0.757632,
        "mttc": 4.466667,
    },
}
EXPECTED_INPUT_SHA256 = {
    "catalog": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
    "dataset": "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579",
    "evaluator/local_evaluator.py": "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564",
    "docs/phase5_results.json": "44a20ecd9d7a1dddf65db570ad21a592f3d5ec2f34c6505e7294035d57342131",
    "docs/phase6_experiment_contract.json": "44d78b22d6ea71f2dbca4151df718459df6a456539af56bd79ec2831b272fc60",
    "docs/phase6_implementation_lock.json": "7df29ad6313c92f4330a693c4cd70b2d971e866f7fe3e05e149c286f14105720",
    "assets/bge-small-en-v1.5-int8/model_manifest.json": "f1130079f60555f7e35dc84344a33cd8e9afdcb4743c42afc94fb42b3991fd76",
    "assets/search-index-bge-small-en-v1.5-v2/manifest.json": "c9b7291004d6ef78473b24886899ea51f427fc2e179c8216c8e8b65f6cf929b2",
    "requirements-runtime.txt": "db8bcb738aa8a27746b78473e7ba0806b9ddb03011917a82285ee7ca0e9523b5",
    "conversational_search/questions.py": "900a0a129b12e5a9628cea845dd5e9dd859395d1c8d745b4ccef987ac4964cfd",
    "conversational_search/ranking.py": "495d2e6beafc2abd60f21cf92f0327c734d4b5722131dcfbf54c1c48036ddca2",
    "conversational_search/retrieval.py": "4b9a826e6579845b56d52b5a3eae87513647a64c11de25a18f2796f5996b54d2",
    "conversational_search/slates.py": "bb13addadcd73e8c112981c2bff9b99ee4c5aacdea3ae38a40f69239d23f8809",
    "conversational_search/strategy.py": "7f62efb080c19ecf6ded2d54ac960874daafb39ff7444553f7f1fcc2e07de9d0",
    "preprocessing/encoder.py": "8cf3f00fac5e473a4cb0001bac2462d228ec6b1359199e6c3ea5cdb17ce3ea5c",
    "starter/agent.py": "e661179798a99130a5b57eb9f195958d9a145435dd8a27caf669c77292a70e2b",
    "starter/dense.py": "013defff58a9aac956af70ee282c1653d44692309d0131d105514f27aa882ed5",
}
SOURCE_PATHS = (
    "assets/bge-small-en-v1.5-int8/model_manifest.json",
    "assets/search-index-bge-small-en-v1.5-v2/manifest.json",
    "conversational_search/intent.py",
    "conversational_search/questions.py",
    "conversational_search/ranking.py",
    "conversational_search/retrieval.py",
    "conversational_search/service.py",
    "conversational_search/slates.py",
    "conversational_search/strategy.py",
    "docs/phase5_results.json",
    "docs/phase6_experiment_contract.json",
    IMPLEMENTATION_LOCK_RELATIVE,
    "evaluator/local_evaluator.py",
    "preprocessing/encoder.py",
    "requirements-runtime.txt",
    "scripts/run_message_robustness.py",
    "scripts/run_fusion_ablations.py",
    "scripts/run_policy_ablations.py",
    "scripts/run_reranking_ablations.py",
    "starter/agent.py",
    "starter/dense.py",
    "tests/test_intent.py",
    "tests/test_message_robustness.py",
)


_BUYING_RE = re.compile(
    r"^I'm looking for (?P<category>.+?)\. A key requirement is: "
    r"(?P<value>.+)\.$",
    re.DOTALL,
)
_BROWSING_RE = re.compile(
    r"^I'm looking for (?P<category>.+?), but I'm still exploring\.$",
    re.DOTALL,
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
    r"^I don't have a preference for (?P<attribute>[a-z_ ]+); "
    r"please use your judgment\.$"
)
_NO_ADDITIONAL_RE = re.compile(
    r"^I don't have an additional preference for "
    r"(?P<attribute>[a-z_ ]+)\.$"
)
_NOT_RIGHT = (
    "Those options are not quite right yet. "
    "Ask me about one specific attribute."
)
_TENTATIVE_RE = re.compile(
    r"^I'm looking for (?P<category>.+?)\. (?P<value>.+)$",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    kind: str
    slots: tuple[tuple[str, str], ...] = ()

    def slot(self, name: str) -> str:
        for key, value in self.slots:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class Perturbation:
    message: str
    kind: str
    mode: str
    variant: int


class SlotCorruptionError(RuntimeError):
    pass


def parse_canonical_message(message: str) -> ParsedMessage | None:
    patterns = (
        ("buying", _BUYING_RE),
        ("browsing", _BROWSING_RE),
        ("answer", _ANSWER_RE),
        ("override", _OVERRIDE_RE),
        ("boundary_no_preference", _BOUNDARY_RE),
        ("no_additional_preference", _NO_ADDITIONAL_RE),
    )
    for kind, pattern in patterns:
        match = pattern.fullmatch(message)
        if match is not None:
            return ParsedMessage(kind, tuple(match.groupdict().items()))
    if message == _NOT_RIGHT:
        return ParsedMessage("request_question")
    tentative = _TENTATIVE_RE.fullmatch(message)
    if tentative is not None:
        return ParsedMessage("tentative", tuple(tentative.groupdict().items()))
    return None


def _surface_variants(parsed: ParsedMessage) -> tuple[str, ...]:
    kind = parsed.kind
    if kind == "buying":
        category, value = parsed.slot("category"), parsed.slot("value")
        return (
            f"I'm  looking for {category}.  A key requirement is: {value}.",
            f"i'm looking for {category}. a key requirement is: {value}.",
            f"I\u2019m looking for {category}. A key requirement is: {value}.",
            f"I'm looking for {category}. A key requirement is \u2014 {value}.",
        )
    if kind == "browsing":
        category = parsed.slot("category")
        return (
            f"I'm  looking for {category},  but I'm still exploring.",
            f"i'm looking for {category}, but i'm still exploring.",
            f"I\u2019m looking for {category}, but I\u2019m still exploring.",
        )
    if kind == "tentative":
        category, value = parsed.slot("category"), parsed.slot("value")
        return (
            f"I'm  looking for {category}.  {value}",
            f"i'm looking for {category}. {value}",
            f"I\u2019m looking for {category}. {value}",
        )
    if kind == "answer":
        value = parsed.slot("value")
        return (
            f"For that,  what matters is:  {value}.",
            f"for that, what matters is: {value}.",
            f"For that, what matters is \u2014 {value}.",
        )
    if kind == "override":
        value = parsed.slot("value")
        return (
            f"Actually,  ignore my earlier preference.  What I need is: {value}.",
            f"actually, ignore my earlier preference. what i need is: {value}.",
            f"Actually, ignore my earlier preference. What I need is \u2014 {value}.",
        )
    if kind in {"boundary_no_preference", "no_additional_preference"}:
        attribute = parsed.slot("attribute")
        if kind == "boundary_no_preference":
            return (
                f"I  don't have a preference for {attribute};  please use your judgment.",
                f"i don't have a preference for {attribute}; please use your judgment.",
                f"I don\u2019t have a preference for {attribute}; please use your judgment.",
            )
        return (
            f"I  don't have an additional preference for {attribute}.",
            f"i don't have an additional preference for {attribute}.",
            f"I don\u2019t have an additional preference for {attribute}.",
        )
    return (
        "Those  options are not quite right yet.  Ask me about one specific attribute.",
        "those options are not quite right yet. ask me about one specific attribute.",
    )


def _paraphrase_variants(parsed: ParsedMessage) -> tuple[str, ...]:
    kind = parsed.kind
    if kind == "buying":
        category, value = parsed.slot("category"), parsed.slot("value")
        return (
            f"I'm searching for {category}. My main requirement is: {value}.",
            f"For {category}, the key requirement is: {value}.",
        )
    if kind == "browsing":
        category = parsed.slot("category")
        return (
            f"I'm browsing for {category} and still exploring.",
            f"I'm considering {category}, but I have not narrowed it down yet.",
        )
    if kind == "tentative":
        category, value = parsed.slot("category"), parsed.slot("value")
        return (
            f"I'm considering {category}. One tentative preference is: {value}.",
            f"For {category}, my preference for now is: {value}.",
        )
    if kind == "answer":
        value = parsed.slot("value")
        return (
            f"For that, the important detail is: {value}.",
            f"What matters to me there is: {value}.",
        )
    if kind == "override":
        value = parsed.slot("value")
        return (
            f"Actually, disregard my earlier preference. I now need: {value}.",
            f"Change of plan: replace my earlier preference with {value}.",
        )
    attribute = parsed.slot("attribute") if parsed.slots else ""
    if kind == "boundary_no_preference":
        return (
            f"I have no preference for {attribute}; use your judgment.",
            f"For {attribute}, I do not have a preference.",
        )
    if kind == "no_additional_preference":
        return (
            f"I have no additional preference for {attribute}.",
            f"For {attribute}, I do not have another requirement.",
        )
    return (
        "Those still are not right. Please ask about one specific attribute.",
        "The options are not right yet; ask me one focused question.",
    )


_PERTURBATION_BANK_PROBES = (
    "I'm looking for Shoes. A key requirement is: leather.",
    "I'm looking for Shoes, but I'm still exploring.",
    "I'm looking for Shoes. leather",
    "For that, what matters is: leather.",
    "Actually, ignore my earlier preference. What I need is: cotton.",
    "I don't have a preference for material; please use your judgment.",
    "I don't have an additional preference for material.",
    "Those options are not quite right yet. Ask me about one specific attribute.",
)


def perturbation_bank_sha256() -> str:
    digest = hashlib.sha256()
    for message in _PERTURBATION_BANK_PROBES:
        parsed = parse_canonical_message(message)
        if parsed is None:
            raise RuntimeError("perturbation bank probe is not canonical")
        digest.update(f"{parsed.kind}\n".encode("ascii"))
        for variant in (*_surface_variants(parsed), *_paraphrase_variants(parsed)):
            digest.update(variant.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _choice_words(replicate: int, ordinal: int, turn: int) -> tuple[int, int]:
    coordinate = (
        f"{SUITE_VERSION}\0{MASTER_SEED}\0{replicate}\0{ordinal}\0"
        f"{turn}\0message_choice"
    ).encode("ascii")
    digest = hashlib.sha256(coordinate).digest()
    return (
        int.from_bytes(digest[:8], "big", signed=False),
        int.from_bytes(digest[8:16], "big", signed=False),
    )


def coordinate_plan_sha256(replicate: int, sample_count: int) -> str:
    """Hash the complete label-blind choice plan, including unvisited turns."""

    digest = hashlib.sha256()
    for ordinal in range(1, sample_count + 1):
        for turn in range(1, 11):
            first, second = _choice_words(replicate, ordinal, turn)
            digest.update(first.to_bytes(8, "big"))
            digest.update(second.to_bytes(8, "big"))
    return digest.hexdigest()


def coordinate_plan_suite_sha256(sample_count: int) -> str:
    digest = hashlib.sha256()
    for replicate in REPLICATES:
        digest.update(coordinate_plan_sha256(replicate, sample_count).encode("ascii"))
    return digest.hexdigest()


def _mode_and_variant(
    parsed: ParsedMessage,
    mode_word: int,
    variant_word: int,
) -> tuple[str, int]:
    unit = mode_word / 2**64
    if unit < UNCHANGED_CUTOFF:
        mode = "unchanged"
        variant_count = 1
    elif unit < SURFACE_CUTOFF:
        mode = "surface"
        variant_count = len(_surface_variants(parsed))
    else:
        mode = "paraphrase"
        variant_count = len(_paraphrase_variants(parsed))
    return mode, variant_word % variant_count


def decision_plan_suite_sha256(sample_count: int) -> str:
    """Hash every family/mode/variant decision for every possible coordinate."""

    parsed_probes = []
    for message in _PERTURBATION_BANK_PROBES:
        parsed = parse_canonical_message(message)
        if parsed is None:
            raise RuntimeError("decision-plan probe is not canonical")
        parsed_probes.append(parsed)
    digest = hashlib.sha256()
    for replicate in REPLICATES:
        for ordinal in range(1, sample_count + 1):
            for turn in range(1, 11):
                mode_word, variant_word = _choice_words(replicate, ordinal, turn)
                for parsed in parsed_probes:
                    mode, variant = _mode_and_variant(
                        parsed,
                        mode_word,
                        variant_word,
                    )
                    digest.update(
                        f"{parsed.kind}\0{mode}\0{variant}\n".encode("ascii")
                    )
    return digest.hexdigest()


def perturbation_spec_sha256(sample_count: int) -> str:
    """Hash the executable protocol, including thresholds and realized choices."""

    fields = (
        ("suite_version", SUITE_VERSION),
        ("master_seed", str(MASTER_SEED)),
        ("replicates", ",".join(str(item) for item in REPLICATES)),
        ("unchanged_cutoff", format(UNCHANGED_CUTOFF, ".17g")),
        ("surface_cutoff", format(SURFACE_CUTOFF, ".17g")),
        ("choice_mapping_version", CHOICE_MAPPING_VERSION),
        (
            "contract_sha256",
            EXPECTED_INPUT_SHA256["docs/phase6_experiment_contract.json"],
        ),
        ("perturbation_bank_sha256", perturbation_bank_sha256()),
        ("coordinate_plan_suite_sha256", coordinate_plan_suite_sha256(sample_count)),
        ("decision_plan_suite_sha256", decision_plan_suite_sha256(sample_count)),
    )
    digest = hashlib.sha256()
    for key, value in fields:
        digest.update(f"{key}\0{value}\n".encode("ascii"))
    return digest.hexdigest()


def _validate_protocol_goldens(
    repository_root: Path,
    sample_count: int,
) -> dict[str, str]:
    observed = {
        "perturbation_bank_sha256": perturbation_bank_sha256(),
        "coordinate_plan_suite_sha256": coordinate_plan_suite_sha256(sample_count),
        "decision_plan_suite_sha256": decision_plan_suite_sha256(sample_count),
        "perturbation_spec_sha256": perturbation_spec_sha256(sample_count),
    }
    expected = {
        "perturbation_bank_sha256": EXPECTED_PERTURBATION_BANK_SHA256,
        "coordinate_plan_suite_sha256": EXPECTED_COORDINATE_PLAN_SUITE_SHA256,
        "decision_plan_suite_sha256": EXPECTED_DECISION_PLAN_SUITE_SHA256,
        "perturbation_spec_sha256": EXPECTED_PERTURBATION_SPEC_SHA256,
    }
    lock_path = repository_root / IMPLEMENTATION_LOCK_RELATIVE
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Phase 6 implementation lock is unreadable") from error
    expected_lock = {
        "schema_version": 1,
        "lock_id": "phase6-message-perturbation-implementation-v1",
        "status": "locked_before_final_confirmatory_run",
        "sample_count": sample_count,
        "suite_version": SUITE_VERSION,
        "master_seed": MASTER_SEED,
        "replicates": list(REPLICATES),
        "unchanged_cutoff": UNCHANGED_CUTOFF,
        "surface_cutoff": SURFACE_CUTOFF,
        "choice_mapping_version": CHOICE_MAPPING_VERSION,
        "contract_sha256": EXPECTED_INPUT_SHA256[
            "docs/phase6_experiment_contract.json"
        ],
        **expected,
    }
    if lock != expected_lock or observed != expected:
        raise RuntimeError("frozen Phase 6 perturbation protocol drifted")
    return observed


def perturb_message(
    message: str,
    *,
    replicate: int | None,
    ordinal: int,
    turn: int,
) -> Perturbation:
    parsed = parse_canonical_message(message)
    if replicate is None or parsed is None:
        return Perturbation(
            message=message,
            kind="unknown" if parsed is None else parsed.kind,
            mode="unchanged",
            variant=0,
        )

    mode_word, variant_word = _choice_words(replicate, ordinal, turn)
    mode, variant = _mode_and_variant(parsed, mode_word, variant_word)
    if mode == "unchanged":
        variants = (message,)
    elif mode == "surface":
        variants = _surface_variants(parsed)
    else:
        variants = _paraphrase_variants(parsed)
    transformed = variants[variant]
    for _, slot in parsed.slots:
        if slot not in transformed:
            raise SlotCorruptionError("message perturbation corrupted a payload slot")
    return Perturbation(transformed, parsed.kind, mode, variant)


class SeededMessageAgent:
    """Experiment-only adapter that perturbs visible messages without labels."""

    def __init__(
        self,
        delegate: ConversationalSearchAgent,
        *,
        replicate: int | None,
        policy: IntentParsingPolicy,
        state_reader: Callable[[str], IntentState] | None = None,
    ) -> None:
        self._delegate = delegate
        self._replicate = replicate
        self._policy = policy
        self._state_reader = state_reader
        self._next_ordinal = 0
        self._ordinals: dict[str, int] = {}
        self._reference: dict[str, IntentState] = {}
        self._observed: dict[str, IntentState] = {}
        self._families: Counter[str] = Counter()
        self._modes: Counter[str] = Counter()
        self._known_messages = 0
        self._unknown_messages = 0
        self._state_checks = 0
        self._state_matches = 0
        self._query_matches = 0
        self._critical_checks = 0
        self._critical_matches = 0
        self._integration_checks = 0
        self._integration_matches = 0
        self._state_mismatch_families: Counter[str] = Counter()
        self._query_mismatch_families: Counter[str] = Counter()
        self._state_mismatch_components: Counter[str] = Counter()
        self._response_exceptions = 0
        self._invalid_responses = 0
        self._input_processing_exceptions = 0
        self._slot_corruptions = 0
        self._trace = hashlib.sha256()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._next_ordinal += 1
        self._ordinals[session_id] = self._next_ordinal
        self._reference[session_id] = IntentState()
        self._observed[session_id] = IntentState()
        self._delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            perturbation = perturb_message(
                user_message,
                replicate=self._replicate,
                ordinal=self._ordinals[session_id],
                turn=turn,
            )
        except SlotCorruptionError:
            self._slot_corruptions += 1
            self._input_processing_exceptions += 1
            raise
        except Exception:
            self._input_processing_exceptions += 1
            raise
        self._families[perturbation.kind] += 1
        self._modes[perturbation.mode] += 1
        if perturbation.kind == "unknown":
            self._unknown_messages += 1
        else:
            self._known_messages += 1
        self._trace.update(
            f"{perturbation.kind}\0{perturbation.mode}\0"
            f"{perturbation.variant}\n".encode("ascii")
        )

        try:
            reference = apply_user_message(
                self._reference[session_id],
                user_message,
                turn,
                policy=CANONICAL_INTENT_POLICY,
            )
            observed = apply_user_message(
                self._observed[session_id],
                perturbation.message,
                turn,
                policy=self._policy,
            )
        except Exception:
            self._input_processing_exceptions += 1
            raise
        self._state_checks += 1
        state_equal = observed == reference
        self._state_matches += int(state_equal)
        if not state_equal:
            self._state_mismatch_families[perturbation.kind] += 1
            for component in (
                "category",
                "requirements",
                "excluded",
                "no_preference",
                "asked_attributes",
                "last_asked_attribute",
                "intent_version",
                "last_turn",
            ):
                if getattr(observed, component) != getattr(reference, component):
                    self._state_mismatch_components[component] += 1
        queries_equal = (
            render_dense_query(observed) == render_dense_query(reference)
            and render_lexical_query(observed) == render_lexical_query(reference)
        )
        self._query_matches += int(queries_equal)
        if not queries_equal:
            self._query_mismatch_families[perturbation.kind] += 1
        if perturbation.kind in {
            "override",
            "boundary_no_preference",
            "no_additional_preference",
        }:
            self._critical_checks += 1
            self._critical_matches += int(state_equal and queries_equal)

        try:
            response = self._delegate.respond(
                session_id,
                perturbation.message,
                turn,
                top_k,
            )
        except Exception:
            self._response_exceptions += 1
            raise
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            self._invalid_responses += 1
            return response

        ask_attribute = response.get("ask_attribute")
        if isinstance(ask_attribute, str) and ask_attribute in ALLOWED_ATTRIBUTES:
            reference = record_question(reference, ask_attribute)
            observed = record_question(observed, ask_attribute)
        self._reference[session_id] = reference
        self._observed[session_id] = observed
        if self._state_reader is not None:
            self._integration_checks += 1
            self._integration_matches += int(
                self._state_reader(session_id) == observed
            )
        return response

    def summary(self) -> dict:
        transformed = self._modes["surface"] + self._modes["paraphrase"]
        return {
            "known_messages": self._known_messages,
            "unknown_messages": self._unknown_messages,
            "transformed_messages": transformed,
            "family_counts": dict(sorted(self._families.items())),
            "mode_counts": dict(sorted(self._modes.items())),
            "state_checks": self._state_checks,
            "state_matches": self._state_matches,
            "query_matches": self._query_matches,
            "critical_checks": self._critical_checks,
            "critical_matches": self._critical_matches,
            "integration_checks": self._integration_checks,
            "integration_matches": self._integration_matches,
            "response_exceptions": self._response_exceptions,
            "invalid_responses": self._invalid_responses,
            "input_processing_exceptions": self._input_processing_exceptions,
            "slot_corruptions": self._slot_corruptions,
            "state_mismatch_families": dict(
                sorted(self._state_mismatch_families.items())
            ),
            "query_mismatch_families": dict(
                sorted(self._query_mismatch_families.items())
            ),
            "state_mismatch_components": dict(
                sorted(self._state_mismatch_components.items())
            ),
            "choice_trace_sha256": self._trace.hexdigest(),
        }


def _validate_phase5_control(result: dict) -> None:
    observed = {
        key: result[key]
        for key in (
            "sample_count",
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "efficiency",
            "recommended_technical_score",
        )
    }
    if (
        observed != PHASE5_METRICS
        or result.get("scenario_metrics") != PHASE5_SCENARIO_METRICS
    ):
        raise RuntimeError("exact control metrics drifted from Phase 5")


def _validate_frozen_inputs(
    repository_root: Path,
    catalog: Path,
    dataset: Path,
) -> dict[str, str]:
    paths = {
        "catalog": catalog,
        "dataset": dataset,
        **{
            relative: repository_root / relative
            for relative in EXPECTED_INPUT_SHA256
            if relative not in {"catalog", "dataset"}
        },
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    if observed != EXPECTED_INPUT_SHA256:
        drifted = sorted(
            name
            for name, expected in EXPECTED_INPUT_SHA256.items()
            if observed.get(name) != expected
        )
        raise RuntimeError(f"frozen Phase 6 inputs drifted: {drifted}")
    return observed


def _validate_run_health(
    expected_turns: int,
    route_health: dict,
    ranking_health: dict,
    slate_health: dict,
    message_audit: dict,
) -> None:
    if route_health.get("fallback_turns") != 0:
        raise RuntimeError("fallback turns invalidate message robustness")
    expected_ranking = {
        "attempts": expected_turns,
        "successes": expected_turns,
        "failures": 0,
        "unavailable_skips": 0,
    }
    if any(
        ranking_health.get(key) != value
        for key, value in expected_ranking.items()
    ):
        raise RuntimeError("reranker health invalidates message robustness")
    expected_slate = {
        "attempts": expected_turns,
        "successes": expected_turns,
        "failures": 0,
    }
    if any(
        slate_health.get(key) != value for key, value in expected_slate.items()
    ):
        raise RuntimeError("slate health invalidates message robustness")
    fault_keys = (
        "response_exceptions",
        "invalid_responses",
        "input_processing_exceptions",
        "slot_corruptions",
    )
    if any(message_audit.get(key) != 0 for key in fault_keys):
        raise RuntimeError("message adapter health invalidates robustness run")
    coverage_keys = (
        "known_messages",
        "state_checks",
        "integration_checks",
        "integration_matches",
    )
    if (
        message_audit.get("known_messages", 0)
        + message_audit.get("unknown_messages", 0)
        != expected_turns
        or any(
            message_audit.get(key) != expected_turns
            for key in coverage_keys[1:]
        )
    ):
        raise RuntimeError("message state-audit coverage is incomplete")


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    *,
    policy: IntentParsingPolicy,
    replicate: int | None,
) -> tuple[dict, dict]:
    guarded = RouteHealthRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded,
        intent_policy=policy,
    )
    timed = RespondLatencyAgent(agent)
    seeded = SeededMessageAgent(  # type: ignore[arg-type]
        timed,
        replicate=replicate,
        policy=policy,
        state_reader=agent.session_state,
    )
    started = time.perf_counter()
    result = evaluate(seeded, samples, catalog_ids, categories, products)
    elapsed = time.perf_counter() - started
    expected_turns = _expected_turns(result)
    audit = seeded.summary()
    guarded.validate(expected_turns)
    latency = timed.latency_summary()
    if latency["count"] != expected_turns:
        raise RuntimeError("response timing coverage is incomplete")
    route_health = guarded.summary()
    ranking_health = agent.ranking_health
    slate_health = agent.slate_health
    _validate_run_health(
        expected_turns,
        route_health,
        ranking_health,
        slate_health,
        audit,
    )
    return result, {
        "route_health": route_health,
        "ranking_health": ranking_health,
        "slate_health": slate_health,
        "message_audit": audit,
        "evaluation_wall_seconds": round(elapsed, 6),
        "respond_latency_ms": latency,
    }


def _distribution(results: list[dict]) -> dict:
    output: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [float(result[key]) for result in results]
        higher_is_better = key != "mttc"
        output[key] = {
            "mean": round(statistics.fmean(values), 6),
            "population_stddev": round(statistics.pstdev(values), 6),
            "worst": round(min(values) if higher_is_better else max(values), 6),
            "best": round(max(values) if higher_is_better else min(values), 6),
        }
    return output


def _scenario_distribution(results: list[dict]) -> dict:
    names = sorted(
        set().union(
            *((result.get("scenario_metrics") or {}).keys() for result in results)
        )
    )
    output: dict[str, dict] = {}
    for name in names:
        scenario_runs = [result["scenario_metrics"][name] for result in results]
        sample_count = int(scenario_runs[0]["sample_count"])
        metrics: dict[str, dict[str, float]] = {}
        for key in ("hit_rate_at_10", "mrr", "mttc"):
            values = [float(run[key]) for run in scenario_runs]
            higher_is_better = key != "mttc"
            metrics[key] = {
                "mean": round(statistics.fmean(values), 6),
                "population_stddev": round(statistics.pstdev(values), 6),
                "worst": round(min(values) if higher_is_better else max(values), 6),
            }
        output[name] = {"sample_count_per_replicate": sample_count, **metrics}
    return output


def _mean_delta(baseline: dict, candidate: dict) -> dict[str, float]:
    return {
        key: round(
            float(candidate[key]["mean"]) - float(baseline[key]["mean"]),
            6,
        )
        for key in METRIC_KEYS
    }


def _paired_outcomes(
    baseline_results: list[dict], candidate_results: list[dict]
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    scenario_hit_delta: Counter[str] = Counter()
    worst_replicate_scenario_hit_delta: dict[str, int] = {}
    for baseline, candidate in zip(baseline_results, candidate_results, strict=True):
        baseline_sessions = baseline.get("sessions") or []
        candidate_sessions = candidate.get("sessions") or []
        if len(baseline_sessions) != len(candidate_sessions):
            raise RuntimeError("paired robustness runs have different sample counts")
        replicate_scenario_delta: Counter[str] = Counter()
        for left, right in zip(baseline_sessions, candidate_sessions, strict=True):
            if (
                left.get("sample_id") != right.get("sample_id")
                or left.get("scenario_type") != right.get("scenario_type")
            ):
                raise RuntimeError("paired robustness run order drifted")
            left_hit = left.get("hit") is True
            right_hit = right.get("hit") is True
            if left_hit and right_hit:
                counts["both_hit"] += 1
                left_turn = int(left["first_hit_turn"])
                right_turn = int(right["first_hit_turn"])
                counts[
                    "hit_earlier" if right_turn < left_turn else
                    "hit_later" if right_turn > left_turn else
                    "hit_turn_same"
                ] += 1
                left_rank = int(left["best_rank"])
                right_rank = int(right["best_rank"])
                counts[
                    "rank_improved" if right_rank < left_rank else
                    "rank_worsened" if right_rank > left_rank else
                    "rank_same"
                ] += 1
            elif right_hit:
                counts["rescued_hit"] += 1
            elif left_hit:
                counts["regressed_hit"] += 1
            else:
                counts["both_miss"] += 1
            scenario = str(left["scenario_type"])
            hit_delta = int(right_hit) - int(left_hit)
            scenario_hit_delta[scenario] += hit_delta
            replicate_scenario_delta[scenario] += hit_delta
        for scenario, delta in replicate_scenario_delta.items():
            worst_replicate_scenario_hit_delta[scenario] = min(
                delta,
                worst_replicate_scenario_hit_delta.get(scenario, delta),
            )
    keys = (
        "both_hit",
        "rescued_hit",
        "regressed_hit",
        "both_miss",
        "hit_earlier",
        "hit_turn_same",
        "hit_later",
        "rank_improved",
        "rank_same",
        "rank_worsened",
    )
    return {
        **{key: counts[key] for key in keys},
        "pooled_scenario_hit_delta": dict(sorted(scenario_hit_delta.items())),
        "worst_replicate_scenario_hit_delta": dict(
            sorted(worst_replicate_scenario_hit_delta.items())
        ),
    }


def _sum_audits(diagnostics: list[dict]) -> dict:
    numeric_keys = (
        "known_messages",
        "unknown_messages",
        "transformed_messages",
        "state_checks",
        "state_matches",
        "query_matches",
        "critical_checks",
        "critical_matches",
        "integration_checks",
        "integration_matches",
        "response_exceptions",
        "invalid_responses",
        "input_processing_exceptions",
        "slot_corruptions",
    )
    totals = Counter()
    families: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    state_mismatch_families: Counter[str] = Counter()
    query_mismatch_families: Counter[str] = Counter()
    state_mismatch_components: Counter[str] = Counter()
    for diagnostic in diagnostics:
        audit = diagnostic["message_audit"]
        totals.update({key: int(audit[key]) for key in numeric_keys})
        families.update(audit["family_counts"])
        modes.update(audit["mode_counts"])
        state_mismatch_families.update(audit["state_mismatch_families"])
        query_mismatch_families.update(audit["query_mismatch_families"])
        state_mismatch_components.update(audit["state_mismatch_components"])
    checks = totals["state_checks"]
    critical = totals["critical_checks"]
    integration = totals["integration_checks"]
    return {
        **{key: totals[key] for key in numeric_keys},
        "state_equivalence_rate": round(totals["state_matches"] / checks, 6),
        "query_equivalence_rate": round(totals["query_matches"] / checks, 6),
        "critical_equivalence_rate": round(
            totals["critical_matches"] / critical if critical else 1.0,
            6,
        ),
        "service_integration_equivalence_rate": round(
            totals["integration_matches"] / integration if integration else 1.0,
            6,
        ),
        "family_counts": dict(sorted(families.items())),
        "mode_counts": dict(sorted(modes.items())),
        "state_mismatch_families": dict(sorted(state_mismatch_families.items())),
        "query_mismatch_families": dict(sorted(query_mismatch_families.items())),
        "state_mismatch_components": dict(
            sorted(state_mismatch_components.items())
        ),
    }


def _sum_health(diagnostics: list[dict]) -> dict:
    route_bm25: Counter[str] = Counter()
    route_dense: Counter[str] = Counter()
    ranking = Counter()
    slate = Counter()
    fallbacks = 0
    for diagnostic in diagnostics:
        route = diagnostic["route_health"]
        route_bm25.update(route["bm25"])
        route_dense.update(route["dense"])
        fallbacks += int(route["fallback_turns"])
        ranking.update(
            {
                key: int(value)
                for key, value in diagnostic["ranking_health"].items()
                if key != "policy"
            }
        )
        slate.update(
            {
                key: int(value)
                for key, value in diagnostic["slate_health"].items()
                if key != "policy"
            }
        )
    return {
        "bm25": dict(sorted(route_bm25.items())),
        "dense": dict(sorted(route_dense.items())),
        "fallback_turns": fallbacks,
        "ranking": dict(sorted(ranking.items())),
        "slate": dict(sorted(slate.items())),
    }


def _latency_summary(
    baseline: list[dict], candidate: list[dict]
) -> dict[str, float]:
    baseline_p95 = [float(item["respond_latency_ms"]["warm_p95"]) for item in baseline]
    candidate_p95 = [float(item["respond_latency_ms"]["warm_p95"]) for item in candidate]
    ratios = [
        right / left if left > 0 else math.inf
        for left, right in zip(baseline_p95, candidate_p95, strict=True)
    ]
    return {
        "baseline_mean_warm_p95_ms": round(statistics.fmean(baseline_p95), 6),
        "candidate_mean_warm_p95_ms": round(statistics.fmean(candidate_p95), 6),
        "maximum_paired_warm_p95_ratio": round(max(ratios), 6),
        "baseline_total_wall_seconds": round(
            sum(float(item["evaluation_wall_seconds"]) for item in baseline), 6
        ),
        "candidate_total_wall_seconds": round(
            sum(float(item["evaluation_wall_seconds"]) for item in candidate), 6
        ),
    }


_PUBLICATION_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "candidate",
        "baseline",
        "run_configuration",
        "control",
        "robustness",
        "semantic_equivalence",
        "health",
        "latency",
        "determinism",
        "decision_gate",
        "privacy",
        "reproducibility",
    }
)
_ASIN_RE = re.compile(r"B[A-Z0-9]{9}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class _ScalarSchema:
    kind: str


@dataclass(frozen=True, slots=True)
class _SubsetMapSchema:
    allowed_keys: frozenset[str]
    value_schema: object


_BOOLEAN = _ScalarSchema("boolean")
_INTEGER = _ScalarSchema("integer")
_NUMBER = _ScalarSchema("number")
_STRING = _ScalarSchema("string")
_SHA256 = _ScalarSchema("sha256")

_DECISION_GATE_KEYS = frozenset(
    {
        "baseline_control_metrics_match_phase5",
        "candidate_control_evaluator_payload_matches_baseline",
        "candidate_parser_state_equivalence_is_one",
        "candidate_rendered_query_equivalence_is_one",
        "critical_equivalence_is_one",
        "service_integration_equivalence_is_one",
        "unrecognized_canonical_messages_are_zero",
        "message_adapter_faults_are_zero",
        "mean_technical_score_gain_at_least_0_02",
        "mean_hit_rate_gain_at_least_0_02",
        "worst_hit_rate_at_least_0_95",
        "worst_mrr_at_least_0_49",
        "worst_mttc_at_most_3_5",
        "worst_technical_score_at_least_0_78",
        "no_pooled_scenario_hit_loss",
        "warm_p95_ratio_at_most_1_15",
        "route_reranker_slate_and_fallback_faults_are_zero",
        "coordinate_plans_are_deterministic_and_distinct",
        "candidate_replay_payload_and_trace_identical",
        "aggregate_privacy_projection_valid",
        "adopt",
    }
)


def _aggregate_publication_schema() -> dict[str, object]:
    scenario_names = tuple(PHASE5_SCENARIO_METRICS)
    family_names = frozenset(
        {
            "answer",
            "boundary_no_preference",
            "browsing",
            "buying",
            "no_additional_preference",
            "override",
            "request_question",
            "tentative",
            "unknown",
        }
    )
    state_components = frozenset(
        {
            "category",
            "requirements",
            "excluded",
            "no_preference",
            "asked_attributes",
            "last_asked_attribute",
            "intent_version",
            "last_turn",
        }
    )
    token_usage = {
        "prompt_tokens": _INTEGER,
        "completion_tokens": _INTEGER,
        "total_tokens": _INTEGER,
    }
    scenario_metric = {
        "sample_count": _INTEGER,
        "hit_rate_at_10": _NUMBER,
        "mrr": _NUMBER,
        "mttc": _NUMBER,
    }
    scenario_metrics = _SubsetMapSchema(
        frozenset(scenario_names),
        scenario_metric,
    )
    official_summary = {
        "sample_count": _INTEGER,
        **{key: _NUMBER for key in METRIC_KEYS},
        "reported_token_usage": token_usage,
        "scenario_metrics": scenario_metrics,
    }
    route = {
        "bm25": _SubsetMapSchema(frozenset({"ok", "empty"}), _INTEGER),
        "dense": _SubsetMapSchema(frozenset({"ok", "empty"}), _INTEGER),
        "fallback_turns": _INTEGER,
    }
    ranking = {
        "attempts": _INTEGER,
        "successes": _INTEGER,
        "failures": _INTEGER,
        "unavailable_skips": _INTEGER,
    }
    slate = {
        "attempts": _INTEGER,
        "successes": _INTEGER,
        "failures": _INTEGER,
        "initializations": _INTEGER,
        "ranking_resets": _INTEGER,
        "stagnant_turns": _INTEGER,
        "unseen_selected_on_stagnant": _INTEGER,
        "repeat_backfills": _INTEGER,
    }
    control_health = {
        "route": route,
        "ranking": {"policy": _STRING, **ranking},
        "slate": {"policy": _STRING, **slate},
    }
    aggregate_health = {
        "bm25": route["bm25"],
        "dense": route["dense"],
        "fallback_turns": _INTEGER,
        "ranking": ranking,
        "slate": slate,
    }
    metric_distribution = {
        "mean": _NUMBER,
        "population_stddev": _NUMBER,
        "worst": _NUMBER,
        "best": _NUMBER,
    }
    distribution = {key: metric_distribution for key in METRIC_KEYS}
    scenario_distribution_metric = {
        "mean": _NUMBER,
        "population_stddev": _NUMBER,
        "worst": _NUMBER,
    }
    scenario_distribution = _SubsetMapSchema(
        frozenset(scenario_names),
        {
            "sample_count_per_replicate": _INTEGER,
            "hit_rate_at_10": scenario_distribution_metric,
            "mrr": scenario_distribution_metric,
            "mttc": scenario_distribution_metric,
        },
    )
    scenario_counter = _SubsetMapSchema(frozenset(scenario_names), _INTEGER)
    audit = {
        **{
            key: _INTEGER
            for key in (
                "known_messages",
                "unknown_messages",
                "transformed_messages",
                "state_checks",
                "state_matches",
                "query_matches",
                "critical_checks",
                "critical_matches",
                "integration_checks",
                "integration_matches",
                "response_exceptions",
                "invalid_responses",
                "input_processing_exceptions",
                "slot_corruptions",
            )
        },
        "state_equivalence_rate": _NUMBER,
        "query_equivalence_rate": _NUMBER,
        "critical_equivalence_rate": _NUMBER,
        "service_integration_equivalence_rate": _NUMBER,
        "family_counts": _SubsetMapSchema(family_names, _INTEGER),
        "mode_counts": _SubsetMapSchema(
            frozenset({"unchanged", "surface", "paraphrase"}),
            _INTEGER,
        ),
        "state_mismatch_families": _SubsetMapSchema(family_names, _INTEGER),
        "query_mismatch_families": _SubsetMapSchema(family_names, _INTEGER),
        "state_mismatch_components": _SubsetMapSchema(
            state_components,
            _INTEGER,
        ),
    }
    return {
        "schema_version": _INTEGER,
        "experiment_id": _STRING,
        "candidate": _STRING,
        "baseline": _STRING,
        "run_configuration": {
            "suite_version": _STRING,
            "master_seed": _INTEGER,
            "replicate_count": _INTEGER,
            "sample_count_per_replicate": _INTEGER,
            "execution": _STRING,
            "onnx_threads": _INTEGER,
            "shared_immutable_backend": _BOOLEAN,
            "external_api_calls": _INTEGER,
        },
        "control": {
            "baseline": official_summary,
            "candidate": official_summary,
            "evaluator_payloads_equal": _BOOLEAN,
            "baseline_health": control_health,
            "candidate_health": control_health,
        },
        "robustness": {
            "baseline_distribution": distribution,
            "candidate_distribution": distribution,
            "mean_metric_delta": {key: _NUMBER for key in METRIC_KEYS},
            "baseline_scenario_distribution": scenario_distribution,
            "candidate_scenario_distribution": scenario_distribution,
            "paired_outcomes": {
                **{
                    key: _INTEGER
                    for key in (
                        "both_hit",
                        "rescued_hit",
                        "regressed_hit",
                        "both_miss",
                        "hit_earlier",
                        "hit_turn_same",
                        "hit_later",
                        "rank_improved",
                        "rank_same",
                        "rank_worsened",
                    )
                },
                "pooled_scenario_hit_delta": scenario_counter,
                "worst_replicate_scenario_hit_delta": scenario_counter,
            },
        },
        "semantic_equivalence": {
            "baseline": audit,
            "candidate": audit,
        },
        "health": {
            "baseline": aggregate_health,
            "candidate": aggregate_health,
        },
        "latency": {
            "baseline_mean_warm_p95_ms": _NUMBER,
            "candidate_mean_warm_p95_ms": _NUMBER,
            "maximum_paired_warm_p95_ratio": _NUMBER,
            "baseline_total_wall_seconds": _NUMBER,
            "candidate_total_wall_seconds": _NUMBER,
        },
        "determinism": {
            "perturbation_bank_sha256": _SHA256,
            "coordinate_plan_suite_sha256": _SHA256,
            "decision_plan_suite_sha256": _SHA256,
            "perturbation_spec_sha256": _SHA256,
            "candidate_replay_evaluator_payload_equal": _BOOLEAN,
            "candidate_replay_choice_trace_equal": _BOOLEAN,
            "candidate_replay_diagnostics_equal": _BOOLEAN,
        },
        "decision_gate": {
            key: _BOOLEAN for key in _DECISION_GATE_KEYS
        },
        "privacy": {
            "contains_queries_or_messages": _BOOLEAN,
            "contains_profiles": _BOOLEAN,
            "contains_product_or_sample_ids": _BOOLEAN,
            "contains_session_or_turn_rows": _BOOLEAN,
            "contains_per_replicate_rows": _BOOLEAN,
            "choice_trace_hashes_raw_messages": _BOOLEAN,
        },
        "reproducibility": {
            "platform": _STRING,
            "python": _STRING,
            "catalog_sha256": _SHA256,
            "dataset_sha256": _SHA256,
            "frozen_input_sha256": {
                key: _SHA256 for key in EXPECTED_INPUT_SHA256
            },
            "source_sha256": {key: _SHA256 for key in SOURCE_PATHS},
        },
    }


def _validate_schema_node(
    value: object,
    schema: object,
    path: tuple[str, ...],
) -> None:
    label = ".".join(path) or "<root>"
    if isinstance(schema, dict):
        if not isinstance(value, dict) or set(value) != set(schema):
            raise RuntimeError(f"aggregate publication schema drifted at {label}")
        for key, child_schema in schema.items():
            _validate_schema_node(value[key], child_schema, (*path, key))
        return
    if isinstance(schema, _SubsetMapSchema):
        if not isinstance(value, dict) or not set(value) <= schema.allowed_keys:
            raise RuntimeError(f"aggregate publication map drifted at {label}")
        for key, child in value.items():
            _validate_schema_node(child, schema.value_schema, (*path, key))
        return
    if not isinstance(schema, _ScalarSchema):
        raise TypeError(f"invalid aggregate schema at {label}")
    if schema.kind == "boolean":
        valid = type(value) is bool
    elif schema.kind == "integer":
        valid = type(value) is int
    elif schema.kind == "number":
        valid = type(value) in {int, float} and math.isfinite(float(value))
    elif schema.kind == "string":
        valid = isinstance(value, str) and not _ASIN_RE.fullmatch(value)
    elif schema.kind == "sha256":
        valid = isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
    else:
        raise TypeError(f"unknown aggregate scalar schema at {label}")
    if not valid:
        raise RuntimeError(f"aggregate publication value drifted at {label}")


def _validate_aggregate_privacy(payload: dict) -> None:
    if set(payload) != _PUBLICATION_TOP_LEVEL_KEYS:
        raise RuntimeError("aggregate publication top-level schema drifted")
    _validate_schema_node(payload, _aggregate_publication_schema(), ())


def run_message_robustness(
    catalog_path: str | Path,
    dataset_path: str | Path,
) -> dict:
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    frozen_input_sha256 = _validate_frozen_inputs(
        repository_root,
        catalog,
        dataset,
    )
    samples = load_jsonl(dataset)
    protocol_goldens = _validate_protocol_goldens(
        repository_root,
        len(samples),
    )
    catalog_ids, categories, products = catalog_index(catalog)

    runtime = ConversationalSearchAgent(
        catalog,
        intent_policy=CANONICAL_INTENT_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing robustness run")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing robustness run")

    baseline_control, baseline_control_diagnostic = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        policy=CANONICAL_INTENT_POLICY,
        replicate=None,
    )
    _validate_phase5_control(baseline_control)
    candidate_control, candidate_control_diagnostic = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        policy=ROBUST_INTENT_POLICY,
        replicate=None,
    )
    control_payloads_equal = candidate_control == baseline_control

    baseline_results: list[dict] = []
    candidate_results: list[dict] = []
    baseline_diagnostics: list[dict] = []
    candidate_diagnostics: list[dict] = []
    coordinate_plans = {
        replicate: coordinate_plan_sha256(replicate, len(samples))
        for replicate in REPLICATES
    }
    for replicate in REPLICATES:
        baseline_result, baseline_diagnostic = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
            policy=CANONICAL_INTENT_POLICY,
            replicate=replicate,
        )
        candidate_result, candidate_diagnostic = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
            policy=ROBUST_INTENT_POLICY,
            replicate=replicate,
        )
        baseline_results.append(baseline_result)
        candidate_results.append(candidate_result)
        baseline_diagnostics.append(baseline_diagnostic)
        candidate_diagnostics.append(candidate_diagnostic)

    replay_result, replay_diagnostic = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        policy=ROBUST_INTENT_POLICY,
        replicate=REPLICATES[0],
    )
    replay_payload_equal = replay_result == candidate_results[0]
    replay_trace_equal = (
        replay_diagnostic["message_audit"]["choice_trace_sha256"]
        == candidate_diagnostics[0]["message_audit"]["choice_trace_sha256"]
    )
    replay_diagnostics_equal = {
        key: replay_diagnostic[key]
        for key in (
            "route_health",
            "ranking_health",
            "slate_health",
            "message_audit",
        )
    } == {
        key: candidate_diagnostics[0][key]
        for key in (
            "route_health",
            "ranking_health",
            "slate_health",
            "message_audit",
        )
    }

    baseline_distribution = _distribution(baseline_results)
    candidate_distribution = _distribution(candidate_results)
    mean_delta = _mean_delta(baseline_distribution, candidate_distribution)
    paired = _paired_outcomes(baseline_results, candidate_results)
    candidate_audit = _sum_audits(candidate_diagnostics)
    baseline_audit = _sum_audits(baseline_diagnostics)
    health = {
        "baseline": _sum_health(baseline_diagnostics),
        "candidate": _sum_health(candidate_diagnostics),
    }
    latency = _latency_summary(baseline_diagnostics, candidate_diagnostics)
    raw_mean_delta = {
        key: statistics.fmean(float(result[key]) for result in candidate_results)
        - statistics.fmean(float(result[key]) for result in baseline_results)
        for key in METRIC_KEYS
    }
    raw_candidate_worst = {
        key: (
            max(float(result[key]) for result in candidate_results)
            if key == "mttc"
            else min(float(result[key]) for result in candidate_results)
        )
        for key in METRIC_KEYS
    }
    raw_latency_ratios = [
        (
            float(candidate["respond_latency_ms"]["warm_p95"])
            / float(baseline["respond_latency_ms"]["warm_p95"])
            if float(baseline["respond_latency_ms"]["warm_p95"]) > 0
            else math.inf
        )
        for baseline, candidate in zip(
            baseline_diagnostics,
            candidate_diagnostics,
            strict=True,
        )
    ]
    scenario_hit_loss = min(
        paired["pooled_scenario_hit_delta"].values(),  # type: ignore[union-attr]
        default=0,
    )
    candidate_health = health["candidate"]
    gates = {
        "baseline_control_metrics_match_phase5": True,
        "candidate_control_evaluator_payload_matches_baseline": control_payloads_equal,
        "candidate_parser_state_equivalence_is_one": (
            candidate_audit["state_equivalence_rate"] == 1.0
        ),
        "candidate_rendered_query_equivalence_is_one": (
            candidate_audit["query_equivalence_rate"] == 1.0
        ),
        "critical_equivalence_is_one": (
            candidate_audit["critical_equivalence_rate"] == 1.0
        ),
        "service_integration_equivalence_is_one": (
            candidate_audit["service_integration_equivalence_rate"] == 1.0
        ),
        "unrecognized_canonical_messages_are_zero": (
            candidate_audit["unknown_messages"] == 0
        ),
        "message_adapter_faults_are_zero": (
            candidate_audit["response_exceptions"] == 0
            and candidate_audit["invalid_responses"] == 0
            and candidate_audit["input_processing_exceptions"] == 0
            and candidate_audit["slot_corruptions"] == 0
        ),
        "mean_technical_score_gain_at_least_0_02": (
            raw_mean_delta["recommended_technical_score"] >= 0.02
        ),
        "mean_hit_rate_gain_at_least_0_02": (
            raw_mean_delta["hit_rate_at_10"] >= 0.02
        ),
        "worst_hit_rate_at_least_0_95": (
            raw_candidate_worst["hit_rate_at_10"] >= 0.95
        ),
        "worst_mrr_at_least_0_49": (
            raw_candidate_worst["mrr"] >= 0.49
        ),
        "worst_mttc_at_most_3_5": (
            raw_candidate_worst["mttc"] <= 3.5
        ),
        "worst_technical_score_at_least_0_78": (
            raw_candidate_worst["recommended_technical_score"] >= 0.78
        ),
        "no_pooled_scenario_hit_loss": scenario_hit_loss >= 0,
        "warm_p95_ratio_at_most_1_15": (
            max(raw_latency_ratios) <= 1.15
        ),
        "route_reranker_slate_and_fallback_faults_are_zero": (
            candidate_health["fallback_turns"] == 0
            and candidate_health["ranking"].get("failures", 0) == 0
            and candidate_health["slate"].get("failures", 0) == 0
        ),
        "coordinate_plans_are_deterministic_and_distinct": (
            coordinate_plans
            == {
                replicate: coordinate_plan_sha256(replicate, len(samples))
                for replicate in REPLICATES
            }
            and len(set(coordinate_plans.values())) == len(REPLICATES)
        ),
        "candidate_replay_payload_and_trace_identical": (
            replay_payload_equal and replay_trace_equal and replay_diagnostics_equal
        ),
    }
    expected_preprivacy_gates = _DECISION_GATE_KEYS - {
        "aggregate_privacy_projection_valid",
        "adopt",
    }
    if set(gates) != expected_preprivacy_gates:
        raise RuntimeError("Phase 6 decision-gate schema drifted")
    gates["aggregate_privacy_projection_valid"] = False
    gates["adopt"] = False
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "phase6-seeded-message-robustness-v1",
        "candidate": "phase6-robust-intent-reducer-v1",
        "baseline": "phase5-stagnation-aware-slate-v1",
        "run_configuration": {
            "suite_version": SUITE_VERSION,
            "master_seed": MASTER_SEED,
            "replicate_count": len(REPLICATES),
            "sample_count_per_replicate": len(samples),
            "execution": "sequential",
            "onnx_threads": 1,
            "shared_immutable_backend": True,
            "external_api_calls": 0,
        },
        "control": {
            "baseline": _official_summary(baseline_control),
            "candidate": _official_summary(candidate_control),
            "evaluator_payloads_equal": control_payloads_equal,
            "baseline_health": {
                "route": baseline_control_diagnostic["route_health"],
                "ranking": baseline_control_diagnostic["ranking_health"],
                "slate": baseline_control_diagnostic["slate_health"],
            },
            "candidate_health": {
                "route": candidate_control_diagnostic["route_health"],
                "ranking": candidate_control_diagnostic["ranking_health"],
                "slate": candidate_control_diagnostic["slate_health"],
            },
        },
        "robustness": {
            "baseline_distribution": baseline_distribution,
            "candidate_distribution": candidate_distribution,
            "mean_metric_delta": mean_delta,
            "baseline_scenario_distribution": _scenario_distribution(baseline_results),
            "candidate_scenario_distribution": _scenario_distribution(candidate_results),
            "paired_outcomes": paired,
        },
        "semantic_equivalence": {
            "baseline": baseline_audit,
            "candidate": candidate_audit,
        },
        "health": health,
        "latency": latency,
        "determinism": {
            **protocol_goldens,
            "candidate_replay_evaluator_payload_equal": replay_payload_equal,
            "candidate_replay_choice_trace_equal": replay_trace_equal,
            "candidate_replay_diagnostics_equal": replay_diagnostics_equal,
        },
        "decision_gate": gates,
        "privacy": {
            "contains_queries_or_messages": False,
            "contains_profiles": False,
            "contains_product_or_sample_ids": False,
            "contains_session_or_turn_rows": False,
            "contains_per_replicate_rows": False,
            "choice_trace_hashes_raw_messages": False,
        },
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "catalog_sha256": _sha256(catalog),
            "dataset_sha256": _sha256(dataset),
            "frozen_input_sha256": frozen_input_sha256,
            "source_sha256": {
                relative: _sha256(repository_root / relative)
                for relative in SOURCE_PATHS
            },
        },
    }
    _validate_aggregate_privacy(payload)
    gates["aggregate_privacy_projection_valid"] = True
    gates["adopt"] = all(
        value for key, value in gates.items() if key != "adopt"
    )
    _validate_aggregate_privacy(payload)
    return payload


def _validate_output(output: Path, catalog: Path, dataset: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    resolved_output = output.resolve()
    if any(
        resolved_output.is_relative_to(repository_root / directory)
        for directory in ("docs", "benchmarks")
    ):
        raise ValueError(
            "the experiment runner cannot write append-only publication paths"
        )
    protected = {
        catalog.resolve(),
        dataset.resolve(),
        *(repository_root / relative for relative in SOURCE_PATHS),
    }
    if resolved_output in protected:
        raise ValueError("output must not overwrite an input or source file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sequential Phase 6 message-robustness ablation"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    catalog = Path(args.catalog).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    _validate_output(output, catalog, dataset)
    _write_json_atomic(output, run_message_robustness(catalog, dataset))


if __name__ == "__main__":
    main()
