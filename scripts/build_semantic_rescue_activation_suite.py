"""Build a deterministic, public-target-disjoint semantic-rescue activation suite.

The suite is deliberately candidate-conditioned: it contains only catalog
category/constraint groups for which the frozen rescue reaches ``APPLIED``.
It can prove reachability and invariants, but it cannot support quality or
promotion claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from conversational_search.intent import (
    IntentState,
    Requirement,
    apply_user_message,
    classify_requirement,
    render_dense_query,
    render_lexical_query,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval import (
    SHARED_DENSE_TERMS_RESCUE_POLICY,
    SemanticLexicalRescueStatus,
    SemanticLexicalRetrievalResult,
)
from conversational_search.service import ConversationalSearchAgent
from evaluator.local_evaluator import load_jsonl
from scripts.run_fusion_ablations import _sha256


SCHEMA_VERSION = 1
SUITE_ID = "phase16b-semantic-rescue-activation-v1"
SELECTION_SALT = "phase16b-semantic-rescue-activation-target-v1"
MAX_CASES = 64
MIN_GROUP_SUPPORT = 3
MAX_SUPPORT_IDS = 200
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _digest(*values: str) -> bytes:
    return hashlib.sha256("\0".join(values).encode("utf-8")).digest()


def _public_target_ids(rows: Sequence[Mapping[str, object]]) -> frozenset[str]:
    targets: set[str] = set()
    for row in rows:
        ground_truth = row.get("ground_truth")
        value = (
            ground_truth.get("parent_asin")
            if isinstance(ground_truth, dict)
            else None
        )
        if not isinstance(value, str) or not value:
            raise ValueError("public row has no target product")
        targets.add(value)
    return frozenset(targets)


def select_unique_target(
    support_ids: Sequence[str],
    *,
    public_targets: frozenset[str],
    selected_targets: frozenset[str],
    category: str,
    constraint: str,
) -> str | None:
    """Choose a target without consulting either arm's rank or hit outcome."""

    eligible = tuple(
        parent_asin
        for parent_asin in dict.fromkeys(support_ids)
        if parent_asin not in public_targets
        and parent_asin not in selected_targets
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda parent_asin: _digest(
            SELECTION_SALT,
            category,
            constraint,
            parent_asin,
        ),
    )


def _candidate_groups(backend: object) -> tuple[tuple[str, str, int], ...]:
    connection = getattr(backend, "_connection", None)
    if connection is None:
        raise RuntimeError("protocol catalog connection is unavailable")
    rows = connection.execute(
        """
        WITH values_by_product AS (
          SELECT parent_asin, coarse_category, hard_0 AS value
          FROM protocol_products WHERE hard_0 IS NOT NULL
          UNION ALL
          SELECT parent_asin, coarse_category, hard_1
          FROM protocol_products WHERE hard_1 IS NOT NULL
          UNION ALL
          SELECT parent_asin, coarse_category, soft_0
          FROM protocol_products WHERE soft_0 IS NOT NULL
          UNION ALL
          SELECT parent_asin, coarse_category, soft_1
          FROM protocol_products WHERE soft_1 IS NOT NULL
        )
        SELECT coarse_category, value, COUNT(DISTINCT parent_asin)
        FROM values_by_product
        WHERE coarse_category <> '' AND value <> ''
        GROUP BY coarse_category, value
        HAVING COUNT(DISTINCT parent_asin) >= ?
        """,
        (MIN_GROUP_SUPPORT,),
    ).fetchall()
    groups = tuple(
        (str(category), str(value), int(count))
        for category, value, count in rows
    )
    return tuple(
        sorted(
            groups,
            key=lambda item: _digest(SELECTION_SALT, item[0], item[1]),
        )
    )


def _state(category: str, constraint: str) -> IntentState:
    message = (
        f"I'm looking for {category}. "
        f"A key requirement is: {constraint}."
    )
    parsed = apply_user_message(IntentState(), message, 1)
    expected = IntentState(
        category=category,
        requirements=(
            Requirement(
                constraint,
                "initial_explicit",
                1,
                classify_requirement(constraint),
            ),
        ),
        last_turn=1,
    )
    if parsed != expected:
        raise RuntimeError("activation message does not round-trip through intent")
    return parsed


def build_activation_suite(
    backend: object,
    public_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict], dict[str, object]]:
    public_targets = _public_target_ids(public_rows)
    selected_targets: set[str] = set()
    suite: list[dict] = []
    statuses: Counter[str] = Counter()
    inadequate_groups = 0
    for category, constraint, _ in _candidate_groups(backend):
        support = tuple(
            backend.protocol_exact_candidates(
                category,
                (constraint,),
                limit=MAX_SUPPORT_IDS,
            )
        )
        target = select_unique_target(
            support,
            public_targets=public_targets,
            selected_targets=frozenset(selected_targets),
            category=category,
            constraint=constraint,
        )
        if target is None:
            continue
        state = _state(category, constraint)
        lexical = render_lexical_query(state)
        base_bm25_ids = tuple(
            backend._sanitize_route(backend._bm25(lexical))
        )
        if backend._bm25_has_credible_structural_support(
            base_bm25_ids,
            frozenset(support),
            lexical,
            category,
        ):
            statuses[SemanticLexicalRescueStatus.NOT_NEEDED.value] += 1
            continue
        inadequate_groups += 1
        result = backend.search_with_trace(
            render_dense_query(state),
            lexical,
            top_k=10,
            use_dense=False,
            dense_rescue_on_bm25_failure=False,
            bm25_only_support_ids=support,
            semantic_lexical_rescue_policy=(
                SHARED_DENSE_TERMS_RESCUE_POLICY
            ),
            semantic_rescue_category=category,
        )
        if not isinstance(result, SemanticLexicalRetrievalResult):
            raise RuntimeError("activation probe returned no semantic trace")
        status = result.semantic_trace.status
        statuses[status.value] += 1
        if status is not SemanticLexicalRescueStatus.APPLIED:
            continue
        selected_targets.add(target)
        case_digest = _digest(SELECTION_SALT, category, constraint, target).hex()
        suite.append(
            {
                "sample_id": f"activation_{case_digest[:20]}",
                "scenario_type": "buying",
                "user_profile": {},
                "ground_truth": {"parent_asin": target},
                "intent_card": {
                    "target_category": category,
                    "hard_constraints": [constraint],
                    "soft_preferences": [],
                },
                "behavior": {"scenario_type": "buying"},
            }
        )
        if len(suite) >= MAX_CASES:
            break
    if not suite:
        raise RuntimeError("no target-disjoint semantic activation case exists")
    validate_activation_suite(suite, public_targets=public_targets)
    return suite, {
        "suite_id": SUITE_ID,
        "selection_is_candidate_conditioned": True,
        "quality_or_promotion_claim_allowed": False,
        "case_count": len(suite),
        "unique_target_count": len(selected_targets),
        "public_target_overlap": len(selected_targets.intersection(public_targets)),
        "inadequate_groups_examined": inadequate_groups,
        "semantic_status_counts": dict(sorted(statuses.items())),
        "maximum_cases": MAX_CASES,
        "minimum_group_support": MIN_GROUP_SUPPORT,
        "selection_salt_sha256": hashlib.sha256(
            SELECTION_SALT.encode("utf-8")
        ).hexdigest(),
    }


def validate_activation_suite(
    rows: Sequence[Mapping[str, object]],
    *,
    public_targets: frozenset[str],
) -> None:
    if not rows or len(rows) > MAX_CASES:
        raise ValueError("activation suite size is invalid")
    sample_ids: set[str] = set()
    targets: set[str] = set()
    for row in rows:
        if set(row) != {
            "sample_id",
            "scenario_type",
            "user_profile",
            "ground_truth",
            "intent_card",
            "behavior",
        }:
            raise ValueError("activation row schema drifted")
        sample_id = row.get("sample_id")
        ground_truth = row.get("ground_truth")
        target = (
            ground_truth.get("parent_asin")
            if isinstance(ground_truth, dict)
            else None
        )
        card = row.get("intent_card")
        if (
            not isinstance(sample_id, str)
            or not sample_id.startswith("activation_")
            or sample_id in sample_ids
            or row.get("scenario_type") != "buying"
            or row.get("user_profile") != {}
            or not isinstance(target, str)
            or not target
            or target in targets
            or target in public_targets
            or not isinstance(card, dict)
            or not isinstance(card.get("target_category"), str)
            or not isinstance(card.get("hard_constraints"), list)
            or len(card["hard_constraints"]) != 1
            or card.get("soft_preferences") != []
        ):
            raise ValueError("activation row is invalid or not target-disjoint")
        sample_ids.add(sample_id)
        targets.add(target)


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the target-disjoint semantic rescue activation suite"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--output")
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the suite; without this flag the command is a dry run",
    )
    arguments = parser.parse_args()
    if arguments.write and not arguments.output:
        parser.error("--write requires --output")
    catalog = Path(arguments.catalog).resolve()
    public = Path(arguments.public).resolve()
    bootstrap = ConversationalSearchAgent(
        catalog,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    suite, diagnostics = build_activation_suite(
        bootstrap.retrieval_backend,
        load_jsonl(public),
    )
    payload = _jsonl_bytes(suite)
    report = {
        "schema_version": SCHEMA_VERSION,
        **diagnostics,
        "catalog_sha256": _sha256(catalog),
        "public_sha256": _sha256(public),
        "suite_sha256": hashlib.sha256(payload).hexdigest(),
        "write_requested": bool(arguments.write),
    }
    if arguments.write:
        output = Path(arguments.output).resolve()
        if output in {catalog, public}:
            raise ValueError("activation output must not overwrite an input")
        _write_exclusive(output, payload)
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
