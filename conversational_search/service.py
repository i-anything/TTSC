"""Stateful conversational-search core assembled from swappable policies.

``ConversationalSearchAgent`` owns the complete session pipeline: intent
reduction, retrieval-route planning, hybrid BM25+dense retrieval with
weighted RRF fusion, Stage-A and exact-evidence reranking, full-catalog
protocol fusion, exposure gating, and novelty-aware slate selection.  Every
stage is selected through an injected policy constant, so bounded
alternatives stay available for reproducible ablation while one released
configuration serves the competition adapter in ``starter.agent``.

Ranked results are cached and reused only while the exact ranking-relevant
dependency digest is unchanged, and every failure path fails open to the
deterministic hybrid result described in ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from conversational_search.decision_policy import (
    EXPECTED_UTILITY_DECISION_POLICY,
    PROTECTED_DECISION_POLICY,
    PROTOCOL_DECISION_POLICIES,
    PROTOCOL_UTILITY_DECISION_POLICY,
    DecisionPolicy,
)
from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
    PROTOCOL_POSTERIOR_EXPOSURE_POLICY,
    PROTOCOL_REPLY_TREE_EXPOSURE_POLICY,
    TOP3_STRUCTURAL_EXPOSURE_POLICY,
    EvidenceExposureDecision,
    EvidenceExposurePolicy,
    EvidenceExposureStatus,
)
from conversational_search.exact_evidence import (
    DISABLED_SEMANTIC_TIEBREAK_POLICY,
    SemanticTieBreakPolicy,
    SemanticTieBreakStatus,
)
from conversational_search.field_semantic import (
    DISABLED_FIELD_SEMANTIC_POLICY,
    MAX_FIELD_SEMANTIC_CANDIDATES,
    MAX_FIELD_SEMANTIC_REQUIREMENTS,
    FieldSemanticPolicy,
    FieldSemanticStatus,
)
from conversational_search.intent import (
    LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
    IntentReductionStatus,
    IntentState,
    IntentParsingPolicy,
    apply_user_message,
    apply_user_message_with_trace,
    record_question,
    render_dense_query,
    render_lexical_query,
    render_requirement_probe_candidates,
)
from conversational_search.local_intent import (
    LlamaCppStructuredIntentParser,
    LocalIntentTrigger,
    StructuredIntentParseResult,
    apply_structured_intent_delta,
    local_intent_trigger,
)
from conversational_search.orchestration import (
    BackendSnapshotToken,
    DEFAULT_PROFILE_DEPENDENCY_DIGEST,
    DEFAULT_RANKING_CACHE_CAPACITY,
    EXACT_RANKING_CACHE_CAPABILITY,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    OrchestrationPlanner,
    OrchestrationPolicy,
    QueryAction,
)
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    NEUTRAL_PROFILE_PRIOR,
    PROFILE_THEME_MASK_BYTES,
    ProfilePolicy,
    ProfilePrior,
    parse_profile_prior,
)
from conversational_search.protocol_index import (
    DISABLED_PROTOCOL_CATALOG_POLICY,
    DISABLED_PROTOCOL_REFUTATION_POLICY,
    ELIGIBLE_CONTINUATION_REFUTATION_POLICY,
    FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
    ProtocolCatalogPolicy,
    ProtocolRefutationPolicy,
    ProtocolResolution,
    fuse_protocol_candidates,
    resolve_protocol_transcript,
)
from conversational_search.questions import (
    CONSERVATIVE_EARLY_OTHER_POLICY,
    QUESTION_TEXT,
    QuestionPolicy,
)
from conversational_search.ranking import (
    MAX_CLAUSES,
    Bm25RescueRankingResult,
    Bm25RescueStatus,
    CandidateDocument,
    STAGE_A_RANKING_POLICY,
    ProfileRankingResult,
    ProfileResidualStatus,
    RankingPolicy,
    RankingResult,
    RankingTrace,
    RouteRedundancyRankingResult,
    RouteRedundancyStatus,
    rerank_stage_a,
    rerank_stage_a_with_profile,
    rerank_stage_a_with_profile_and_bm25_rescue,
    rerank_stage_a_with_profile_and_route_redundancy,
)
from conversational_search.retrieval import (
    DISABLED_REQUIREMENT_PROBE_POLICY,
    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
    MAX_CANDIDATE_DOCUMENTS,
    MAX_REQUIREMENT_PROBES,
    MAX_SEMANTIC_EXPANSION_TERMS,
    FIELD_SEMANTIC_CAPABILITY,
    REQUIREMENT_PROBE_CAPABILITY,
    PROTOCOL_EVIDENCE_CAPABILITY,
    ROUTE_LIMIT,
    HybridRetriever,
    RequirementProbeRetrievalResult,
    RequirementProbePolicy,
    RequirementProbeTrace,
    RetrievalResult,
    RetrievalTrace,
    SemanticLexicalRescuePolicy,
    SemanticLexicalRescueStatus,
    SemanticLexicalRescueTrace,
    SemanticLexicalRetrievalResult,
)
from conversational_search.retrieval_routing import (
    ALWAYS_HYBRID_RETRIEVAL_ROUTING_POLICY,
    SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
    RetrievalRouteMode,
    RetrievalRoutePlan,
    RetrievalRouteReason,
    RetrievalRoutingPolicy,
    plan_retrieval_route,
)
from conversational_search.slates import (
    INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    MAX_SLATE_CANDIDATES,
    REPEAT_TOP_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
    IntentEpochSlateSelection,
    IntentEpochSlateStatus,
    SlatePolicy,
    SlateSelection,
    SlateState,
    SlateTrace,
    ranking_signature,
    select_slate,
    select_slate_with_intent_epoch_novelty,
)
from conversational_search.strategy import (
    COMPLETENESS_ADAPTIVE_RRF_POLICY,
    FusionPolicy,
    RouteWeights,
    intent_completeness,
)

if TYPE_CHECKING:
    from conversational_search.exact_evidence import ExactEvidenceResult
    from conversational_search.protocol import (
        ObservedProtocolEvent,
        ProductProtocolEvidence,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ASSETS = (
    REPOSITORY_ROOT / "assets" / "bge-small-en-v1.5-int8"
)
DEFAULT_DENSE_INDEX = (
    REPOSITORY_ROOT
    / "assets"
    / "search-index-bge-small-en-v1.5-v2"
)
DEFAULT_LOCAL_INTENT_MODEL = (
    REPOSITORY_ROOT
    / "assets"
    / "qwen3-1.7b-intent"
    / "Qwen3-1.7B-Q4_K_M.gguf"
)
CACHEABLE_ROUTE_STATUSES = frozenset({"ok", "empty", "skipped"})
DENSE_ALWAYS_RETRIEVAL_POLICY = "dense-always-v1"
PROTOCOL_CONDITIONAL_DENSE_RETRIEVAL_POLICY = (
    "protocol-bm25-first-structural-gate-v2"
)
REQUIREMENT_PROBE_STATUSES = (
    "disabled",
    "no_eligible",
    "capacity",
    "ok",
    "empty",
    "no_additions",
    "unavailable",
    "error",
)
CACHEABLE_REQUIREMENT_PROBE_STATUSES = frozenset(
    {"disabled", "no_eligible", "capacity", "ok", "empty", "no_additions"}
)
STAGE_A_RANKING_POLICIES = frozenset(
    {
        RankingPolicy.STAGE_A,
        RankingPolicy.COMPLETENESS_BM25_RESCUE,
        RankingPolicy.ROUTE_REDUNDANCY_CORRECTED,
        RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE,
        RankingPolicy.IMPORTANCE_AWARE_SATISFACTION,
    }
)
PROTOCOL_DECISION_OUTCOMES = (
    "applied",
    "unsupported_or_disabled",
    "capability_unavailable",
    "candidate_or_evidence_error",
    "fail_open_evidence",
    "fail_open_no_candidates",
    "fail_open_no_support",
    "fail_open_validation",
)
PROTOCOL_QUESTION_ACTIONS = (*QUESTION_TEXT, "none")
EXACT_EVIDENCE_OUTCOMES = (
    "applied_reordered",
    "applied_unchanged",
    "zero_support_fail_open",
    "capability_unavailable",
    "evidence_error",
    "validation_error",
)
IMPORTANCE_SATISFACTION_OUTCOMES = (
    "applied_reordered",
    "applied_unchanged",
    "no_requirements",
    "capability_unavailable",
    "evidence_error",
    "validation_error",
)
SEMANTIC_RESCUE_CACHEABLE_STATUSES = frozenset(
    {
        SemanticLexicalRescueStatus.NOT_NEEDED,
        SemanticLexicalRescueStatus.APPLIED,
    }
)
@dataclass(frozen=True, slots=True)
class _ExactRankingContext:
    base_ranked_ids: tuple[str, ...]
    evidence: tuple[object, ...]
    result: object | None
    output_ranked_ids: tuple[str, ...]
    cacheable: bool


@dataclass(frozen=True, slots=True)
class _ExpectedExactReplay:
    dependency_digest: bytes
    backend_snapshot_token: BackendSnapshotToken
    base_ranked_ids: tuple[str, ...]
    output_ranked_ids: tuple[str, ...]


def _profile_session_key(session_id: str) -> bytes:
    """Hash profile-state keys so the profile store retains no raw session ID."""

    return hashlib.sha256(session_id.encode("utf-8")).digest()


def _validate_dense_pair(encoder: object, dense_index: object) -> None:
    metadata = getattr(encoder, "metadata", None)
    manifest = getattr(dense_index, "manifest", None)
    if metadata is None or not isinstance(manifest, dict):
        raise ValueError("dense encoder and index do not expose compatibility metadata")
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("dense index manifest does not contain model metadata")

    index_asset_identity = model.get("asset_identity_sha256")
    runtime_asset_identity = getattr(metadata, "asset_identity_sha256", "")
    if index_asset_identity:
        if index_asset_identity != runtime_asset_identity:
            raise ValueError(
                "dense runtime mismatch for asset_identity_sha256: "
                f"{runtime_asset_identity!r} != {index_asset_identity!r}"
            )
    elif model.get("asset_manifest_sha256") != getattr(
        metadata, "asset_manifest_sha256", None
    ):
        # Historical schema-2 BGE artifacts predate the semantic asset
        # identity, so retain their exact-manifest compatibility rule.
        raise ValueError(
            "dense runtime mismatch for asset_manifest_sha256: "
            f"{getattr(metadata, 'asset_manifest_sha256', None)!r} != "
            f"{model.get('asset_manifest_sha256')!r}"
        )

    keys = (
        "model_id",
        "revision",
        "model_sha256",
        "source_model_sha256",
        "tokenizer_sha256",
        "dimension",
        "max_sequence_length",
        "pooling",
        "normalization",
        "document_prefix",
        "query_prefix",
        "provider",
        "compute_dtype",
    )
    for key in keys:
        runtime_value = getattr(metadata, key, None)
        if model.get(key) != runtime_value:
            raise ValueError(
                f"dense runtime mismatch for {key}: "
                f"{runtime_value!r} != {model.get(key)!r}"
            )

    dimension = getattr(dense_index, "dimension", None)
    if dimension != metadata.dimension:
        raise ValueError(
            f"dense index dimension {dimension!r} != encoder dimension "
            f"{metadata.dimension!r}"
        )


def _validate_catalog_pair(catalog_path: str | Path, dense_index: object) -> None:
    manifest = getattr(dense_index, "manifest", None)
    if not isinstance(manifest, dict):
        raise ValueError("dense index does not expose a manifest")
    catalog = manifest.get("catalog")
    if not isinstance(catalog, dict):
        raise ValueError("dense index manifest does not contain catalog metadata")
    expected_rows = catalog.get("rows")
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows <= 0
    ):
        raise ValueError("dense index manifest has an invalid catalog row count")
    if getattr(dense_index, "row_count", None) != expected_rows:
        raise ValueError("dense index row count does not match its catalog metadata")

    expected_sha256 = catalog.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("dense index manifest has an invalid catalog checksum")
    from preprocessing.encoder import sha256_file

    if sha256_file(catalog_path) != expected_sha256:
        raise ValueError("runtime catalog checksum does not match the dense index")


def _load_dense_runtime(
    catalog_path: str | Path,
    model_assets: str | Path,
    dense_index_path: str | Path,
) -> tuple[object, object]:
    # Delayed imports keep BM25-only tests and fallback environments lightweight.
    from preprocessing.encoder import OnnxTextEncoder
    from starter.dense import ShardedDenseIndex

    encoder = OnnxTextEncoder(model_assets, threads=1)
    dense_index = ShardedDenseIndex(dense_index_path)
    _validate_dense_pair(encoder, dense_index)
    _validate_catalog_pair(catalog_path, dense_index)
    return encoder, dense_index


class ConversationalSearchAgent:
    """Stateful orchestration core behind the competition Agent adapter."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        retriever: object | None = None,
        question_policy: QuestionPolicy = CONSERVATIVE_EARLY_OTHER_POLICY,
        fusion_policy: FusionPolicy = COMPLETENESS_ADAPTIVE_RRF_POLICY,
        retrieval_routing_policy: RetrievalRoutingPolicy = (
            ALWAYS_HYBRID_RETRIEVAL_ROUTING_POLICY
        ),
        ranking_policy: RankingPolicy = STAGE_A_RANKING_POLICY,
        profile_policy: ProfilePolicy = BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy: SlatePolicy = STAGNATION_AWARE_SLATE_POLICY,
        intent_policy: IntentParsingPolicy = ROBUST_INTENT_POLICY,
        decision_policy: DecisionPolicy = PROTECTED_DECISION_POLICY,
        requirement_probe_policy: RequirementProbePolicy = (
            DISABLED_REQUIREMENT_PROBE_POLICY
        ),
        semantic_lexical_rescue_policy: SemanticLexicalRescuePolicy = (
            DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        ),
        semantic_tiebreak_policy: SemanticTieBreakPolicy = (
            DISABLED_SEMANTIC_TIEBREAK_POLICY
        ),
        field_semantic_policy: FieldSemanticPolicy = (
            DISABLED_FIELD_SEMANTIC_POLICY
        ),
        evidence_exposure_policy: EvidenceExposurePolicy = (
            DISABLED_EVIDENCE_EXPOSURE_POLICY
        ),
        protocol_catalog_policy: ProtocolCatalogPolicy = (
            DISABLED_PROTOCOL_CATALOG_POLICY
        ),
        protocol_refutation_policy: ProtocolRefutationPolicy = (
            DISABLED_PROTOCOL_REFUTATION_POLICY
        ),
        orchestration_policy: OrchestrationPolicy = (
            EXACT_RANKING_REUSE_ORCHESTRATION_POLICY
        ),
        ranking_cache_capacity: int = DEFAULT_RANKING_CACHE_CAPACITY,
        model_assets: str | Path = DEFAULT_MODEL_ASSETS,
        dense_index_path: str | Path = DEFAULT_DENSE_INDEX,
        local_intent_parser: object | None = None,
        local_intent_model_path: str | Path | None = None,
    ) -> None:
        if not isinstance(question_policy, QuestionPolicy):
            raise TypeError("question_policy must be a QuestionPolicy")
        if not isinstance(fusion_policy, FusionPolicy):
            raise TypeError("fusion_policy must be a FusionPolicy")
        if not isinstance(retrieval_routing_policy, RetrievalRoutingPolicy):
            raise TypeError(
                "retrieval_routing_policy must be a RetrievalRoutingPolicy"
            )
        if not isinstance(ranking_policy, RankingPolicy):
            raise TypeError("ranking_policy must be a RankingPolicy")
        if not isinstance(profile_policy, ProfilePolicy):
            raise TypeError("profile_policy must be a ProfilePolicy")
        if not isinstance(slate_policy, SlatePolicy):
            raise TypeError("slate_policy must be a SlatePolicy")
        if not isinstance(intent_policy, IntentParsingPolicy):
            raise TypeError("intent_policy must be an IntentParsingPolicy")
        if not isinstance(decision_policy, DecisionPolicy):
            raise TypeError("decision_policy must be a DecisionPolicy")
        if not isinstance(requirement_probe_policy, RequirementProbePolicy):
            raise TypeError(
                "requirement_probe_policy must be a RequirementProbePolicy"
            )
        if not isinstance(
            semantic_lexical_rescue_policy,
            SemanticLexicalRescuePolicy,
        ):
            raise TypeError(
                "semantic_lexical_rescue_policy must be a "
                "SemanticLexicalRescuePolicy"
            )
        if not isinstance(semantic_tiebreak_policy, SemanticTieBreakPolicy):
            raise TypeError(
                "semantic_tiebreak_policy must be a SemanticTieBreakPolicy"
            )
        if not isinstance(field_semantic_policy, FieldSemanticPolicy):
            raise TypeError("field_semantic_policy must be a FieldSemanticPolicy")
        if not isinstance(evidence_exposure_policy, EvidenceExposurePolicy):
            raise TypeError(
                "evidence_exposure_policy must be an EvidenceExposurePolicy"
            )
        if not isinstance(protocol_catalog_policy, ProtocolCatalogPolicy):
            raise TypeError("protocol_catalog_policy must be a ProtocolCatalogPolicy")
        if not isinstance(protocol_refutation_policy, ProtocolRefutationPolicy):
            raise TypeError(
                "protocol_refutation_policy must be a ProtocolRefutationPolicy"
            )
        if (
            protocol_refutation_policy
            is not DISABLED_PROTOCOL_REFUTATION_POLICY
            and protocol_catalog_policy
            is not FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
        ):
            raise ValueError(
                "protocol refutation requires full transcript catalog resolution"
            )
        if (
            evidence_exposure_policy
            in {
                PROTOCOL_POSTERIOR_EXPOSURE_POLICY,
                PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
                PROTOCOL_REPLY_TREE_EXPOSURE_POLICY,
            }
            and protocol_catalog_policy
            is not FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
        ):
            raise ValueError(
                "protocol posterior exposure requires full catalog resolution"
            )
        if (
            evidence_exposure_policy
            in {
                PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
                PROTOCOL_REPLY_TREE_EXPOSURE_POLICY,
            }
            and protocol_refutation_policy
            is not ELIGIBLE_CONTINUATION_REFUTATION_POLICY
        ):
            raise ValueError(
                "metric-aware protocol enumeration requires continuation "
                "refutation"
            )
        if not isinstance(orchestration_policy, OrchestrationPolicy):
            raise TypeError("orchestration_policy must be an OrchestrationPolicy")
        if (
            requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY
            and ranking_policy is not RankingPolicy.STAGE_A
        ):
            raise ValueError(
                "requirement probes are supported only with the Stage-A policy"
            )
        if (
            decision_policy in PROTOCOL_DECISION_POLICIES
            and requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY
        ):
            raise ValueError(
                "protocol decisions and requirement probes are separate ablations"
            )
        if (
            semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
            and requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY
        ):
            raise ValueError(
                "semantic rescue and requirement probes are separate ablations"
            )
        if (
            retrieval_routing_policy
            is SMART_HYBRID_RETRIEVAL_ROUTING_POLICY
            and semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        ):
            raise ValueError(
                "smart hybrid routing and semantic lexical rescue are "
                "separate policies"
            )
        if (
            semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
            or semantic_tiebreak_policy
            is not DISABLED_SEMANTIC_TIEBREAK_POLICY
            or evidence_exposure_policy
            is not DISABLED_EVIDENCE_EXPOSURE_POLICY
        ):
            if ranking_policy is not RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE:
                raise ValueError(
                    "semantic policies and evidence exposure require "
                    "lexicographic exact evidence ranking"
                )
        if field_semantic_policy is not DISABLED_FIELD_SEMANTIC_POLICY:
            if ranking_policy is not RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE:
                raise ValueError(
                    "field-semantic scoring requires lexicographic exact "
                    "evidence ranking"
                )
            if decision_policy is not PROTECTED_DECISION_POLICY:
                raise ValueError(
                    "field-semantic scoring is isolated from protocol "
                    "decision planners"
                )
            if decision_policy is not PROTECTED_DECISION_POLICY:
                raise ValueError(
                    "semantic policies and evidence exposure are isolated from "
                    "protocol decision planners"
                )
        if (
            decision_policy is PROTOCOL_UTILITY_DECISION_POLICY
            and ranking_policy is RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE
        ):
            raise ValueError(
                "protocol utility and lexicographic exact ranking require a "
                "unified planner integration"
            )
        if decision_policy is EXPECTED_UTILITY_DECISION_POLICY:
            if ranking_policy is not RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE:
                raise ValueError(
                    "expected utility requires lexicographic exact evidence ranking"
                )
            if slate_policy is not INTENT_EPOCH_NOVELTY_SLATE_POLICY:
                raise ValueError(
                    "expected utility requires intent-epoch novelty selection"
                )
            if orchestration_policy is not EXACT_RANKING_REUSE_ORCHESTRATION_POLICY:
                raise ValueError(
                    "expected utility requires exact ranking reuse orchestration"
                )
        if (
            ranking_policy is RankingPolicy.IMPORTANCE_AWARE_SATISFACTION
            and decision_policy is not PROTECTED_DECISION_POLICY
        ):
            raise ValueError(
                "importance-aware satisfaction is an isolated reranker ablation"
            )
        if local_intent_parser is not None and not callable(
            getattr(local_intent_parser, "parse", None)
        ):
            raise TypeError("local_intent_parser must expose parse(...) or be None")
        self.local_intent_initialization_error: str | None = None
        if local_intent_parser is None and local_intent_model_path is not None:
            local_model_path = Path(local_intent_model_path)
            if local_model_path.is_file():
                try:
                    local_intent_parser = LlamaCppStructuredIntentParser(
                        local_model_path
                    )
                except (ImportError, OSError, RuntimeError, ValueError) as error:
                    self.local_intent_initialization_error = (
                        f"{type(error).__name__}: {error}"
                    )
        self._local_intent_parser = local_intent_parser
        self._local_intent_attempts = 0
        self._local_intent_applied = 0
        self._local_intent_no_delta = 0
        self._local_intent_failures = 0
        self._local_intent_free_text_attempts = 0
        self._local_intent_complex_attempts = 0
        self._local_intent_prompt_tokens = 0
        self._local_intent_completion_tokens = 0
        self._local_intent_semantic_sessions: set[str] = set()
        self.dense_initialization_error: str | None = None
        if retriever is None:
            try:
                encoder, dense_index = _load_dense_runtime(
                    catalog_path,
                    model_assets,
                    dense_index_path,
                )
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                self.dense_initialization_error = f"{type(error).__name__}: {error}"
                encoder = None
                dense_index = None
            retriever = HybridRetriever(
                catalog_path,
                encoder=encoder,
                dense_index=dense_index,
                protocol_evidence=(
                    decision_policy in PROTOCOL_DECISION_POLICIES
                    or protocol_catalog_policy
                    is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
                    or retrieval_routing_policy
                    is SMART_HYBRID_RETRIEVAL_ROUTING_POLICY
                    or field_semantic_policy
                    is not DISABLED_FIELD_SEMANTIC_POLICY
                    or ranking_policy
                    in {
                        RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE,
                        RankingPolicy.IMPORTANCE_AWARE_SATISFACTION,
                    }
                ),
            )
        self._retriever = retriever
        self._protocol_catalog_policy = protocol_catalog_policy
        self._protocol_refutation_policy = protocol_refutation_policy
        self._question_policy = question_policy
        self._fusion_policy = fusion_policy
        self._retrieval_routing_policy = retrieval_routing_policy
        self._retrieval_route_reason_counts = [
            0
        ] * len(RetrievalRouteReason)
        self._retrieval_route_outcome_counts = [0] * 10
        self._ranking_policy = ranking_policy
        self._semantic_tiebreak_policy = semantic_tiebreak_policy
        self._field_semantic_policy = field_semantic_policy
        self._profile_policy = profile_policy
        self._slate_policy = slate_policy
        self._intent_policy = intent_policy
        if (
            semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        ):
            self._semantic_lexical_rescue_policy = (
                semantic_lexical_rescue_policy
            )
        if evidence_exposure_policy is not DISABLED_EVIDENCE_EXPOSURE_POLICY:
            self._evidence_exposure_policy = evidence_exposure_policy
        if ranking_policy is RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE:
            self._exact_evidence_counts = [0] * 9
        if semantic_tiebreak_policy is not DISABLED_SEMANTIC_TIEBREAK_POLICY:
            self._semantic_tiebreak_counts = [
                0
            ] * (len(SemanticTieBreakStatus) + 2)
        if field_semantic_policy is not DISABLED_FIELD_SEMANTIC_POLICY:
            self._field_semantic_counts = [0] * 7
        if ranking_policy is RankingPolicy.IMPORTANCE_AWARE_SATISFACTION:
            self._importance_satisfaction_counts = [0] * 15
        if (
            decision_policy in PROTOCOL_DECISION_POLICIES
            or protocol_catalog_policy
            is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
        ):
            self._decision_policy = decision_policy
            self._protocol_consistency: dict[str, bool] = {}
            self._protocol_events: dict[
                str,
                tuple[object, ...],
            ] = {}
            self._protocol_override_pending: dict[str, bool] = {}
            self._protocol_shown_ids: dict[str, tuple[str, ...]] = {}
            self._protocol_decision_counts = [0] * 9
            self._protocol_question_counts = [0] * len(PROTOCOL_QUESTION_ACTIONS)
            self._protocol_width_counts = [0] * 11
            self._protocol_requested_total = 0
            self._protocol_presented_total = 0
            self._protocol_action_traces: dict[str, dict[str, object]] = {}
        if protocol_catalog_policy is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY:
            self._protocol_refuted_ids: dict[str, tuple[str, ...]] = {}
            self._protocol_pending_ids: dict[str, tuple[str, ...]] = {}
            self._protocol_pending_refutable: dict[str, bool] = {}
        if decision_policy is PROTOCOL_UTILITY_DECISION_POLICY:
            self._protocol_hybrid_sticky: dict[str, bool] = {}
            self._protocol_route_modes: dict[str, str] = {}
        if decision_policy is EXPECTED_UTILITY_DECISION_POLICY:
            self._expected_exact_replays: dict[str, _ExpectedExactReplay] = {}
        if requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY:
            self._requirement_probe_policy = requirement_probe_policy
            self._requirement_probe_counts = [0] * 12
        if (
            semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        ):
            self._semantic_rescue_counts = [
                0
            ] * (len(SemanticLexicalRescueStatus) + 2)
        if evidence_exposure_policy is not DISABLED_EVIDENCE_EXPOSURE_POLICY:
            self._evidence_exposure_counts = [
                0
            ] * (len(EvidenceExposureStatus) + 3)
        self._orchestrator = OrchestrationPlanner(
            orchestration_policy,
            capacity=ranking_cache_capacity,
        )
        self._reranking_attempts = 0
        self._reranking_successes = 0
        self._reranking_failures = 0
        self._reranking_unavailable_skips = 0
        self._bm25_rescue_attempts = 0
        self._bm25_rescue_zero_completeness = 0
        self._bm25_rescue_unavailable_or_empty = 0
        self._bm25_rescue_no_positive_uplift = 0
        self._bm25_rescue_constant_uplift = 0
        self._bm25_rescue_unchanged_order = 0
        self._bm25_rescue_successful_reorders = 0
        self._bm25_rescue_fallbacks = 0
        self._sessions: dict[str, IntentState] = {}
        self._profile_priors: dict[bytes, ProfilePrior] = {}
        self._profiles_reset = 0
        self._zero_mask_profiles = 0
        self._nonzero_mask_profiles = 0
        self._recognized_theme_count = 0
        self._turns_disabled_by_active_requirements = 0
        self._eligible_stage_a_attempts = 0
        self._empty_represented_theme_fallbacks = 0
        self._constant_score_neutral_fallbacks = 0
        self._successful_residual_applications = 0
        self._parsing_or_scoring_fallbacks = 0
        self._slates: dict[str, SlateState] = {}
        self._slate_attempts = 0
        self._slate_successes = 0
        self._slate_failures = 0
        self._slate_initializations = 0
        self._slate_ranking_resets = 0
        self._slate_stagnant_turns = 0
        self._slate_unseen_selected_on_stagnant = 0
        self._slate_repeat_backfills = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(user_profile, dict):
            raise TypeError("user_profile must be a dictionary")
        if self._profile_policy is ProfilePolicy.DISABLED:
            profile_prior = NEUTRAL_PROFILE_PRIOR
        else:
            try:
                profile_prior = parse_profile_prior(user_profile)
                if not isinstance(profile_prior, ProfilePrior):
                    raise TypeError("parse_profile_prior must return ProfilePrior")
            except Exception:
                profile_prior = NEUTRAL_PROFILE_PRIOR
                self._parsing_or_scoring_fallbacks += 1
        self._profiles_reset += 1
        if profile_prior.is_neutral:
            self._zero_mask_profiles += 1
        else:
            self._nonzero_mask_profiles += 1
        self._recognized_theme_count += profile_prior.active_theme_count
        self._profile_priors[_profile_session_key(session_id)] = profile_prior
        self._orchestrator.reset(session_id)
        self._sessions[session_id] = IntentState()
        self._slates[session_id] = SlateState()
        self._local_intent_semantic_sessions.discard(session_id)
        if (
            self.decision_policy in PROTOCOL_DECISION_POLICIES
            or self.protocol_catalog_policy
            is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
        ):
            self._protocol_consistency[session_id] = True
            self._protocol_events[session_id] = ()
            self._protocol_override_pending[session_id] = False
            self._protocol_shown_ids[session_id] = ()
            self._protocol_action_traces.pop(session_id, None)
        if self.protocol_catalog_policy is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY:
            self._protocol_refuted_ids[session_id] = ()
            self._protocol_pending_ids[session_id] = ()
            self._protocol_pending_refutable[session_id] = False
        if self.decision_policy is PROTOCOL_UTILITY_DECISION_POLICY:
            self._protocol_hybrid_sticky[session_id] = False
            self._protocol_route_modes.pop(session_id, None)
        if self.decision_policy is EXPECTED_UTILITY_DECISION_POLICY:
            self._expected_exact_replays.pop(session_id, None)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        prior_state = self._sessions[session_id]
        intent_cacheable = True
        if self._intent_policy is LOSSLESS_MULTI_SLOT_INTENT_POLICY:
            reduction = apply_user_message_with_trace(
                prior_state,
                user_message,
                turn,
                policy=self._intent_policy,
            )
            state = reduction.state
            intent_cacheable = reduction.status not in {
                IntentReductionStatus.BOUNDS,
                IntentReductionStatus.VALIDATION_FALLBACK,
            }
        else:
            state = apply_user_message(
                prior_state,
                user_message,
                turn,
                policy=self._intent_policy,
            )
        local_prompt_tokens = 0
        local_completion_tokens = 0
        try:
            local_trigger = local_intent_trigger(state, user_message, turn)
        except Exception:
            local_trigger = None
        if self._local_intent_parser is not None and local_trigger is not None:
            self._local_intent_attempts += 1
            if local_trigger is LocalIntentTrigger.FREE_TEXT:
                self._local_intent_free_text_attempts += 1
            else:
                self._local_intent_complex_attempts += 1
            try:
                local_result = self._local_intent_parser.parse(
                    prior_state,
                    user_message,
                    turn,
                )
                if not isinstance(local_result, StructuredIntentParseResult):
                    raise TypeError(
                        "local intent parser must return StructuredIntentParseResult"
                    )
                local_state = apply_structured_intent_delta(
                    prior_state,
                    state,
                    local_result.delta,
                    turn,
                )
            except Exception:
                self._local_intent_failures += 1
            else:
                local_prompt_tokens = local_result.prompt_tokens
                local_completion_tokens = local_result.completion_tokens
                self._local_intent_prompt_tokens += local_prompt_tokens
                self._local_intent_completion_tokens += local_completion_tokens
                if local_state == state:
                    self._local_intent_no_delta += 1
                else:
                    state = local_state
                    intent_cacheable = False
                    self._local_intent_applied += 1
                    self._local_intent_semantic_sessions.add(session_id)
        try:
            retrieval_route_plan = plan_retrieval_route(
                self.retrieval_routing_policy,
                state,
                self._retriever,
                intent_cacheable=intent_cacheable,
            )
        except Exception:
            retrieval_route_plan = RetrievalRoutePlan(
                self.retrieval_routing_policy,
                RetrievalRouteMode.HYBRID,
                RetrievalRouteReason.EVIDENCE_ERROR,
            )
        self._record_retrieval_route_plan(retrieval_route_plan)
        protocol_turn_eligible = False
        protocol_exact_ids: tuple[str, ...] = ()
        protocol_structural_support_ids: tuple[str, ...] = ()
        exact_constraint_count = 0
        protocol_route_conditions: object | None = None
        protocol_route_digest: str | None = None
        protocol_outcome: str | None = None
        semantic_structural_support_ids: tuple[str, ...] | None = None
        semantic_support_ready = True
        if (
            self.decision_policy is EXPECTED_UTILITY_DECISION_POLICY
            or (
                self.protocol_catalog_policy
                is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
                and self.decision_policy
                is not PROTOCOL_UTILITY_DECISION_POLICY
            )
        ):
            self._protocol_decision_counts[0] += 1
            protocol_turn_eligible, protocol_outcome = (
                self._observe_expected_protocol_turn(
                    session_id,
                    state,
                    user_message,
                    turn,
                    intent_cacheable=intent_cacheable,
                )
            )
            if (
                protocol_turn_eligible
                and self.protocol_refutation_policy
                is ELIGIBLE_CONTINUATION_REFUTATION_POLICY
                and self._protocol_pending_refutable.get(session_id, False)
            ):
                prior_refuted = self._protocol_refuted_ids.get(session_id, ())
                pending_ids = self._protocol_pending_ids.get(session_id, ())
                self._protocol_refuted_ids[session_id] = tuple(
                    dict.fromkeys((*prior_refuted, *pending_ids))
                )
        elif self.decision_policy is PROTOCOL_UTILITY_DECISION_POLICY:
            self._protocol_decision_counts[0] += 1
            try:
                from conversational_search.decision import (
                    ProtocolObservation,
                    derive_bm25_only_conditions,
                    exact_query_constraints,
                    parse_protocol_event,
                    protocol_events_are_structured_for_routing,
                    protocol_route_dependency_digest,
                    protocol_state_is_consistent,
                    recognize_protocol_observation,
                )
                from conversational_search.protocol import ProtocolEventKind
            except Exception:
                self._protocol_consistency[session_id] = False
                protocol_outcome = "candidate_or_evidence_error"
            else:
                try:
                    observation = recognize_protocol_observation(
                        user_message,
                        turn,
                    )
                except Exception:
                    observation = ProtocolObservation.UNSUPPORTED
                if observation is not ProtocolObservation.UNSUPPORTED:
                    try:
                        event = parse_protocol_event(
                            user_message,
                            observation,
                            turn,
                            asked_attribute=state.last_asked_attribute,
                        )
                        prior_events = self._protocol_events.get(session_id, ())
                        if prior_events and event.turn <= prior_events[-1].turn:
                            raise ValueError("protocol events must advance by turn")
                        self._protocol_events[session_id] = (*prior_events, event)
                        if event.kind is ProtocolEventKind.INITIAL_TENTATIVE:
                            self._protocol_override_pending[session_id] = True
                            self._protocol_hybrid_sticky[session_id] = True
                        elif event.kind is ProtocolEventKind.OVERRIDE:
                            self._protocol_override_pending[session_id] = False
                            self._protocol_hybrid_sticky[session_id] = True
                    except Exception:
                        observation = ProtocolObservation.UNSUPPORTED
                if observation is ProtocolObservation.UNSUPPORTED:
                    self._protocol_consistency[session_id] = False
                    protocol_outcome = "unsupported_or_disabled"
                protocol_events = self._protocol_events.get(session_id, ())
                protocol_turn_eligible = self._protocol_consistency.get(
                    session_id,
                    False,
                )
                try:
                    state_consistent = (
                        protocol_turn_eligible
                        and intent_cacheable
                        and protocol_state_is_consistent(
                            state,
                            protocol_events,
                            turn,
                        )
                    )
                except Exception:
                    state_consistent = False
                if not state_consistent:
                    protocol_turn_eligible = False
                if protocol_turn_eligible:
                    try:
                        protocol_capable = (
                            getattr(
                                self._retriever,
                                "protocol_evidence_capability",
                            )
                            is PROTOCOL_EVIDENCE_CAPABILITY
                        )
                    except Exception:
                        protocol_capable = False
                    if not protocol_capable:
                        self._protocol_consistency[session_id] = False
                        protocol_turn_eligible = False
                        protocol_outcome = "capability_unavailable"
                    else:
                        category_exact = False
                        constraints: tuple[str, ...] = ()
                        try:
                            category_exact = bool(
                                self._retriever.protocol_category_exists(
                                    state.category or ""
                                )
                            )
                            constraints = exact_query_constraints(
                                state,
                                protocol_events,
                            )
                            if category_exact and constraints:
                                raw_exact_constraint_count = (
                                    self._retriever.protocol_exact_constraint_count(
                                        state.category or "", constraints
                                    )
                                )
                                if (
                                    type(raw_exact_constraint_count) is not int
                                    or not 0
                                    <= raw_exact_constraint_count
                                    <= len(constraints)
                                ):
                                    raise ValueError(
                                        "exact constraint count is invalid"
                                    )
                                exact_constraint_count = raw_exact_constraint_count
                                protocol_exact_ids = tuple(
                                    self._retriever.protocol_exact_candidates(
                                        state.category or "",
                                        constraints,
                                        limit=MAX_CANDIDATE_DOCUMENTS,
                                    )
                                )
                        except Exception:
                            protocol_outcome = "candidate_or_evidence_error"
                            category_exact = False
                            constraints = ()
                            exact_constraint_count = 0
                            protocol_exact_ids = ()
                        try:
                            protocol_route_conditions = (
                                derive_bm25_only_conditions(
                                    state,
                                    message_is_exact_protocol=True,
                                    session_state_is_consistent=(
                                        state_consistent
                                    ),
                                    category_is_exactly_recognized=(
                                        category_exact
                                    ),
                                    exact_product_constraints=exact_constraint_count,
                                    session_forces_hybrid=(
                                        self._protocol_hybrid_sticky.get(
                                            session_id,
                                            False,
                                        )
                                    ),
                                    protocol_values_are_structured=(
                                        protocol_events_are_structured_for_routing(
                                            protocol_events
                                        )
                                    ),
                                )
                            )
                        except Exception:
                            protocol_route_conditions = None
                            protocol_outcome = "candidate_or_evidence_error"
                        if (
                            protocol_route_conditions is not None
                            and protocol_route_conditions.pre_bm25_eligible
                            and protocol_exact_ids
                        ):
                            try:
                                from conversational_search.exact_evidence import (
                                    rank_exact_evidence,
                                )

                                exact_evidence = (
                                    self._retriever.candidate_protocol_evidence(
                                        protocol_exact_ids
                                    )
                                )
                                exact_route = (
                                    self._validate_exact_evidence_ranking(
                                        rank_exact_evidence(
                                            protocol_exact_ids,
                                            exact_evidence,
                                            state,
                                            protocol_events=protocol_events,
                                        ),
                                        expected_ids=protocol_exact_ids,
                                    )
                                )
                                protocol_structural_support_ids = (
                                    exact_route.consistent_support_ids
                                )
                            except Exception:
                                protocol_structural_support_ids = ()
                                protocol_outcome = (
                                    "candidate_or_evidence_error"
                                )
                        if protocol_route_conditions is not None:
                            try:
                                protocol_route_digest = (
                                    protocol_route_dependency_digest(
                                        protocol_route_conditions,
                                        protocol_events,
                                        protocol_structural_support_ids,
                                    )
                                )
                            except Exception:
                                protocol_route_conditions = None
                                protocol_structural_support_ids = ()
                                protocol_route_digest = None
                                protocol_outcome = (
                                    "candidate_or_evidence_error"
                                )
                elif protocol_route_conditions is None:
                    try:
                        protocol_route_conditions = derive_bm25_only_conditions(
                            state,
                            message_is_exact_protocol=(
                                observation is not ProtocolObservation.UNSUPPORTED
                            ),
                            session_state_is_consistent=False,
                            category_is_exactly_recognized=False,
                            exact_product_constraints=0,
                            session_forces_hybrid=(
                                self._protocol_hybrid_sticky.get(
                                    session_id,
                                    False,
                                )
                            ),
                            protocol_values_are_structured=False,
                        )
                    except Exception:
                        protocol_route_conditions = None
        if (
            self.semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        ):
            try:
                from conversational_search.decision import exact_query_constraints

                semantic_capable = (
                    getattr(self._retriever, "protocol_evidence_capability")
                    is PROTOCOL_EVIDENCE_CAPABILITY
                )
                semantic_constraints = exact_query_constraints(state)
                semantic_category = state.category or ""
                if (
                    semantic_capable
                    and semantic_category
                    and semantic_constraints
                    and self._retriever.protocol_category_exists(
                        semantic_category
                    )
                ):
                    raw_support = tuple(
                        self._retriever.protocol_exact_candidates(
                            semantic_category,
                            semantic_constraints,
                            limit=MAX_CANDIDATE_DOCUMENTS,
                        )
                    )
                    semantic_structural_support_ids = tuple(
                        self._sanitize(raw_support, MAX_CANDIDATE_DOCUMENTS)
                    )
                    if semantic_structural_support_ids != raw_support:
                        raise ValueError(
                            "semantic structural support is malformed"
                        )
                elif not semantic_capable:
                    semantic_support_ready = False
                else:
                    # ``None`` means the state has no safe structured rescue
                    # contract.  An empty tuple means the contract is valid but
                    # the catalog contains no satisfying candidate.
                    semantic_structural_support_ids = None
            except Exception:
                semantic_structural_support_ids = None
                semantic_support_ready = False
        profile_key = _profile_session_key(session_id)
        profile_prior = self._profile_priors.get(profile_key)
        if not isinstance(profile_prior, ProfilePrior):
            profile_prior = NEUTRAL_PROFILE_PRIOR
            self._profile_priors[profile_key] = profile_prior
            self._parsing_or_scoring_fallbacks += 1
        try:
            profile_digest = self._profile_policy.ranking_digest(profile_prior)
        except Exception:
            profile_prior = NEUTRAL_PROFILE_PRIOR
            self._profile_priors[profile_key] = profile_prior
            profile_digest = DEFAULT_PROFILE_DEPENDENCY_DIGEST
            self._parsing_or_scoring_fallbacks += 1
        profile_enabled_and_recognized = (
            self._profile_policy is ProfilePolicy.BOUNDED_RESIDUAL
            and not profile_prior.is_neutral
        )
        profile_residual_eligible = (
            profile_enabled_and_recognized and not state.requirements
        )
        if profile_enabled_and_recognized and state.requirements:
            self._turns_disabled_by_active_requirements += 1
        dense_query = render_dense_query(state)
        lexical_query = render_lexical_query(state)
        probe_enabled = (
            self.requirement_probe_policy
            is not DISABLED_REQUIREMENT_PROBE_POLICY
        )
        probe_candidates: tuple[str, ...] = ()
        probe_render_failed = False
        if probe_enabled:
            try:
                probe_candidates = render_requirement_probe_candidates(state)
            except Exception:
                probe_render_failed = True
        if probe_enabled:
            try:
                probe_backend_capable = (
                    getattr(self._retriever, "requirement_probe_capability")
                    is REQUIREMENT_PROBE_CAPABILITY
                )
            except Exception:
                probe_backend_capable = False
        else:
            probe_backend_capable = False
        result_count = min(max(top_k, 0), 10)
        output_count = result_count
        route_weights = self._fusion_policy.choose(state)
        protocol_conditional_dense = bool(
            self.decision_policy is PROTOCOL_UTILITY_DECISION_POLICY
            and protocol_turn_eligible
            and protocol_route_conditions is not None
            and protocol_route_conditions.pre_bm25_eligible
            and protocol_route_digest is not None
        )
        smart_conditional_dense = bool(
            self.retrieval_routing_policy
            is SMART_HYBRID_RETRIEVAL_ROUTING_POLICY
            and retrieval_route_plan.mode is RetrievalRouteMode.BM25_FIRST
        )
        conditional_dense = protocol_conditional_dense or smart_conditional_dense
        semantic_rescue_enabled = (
            self.semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        )
        use_dense = not conditional_dense and not semantic_rescue_enabled
        dense_retrieval_policy = (
            self.semantic_lexical_rescue_policy.value
            if semantic_rescue_enabled
            else retrieval_route_plan.dependency_key
            if self.retrieval_routing_policy
            is SMART_HYBRID_RETRIEVAL_ROUTING_POLICY
            else DENSE_ALWAYS_RETRIEVAL_POLICY
            if use_dense
            else PROTOCOL_CONDITIONAL_DENSE_RETRIEVAL_POLICY
        )
        retrieval_policy: str | None = (
            self.requirement_probe_policy.value if probe_enabled else None
        )
        if self.decision_policy is PROTOCOL_UTILITY_DECISION_POLICY:
            route_policy_dependency = dense_retrieval_policy
            if conditional_dense:
                route_policy_dependency += f":{protocol_route_digest}"
            retrieval_policy = route_policy_dependency + (
                f"|{retrieval_policy}" if retrieval_policy is not None else ""
            )
        elif (
            self.retrieval_routing_policy
            is SMART_HYBRID_RETRIEVAL_ROUTING_POLICY
        ):
            retrieval_policy = dense_retrieval_policy + (
                f"|{retrieval_policy}" if retrieval_policy is not None else ""
            )
        elif semantic_rescue_enabled:
            retrieval_policy = dense_retrieval_policy
        if self.semantic_tiebreak_policy is not DISABLED_SEMANTIC_TIEBREAK_POLICY:
            semantic_tiebreak_dependency = (
                f"semantic-tiebreak:{self.semantic_tiebreak_policy.value}"
            )
            retrieval_policy = (
                semantic_tiebreak_dependency
                if retrieval_policy is None
                else f"{retrieval_policy}|{semantic_tiebreak_dependency}"
            )
        if self.field_semantic_policy is not DISABLED_FIELD_SEMANTIC_POLICY:
            field_semantic_dependency = (
                f"field-semantic:{self.field_semantic_policy.value}"
            )
            retrieval_policy = (
                field_semantic_dependency
                if retrieval_policy is None
                else f"{retrieval_policy}|{field_semantic_dependency}"
            )
        try:
            capability = getattr(self._retriever, "ranking_cache_capability")
            backend_cache_capable = capability is EXACT_RANKING_CACHE_CAPABILITY
            candidate_snapshot_token = (
                getattr(self._retriever, "snapshot_token")
                if backend_cache_capable
                else None
            )
        except Exception:
            backend_cache_capable = False
            candidate_snapshot_token = None
        backend_snapshot_token = (
            candidate_snapshot_token
            if backend_cache_capable
            and type(candidate_snapshot_token) is BackendSnapshotToken
            else None
        )
        cache_eligible = (
            self._ranking_policy in STAGE_A_RANKING_POLICIES
            and backend_snapshot_token is not None
            and (
                not probe_enabled
                or (probe_backend_capable and not probe_render_failed)
            )
            and (not semantic_rescue_enabled or semantic_support_ready)
        )
        decision = self._orchestrator.decide(
            session_id,
            state,
            dense_query,
            lexical_query,
            route_weights,
            self._ranking_policy,
            result_count,
            backend_snapshot_token,
            cache_eligible,
            profile_digest=profile_digest,
            retrieval_policy=retrieval_policy,
        )

        retrieval: RetrievalResult | None = None
        exact_context: _ExactRankingContext | None = None
        protocol_resolution: ProtocolResolution | None = None
        probe_result_cacheable = True
        parent_asins: object = ()
        full_ranked_ids: tuple[str, ...] | None = None
        if semantic_rescue_enabled:
            dense_options: dict[str, object] = {
                "use_dense": False,
                "dense_rescue_on_bm25_failure": False,
                "bm25_only_support_ids": semantic_structural_support_ids,
                "semantic_lexical_rescue_policy": (
                    self.semantic_lexical_rescue_policy
                ),
                "semantic_rescue_category": state.category or "",
            }
        elif use_dense:
            dense_options = {}
        else:
            dense_options = {
                "use_dense": False,
                "bm25_only_support_ids": (
                    protocol_structural_support_ids
                    if protocol_conditional_dense
                    else retrieval_route_plan.structural_support_ids
                ),
                "bm25_only_requires_all_support": smart_conditional_dense,
            }
        if decision.action is QueryAction.REUSE:
            if (
                self._slate_policy is REPEAT_TOP_SLATE_POLICY
                and self.evidence_exposure_policy
                is DISABLED_EVIDENCE_EXPOSURE_POLICY
            ):
                parent_asins = decision.cached_ranked_ids[:result_count]
            else:
                full_ranked_ids = decision.cached_ranked_ids
                parent_asins = full_ranked_ids
        elif decision.action is QueryAction.SEARCH:
            if self.decision_policy is EXPECTED_UTILITY_DECISION_POLICY:
                self._expected_exact_replays.pop(session_id, None)
            if not probe_enabled:
                try:
                    retrieval = self._retriever.search_with_trace(
                        dense_query,
                        lexical_query,
                        top_k=result_count,
                        route_weights=route_weights,
                        **dense_options,
                    )
                    if not isinstance(retrieval, RetrievalResult):
                        raise TypeError(
                            "search_with_trace must return RetrievalResult"
                        )
                    if semantic_rescue_enabled:
                        retrieval = self._validate_semantic_lexical_retrieval(
                            retrieval
                        )
                        semantic_status = retrieval.semantic_trace.status
                        probe_result_cacheable = (
                            semantic_status in SEMANTIC_RESCUE_CACHEABLE_STATUSES
                        )
                        self._record_semantic_rescue_status(semantic_status)
                except Exception:
                    if semantic_rescue_enabled:
                        probe_result_cacheable = False
                        self._record_semantic_rescue_status(
                            None,
                            validation_fallback=True,
                        )
                        try:
                            retrieval = self._retriever.search_with_trace(
                                dense_query,
                                lexical_query,
                                top_k=result_count,
                                route_weights=route_weights,
                                use_dense=False,
                                dense_rescue_on_bm25_failure=False,
                            )
                            if not isinstance(retrieval, RetrievalResult):
                                raise TypeError(
                                    "baseline fail-open must return RetrievalResult"
                                )
                        except Exception:
                            retrieval = None
                    else:
                        retrieval = None
            elif probe_render_failed or not probe_backend_capable:
                probe_result_cacheable = False
                self._record_requirement_probe_status(
                    "error" if probe_render_failed else "unavailable",
                    validation_fallback=probe_render_failed,
                )
                try:
                    retrieval = self._retriever.search_with_trace(
                        dense_query,
                        lexical_query,
                        top_k=result_count,
                        route_weights=route_weights,
                        **dense_options,
                    )
                    if not isinstance(retrieval, RetrievalResult):
                        raise TypeError(
                            "search_with_trace must return RetrievalResult"
                        )
                except Exception:
                    retrieval = None
            else:
                try:
                    retrieval = self._retriever.search_with_trace(
                        dense_query,
                        lexical_query,
                        top_k=result_count,
                        route_weights=route_weights,
                        requirement_probe_policy=(
                            self.requirement_probe_policy
                        ),
                        requirement_probe_candidates=probe_candidates,
                        **dense_options,
                    )
                    self._validate_requirement_probe_retrieval(retrieval)
                except Exception:
                    retrieval = None
                    probe_result_cacheable = False
                    self._record_requirement_probe_status(
                        "error",
                        validation_fallback=True,
                    )
                else:
                    probe_status = retrieval.probe_trace.status
                    probe_result_cacheable = (
                        probe_status in CACHEABLE_REQUIREMENT_PROBE_STATUSES
                    )
                    self._record_requirement_probe_status(
                        probe_status,
                        query_count=retrieval.probe_trace.query_count,
                        additions=len(retrieval.probe_trace.supplemental_ids),
                    )
            parent_asins = () if retrieval is None else retrieval.recommendations

            if (
                retrieval is not None
                and self._ranking_policy in STAGE_A_RANKING_POLICIES
                and retrieval.trace.fused_ids
                and not retrieval.trace.used_fallback
            ):
                self._reranking_attempts += 1
                ranking_cacheable = True
                reranking_failed = False
                documents = None
                try:
                    documents = self._retriever.candidate_documents(
                        retrieval.trace.fused_ids
                    )
                    if documents and profile_residual_eligible:
                        self._eligible_stage_a_attempts += 1
                    if documents:
                        if (
                            self._ranking_policy
                            is RankingPolicy.ROUTE_REDUNDANCY_CORRECTED
                        ):
                            route_statuses_are_safe = (
                                retrieval.trace.bm25_status
                                in CACHEABLE_ROUTE_STATUSES
                                and retrieval.trace.dense_status
                                in CACHEABLE_ROUTE_STATUSES
                            )
                            if not route_statuses_are_safe:
                                self._record_route_redundancy_status(
                                    RouteRedundancyStatus.SCORING_FALLBACK
                                )
                                ranked, _phase9_cacheable = (
                                    self._rerank_exact_phase9(
                                        state,
                                        documents,
                                        retrieval,
                                        route_weights,
                                        profile_prior,
                                    )
                                )
                                ranking_cacheable = False
                            else:
                                try:
                                    redundancy_ranking = (
                                        rerank_stage_a_with_profile_and_route_redundancy(
                                            state,
                                            documents,
                                            bm25_ids=retrieval.trace.bm25_ids,
                                            dense_ids=retrieval.trace.dense_ids,
                                            fused_ids=retrieval.trace.fused_ids,
                                            route_weights=route_weights,
                                            profile_prior=profile_prior,
                                            profile_policy=self._profile_policy,
                                        )
                                    )
                                    self._validate_route_redundancy_ranking(
                                        redundancy_ranking,
                                        state=state,
                                        fused_ids=retrieval.trace.fused_ids,
                                    )
                                except Exception:
                                    self._record_route_redundancy_status(
                                        RouteRedundancyStatus.SCORING_FALLBACK
                                    )
                                    ranked, _phase9_cacheable = (
                                        self._rerank_exact_phase9(
                                            state,
                                            documents,
                                            retrieval,
                                            route_weights,
                                            profile_prior,
                                        )
                                    )
                                    ranking_cacheable = False
                                else:
                                    ranked = redundancy_ranking.ranking
                                    ranking_cacheable = (
                                        redundancy_ranking.status
                                        is not RouteRedundancyStatus.SCORING_FALLBACK
                                        and redundancy_ranking.profile_status
                                        is not ProfileResidualStatus.SCORING_FALLBACK
                                    )
                                    self._record_route_redundancy_status(
                                        redundancy_ranking.status
                                    )
                                    self._record_profile_status(
                                        redundancy_ranking.profile_status
                                    )
                        elif (
                            self._ranking_policy
                            is RankingPolicy.COMPLETENESS_BM25_RESCUE
                        ):
                            self._bm25_rescue_attempts += 1
                            if retrieval.trace.bm25_status != "ok":
                                self._record_bm25_rescue_status(
                                    Bm25RescueStatus.EMPTY_BM25
                                )
                                ranked, ranking_cacheable = (
                                    self._rerank_exact_phase9(
                                        state,
                                        documents,
                                        retrieval,
                                        route_weights,
                                        profile_prior,
                                    )
                                )
                            else:
                                try:
                                    rescue_ranking = (
                                        rerank_stage_a_with_profile_and_bm25_rescue(
                                            state,
                                            documents,
                                            bm25_ids=retrieval.trace.bm25_ids,
                                            dense_ids=retrieval.trace.dense_ids,
                                            fused_ids=retrieval.trace.fused_ids,
                                            route_weights=route_weights,
                                            profile_prior=profile_prior,
                                            profile_policy=self._profile_policy,
                                        )
                                    )
                                    self._validate_bm25_rescue_ranking(
                                        rescue_ranking,
                                        state=state,
                                        fused_ids=retrieval.trace.fused_ids,
                                    )
                                except Exception:
                                    self._record_bm25_rescue_status(
                                        Bm25RescueStatus.SCORING_FALLBACK
                                    )
                                    ranked, _phase9_cacheable = (
                                        self._rerank_exact_phase9(
                                            state,
                                            documents,
                                            retrieval,
                                            route_weights,
                                            profile_prior,
                                        )
                                    )
                                    ranking_cacheable = False
                                else:
                                    ranked = rescue_ranking.ranking
                                    ranking_cacheable = (
                                        rescue_ranking.status
                                        is not Bm25RescueStatus.SCORING_FALLBACK
                                        and rescue_ranking.profile_status
                                        is not ProfileResidualStatus.SCORING_FALLBACK
                                    )
                                    self._record_bm25_rescue_status(
                                        rescue_ranking.status
                                    )
                                    self._record_profile_status(
                                        rescue_ranking.profile_status
                                    )
                        else:
                            ranked, ranking_cacheable = (
                                self._rerank_exact_phase9(
                                    state,
                                    documents,
                                    retrieval,
                                    route_weights,
                                    profile_prior,
                                )
                            )
                    else:
                        ranked = None
                except Exception:
                    reranking_failed = True
                if reranking_failed:
                    self._reranking_failures += 1
                else:
                    if ranked is None:
                        self._reranking_unavailable_skips += 1
                    else:
                        self._reranking_successes += 1
                        sanitized_ranked_ids = tuple(
                            self._sanitize(
                                ranked.ranked_ids,
                                MAX_SLATE_CANDIDATES,
                            )
                        )
                        complete_stage_a_ranking = (
                            bool(sanitized_ranked_ids)
                            and sanitized_ranked_ids == ranked.ranked_ids
                        )
                        if (
                            self.field_semantic_policy
                            is not DISABLED_FIELD_SEMANTIC_POLICY
                            and session_id
                            in self._local_intent_semantic_sessions
                            and sanitized_ranked_ids
                        ):
                            (
                                sanitized_ranked_ids,
                                field_semantic_cacheable,
                            ) = self._apply_field_semantic_ranking(
                                state,
                                sanitized_ranked_ids,
                            )
                            ranking_cacheable = (
                                ranking_cacheable and field_semantic_cacheable
                            )
                        if (
                            self._ranking_policy
                            is RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE
                            and sanitized_ranked_ids
                        ):
                            exact_context = self._apply_exact_evidence_ranking(
                                state,
                                sanitized_ranked_ids,
                                dense_ids=retrieval.trace.dense_ids,
                                dense_scores=retrieval.trace.dense_scores,
                                protocol_events=(
                                    self._protocol_events.get(session_id, ())
                                    if self.decision_policy
                                    is EXPECTED_UTILITY_DECISION_POLICY
                                    else ()
                                ),
                            )
                            sanitized_ranked_ids = (
                                exact_context.output_ranked_ids
                            )
                            exact_cacheable = exact_context.cacheable
                            ranking_cacheable = (
                                ranking_cacheable and exact_cacheable
                            )
                        elif (
                            self._ranking_policy
                            is RankingPolicy.IMPORTANCE_AWARE_SATISFACTION
                            and sanitized_ranked_ids
                        ):
                            (
                                sanitized_ranked_ids,
                                importance_cacheable,
                            ) = self._apply_importance_aware_ranking(
                                state,
                                sanitized_ranked_ids,
                                bm25_ids=retrieval.trace.bm25_ids,
                                documents=documents,
                                profile_prior=profile_prior,
                            )
                            ranking_cacheable = (
                                ranking_cacheable and importance_cacheable
                            )
                        route_is_cacheable = (
                            retrieval.trace.bm25_status
                            in CACHEABLE_ROUTE_STATUSES
                            and retrieval.trace.dense_status
                            in CACHEABLE_ROUTE_STATUSES
                        )
                        complete_ranking = (
                            complete_stage_a_ranking
                            and len(sanitized_ranked_ids)
                            == len(ranked.ranked_ids)
                            and set(sanitized_ranked_ids)
                            == set(ranked.ranked_ids)
                        )
                        if (
                            route_is_cacheable
                            and complete_ranking
                            and ranking_cacheable
                            and intent_cacheable
                            and probe_result_cacheable
                        ):
                            cache_committed = self._orchestrator.commit(
                                session_id,
                                decision,
                                backend_snapshot_token,
                                sanitized_ranked_ids,
                            )
                            if (
                                cache_committed
                                and self.decision_policy
                                is EXPECTED_UTILITY_DECISION_POLICY
                                and exact_context is not None
                                and exact_context.result is not None
                                and decision.dependency_digest is not None
                                and type(backend_snapshot_token)
                                is BackendSnapshotToken
                            ):
                                self._expected_exact_replays[session_id] = (
                                    _ExpectedExactReplay(
                                        decision.dependency_digest,
                                        backend_snapshot_token,
                                        exact_context.base_ranked_ids,
                                        exact_context.output_ranked_ids,
                                    )
                                )
                        if (
                            self._slate_policy is REPEAT_TOP_SLATE_POLICY
                            and self.evidence_exposure_policy
                            is DISABLED_EVIDENCE_EXPOSURE_POLICY
                        ):
                            parent_asins = sanitized_ranked_ids[:result_count]
                        else:
                            full_ranked_ids = sanitized_ranked_ids
                            parent_asins = full_ranked_ids

        if (
            self.decision_policy is EXPECTED_UTILITY_DECISION_POLICY
            and decision.action is QueryAction.REUSE
            and full_ranked_ids is not None
        ):
            replay = self._expected_exact_replays.get(session_id)
            replay_valid = bool(
                isinstance(replay, _ExpectedExactReplay)
                and decision.dependency_digest == replay.dependency_digest
                and decision.cached_ranked_ids == replay.output_ranked_ids
                and backend_snapshot_token is replay.backend_snapshot_token
            )
            if replay_valid:
                assert replay is not None
                exact_context = self._apply_exact_evidence_ranking(
                    state,
                    replay.base_ranked_ids,
                    protocol_events=self._protocol_events.get(session_id, ()),
                )
                if exact_context.result is not None:
                    full_ranked_ids = exact_context.output_ranked_ids
                    parent_asins = full_ranked_ids

        if (
            self.evidence_exposure_policy
            is not DISABLED_EVIDENCE_EXPOSURE_POLICY
            and decision.action is QueryAction.REUSE
            and full_ranked_ids is not None
            and exact_context is None
        ):
            exact_context = self._apply_exact_evidence_ranking(
                state,
                full_ranked_ids,
            )
            if (
                exact_context.result is not None
                and set(exact_context.output_ranked_ids) == set(full_ranked_ids)
            ):
                full_ranked_ids = exact_context.output_ranked_ids
                parent_asins = full_ranked_ids

        if (
            self.protocol_catalog_policy
            is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
            and protocol_turn_eligible
            and full_ranked_ids is not None
            and state.category
        ):
            protected_ranked_ids = full_ranked_ids
            protected_exact_context = exact_context
            try:
                category_evidence = tuple(
                    self._retriever.protocol_category_evidence(state.category)
                )
                resolution = resolve_protocol_transcript(
                    category_evidence,
                    self._protocol_events.get(session_id, ()),
                    observed_turn_count=turn,
                    refuted_ids=frozenset(
                        self._protocol_refuted_ids.get(session_id, ())
                    ),
                )
                protocol_pool = fuse_protocol_candidates(
                    resolution,
                    protected_ranked_ids,
                    limit=MAX_CANDIDATE_DOCUMENTS,
                )
                if not resolution.exact or not protocol_pool:
                    raise ValueError("full protocol resolution has no support")
                augmented_exact_context = self._apply_exact_evidence_ranking(
                    state,
                    protocol_pool,
                    protocol_events=self._protocol_events.get(session_id, ()),
                )
                if (
                    augmented_exact_context.result is None
                    or set(augmented_exact_context.output_ranked_ids)
                    != set(protocol_pool)
                ):
                    raise ValueError("full protocol ranking is incomplete")
            except Exception:
                protocol_resolution = None
                full_ranked_ids = protected_ranked_ids
                parent_asins = full_ranked_ids
                exact_context = protected_exact_context
            else:
                protocol_resolution = resolution
                exact_context = augmented_exact_context
                full_ranked_ids = augmented_exact_context.output_ranked_ids
                parent_asins = full_ranked_ids

        self._record_retrieval_route_outcome(decision.action, retrieval)

        protocol_route_mode = "hybrid"
        if self.decision_policy is PROTOCOL_UTILITY_DECISION_POLICY:
            if decision.action is QueryAction.REUSE:
                protocol_route_mode = self._protocol_route_modes.get(
                    session_id,
                    "hybrid",
                )
                bm25_support = protocol_route_mode == "bm25_only"
            elif retrieval is None:
                bm25_support = False
            else:
                bm25_support = (
                    retrieval.trace.bm25_status == "ok"
                    and bool(
                        set(retrieval.trace.bm25_ids).intersection(
                            protocol_structural_support_ids
                        )
                    )
                )
                if retrieval.trace.used_fallback:
                    protocol_route_mode = "fallback"
                elif (
                    bm25_support
                    and retrieval.trace.dense_status == "skipped"
                ):
                    protocol_route_mode = "bm25_only"
                elif (
                    retrieval.trace.bm25_status
                    in {"unavailable", "error", "empty"}
                    and retrieval.trace.dense_status == "ok"
                ):
                    protocol_route_mode = "dense_rescue"
                else:
                    protocol_route_mode = "hybrid"
                self._protocol_route_modes[session_id] = protocol_route_mode
            if protocol_route_conditions is not None:
                try:
                    protocol_route_conditions = (
                        protocol_route_conditions.with_bm25_support(
                            bm25_support
                        )
                    )
                except Exception:
                    protocol_route_conditions = None
                    protocol_route_mode = "hybrid"

        prior_slate_state = self._slates[session_id]
        protocol_applied = False
        planned_question: str | None = None
        protocol_decision = None
        if (
            self.decision_policy is EXPECTED_UTILITY_DECISION_POLICY
            and protocol_turn_eligible
            and full_ranked_ids is not None
            and result_count > 0
        ):
            try:
                from conversational_search.decision import (
                    ProtocolDecisionStatus,
                    plan_expected_utility_decision,
                )
                from conversational_search.exact_evidence import ExactEvidenceResult

                if (
                    exact_context is None
                    or not isinstance(exact_context.result, ExactEvidenceResult)
                    or not exact_context.evidence
                ):
                    raise ValueError("current exact ranking context is unavailable")
                signature = ranking_signature(
                    state,
                    dense_query,
                    lexical_query,
                    route_weights,
                    self._ranking_policy.value,
                    full_ranked_ids,
                    result_count,
                )
                prior_shown = self._protocol_shown_ids.get(session_id, ())
                protocol_decision = plan_expected_utility_decision(
                    state,
                    exact_context.result,
                    exact_context.evidence,
                    slate_state=prior_slate_state,
                    ranking_signature=signature,
                    shown_ids=prior_shown,
                    protocol_events=self._protocol_events.get(session_id, ()),
                    current_turn=turn,
                    requested_top_k=result_count,
                    protocol_locked=self._protocol_override_pending.get(
                        session_id,
                        False,
                    ),
                    intent_policy=self._intent_policy,
                    retrieval_was_reused=(
                        decision.action is QueryAction.REUSE
                    ),
                )
                if protocol_decision.status is ProtocolDecisionStatus.APPLIED:
                    if not 0 <= protocol_decision.width <= result_count:
                        raise ValueError("protocol width is outside the API bound")
                    if (
                        turn < 10
                        and protocol_decision.question is not None
                        and protocol_decision.question not in QUESTION_TEXT
                    ):
                        raise ValueError("protocol question is not supported")
                    if turn >= 10 and protocol_decision.question is not None:
                        raise ValueError("the final turn cannot ask a question")
                    if protocol_decision.ordered_ids != full_ranked_ids:
                        raise ValueError(
                            "expected-utility planner changed the active ranking"
                        )
                    output_count = protocol_decision.width
                    planned_question = protocol_decision.question
                    protocol_applied = True
                    protocol_outcome = ProtocolDecisionStatus.APPLIED.value
                else:
                    protocol_outcome = protocol_decision.status.value
            except Exception:
                protocol_outcome = "candidate_or_evidence_error"
        elif (
            self.decision_policy is PROTOCOL_UTILITY_DECISION_POLICY
            and
            protocol_turn_eligible
            and full_ranked_ids is not None
            and result_count > 0
        ):
            try:
                from conversational_search.decision import (
                    ProtocolDecisionStatus,
                    plan_protocol_decision,
                )
                from conversational_search.exact_evidence import rank_exact_evidence

                protocol_pool = tuple(
                    self._sanitize(
                        (*full_ranked_ids, *protocol_exact_ids),
                        MAX_CANDIDATE_DOCUMENTS,
                    )
                )
                if protocol_pool[: len(full_ranked_ids)] != full_ranked_ids:
                    raise ValueError("protocol union displaced protected candidates")
                protocol_evidence = self._retriever.candidate_protocol_evidence(
                    protocol_pool
                )
                protocol_events = self._protocol_events.get(session_id, ())
                exact_result = rank_exact_evidence(
                    protocol_pool,
                    protocol_evidence,
                    state,
                    protocol_events=protocol_events,
                )
                prior_shown = self._protocol_shown_ids.get(session_id, ())
                protocol_decision = plan_protocol_decision(
                    state,
                    exact_result,
                    protocol_evidence,
                    shown_ids=prior_shown,
                    protocol_events=protocol_events,
                    current_turn=turn,
                    requested_top_k=result_count,
                    protocol_locked=self._protocol_override_pending.get(
                        session_id,
                        False,
                    ),
                )
                if protocol_decision.status is ProtocolDecisionStatus.APPLIED:
                    if not 0 <= protocol_decision.width <= result_count:
                        raise ValueError("protocol width is outside the API bound")
                    if turn < 10 and protocol_decision.question not in QUESTION_TEXT:
                        raise ValueError("protocol question is not supported")
                    if turn >= 10 and protocol_decision.question is not None:
                        raise ValueError("the final turn cannot ask a question")
                    expected_available = tuple(
                        parent_asin
                        for parent_asin in exact_result.ranked_ids
                        if parent_asin not in frozenset(prior_shown)
                    )
                    if protocol_decision.ordered_ids != expected_available:
                        raise ValueError("protocol decision candidate order is invalid")
                    full_ranked_ids = protocol_decision.ordered_ids
                    parent_asins = full_ranked_ids
                    output_count = protocol_decision.width
                    planned_question = protocol_decision.question
                    protocol_applied = True
                    protocol_outcome = ProtocolDecisionStatus.APPLIED.value
                else:
                    protocol_outcome = protocol_decision.status.value
            except Exception:
                protocol_outcome = "candidate_or_evidence_error"
        elif protocol_turn_eligible and protocol_outcome is None:
            protocol_outcome = "candidate_or_evidence_error"

        exposure_applied = False
        exposure_withheld = False
        if (
            self.evidence_exposure_policy in {
                TOP3_STRUCTURAL_EXPOSURE_POLICY,
                BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
                BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
                BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY,
                PROTOCOL_POSTERIOR_EXPOSURE_POLICY,
                PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
                PROTOCOL_REPLY_TREE_EXPOSURE_POLICY,
            }
            and full_ranked_ids is not None
            and result_count > 0
        ):
            retrieval_fault_or_fallback = False
            if decision.action is QueryAction.SEARCH:
                retrieval_fault_or_fallback = bool(
                    retrieval is None
                    or retrieval.trace.used_fallback
                    or retrieval.trace.bm25_status in {"unavailable", "error"}
                    or retrieval.trace.dense_status == "error"
                )
                if semantic_rescue_enabled:
                    retrieval_fault_or_fallback = bool(
                        retrieval_fault_or_fallback
                        or not isinstance(
                            retrieval,
                            SemanticLexicalRetrievalResult,
                        )
                        or (
                            isinstance(
                                retrieval,
                                SemanticLexicalRetrievalResult,
                            )
                            and retrieval.semantic_trace.status
                            not in SEMANTIC_RESCUE_CACHEABLE_STATUSES
                        )
                    )
            if protocol_resolution is not None and protocol_resolution.exact:
                retrieval_fault_or_fallback = False
            try:
                from conversational_search.exact_evidence import ExactEvidenceResult
                from conversational_search.exposure import (
                    plan_evidence_gated_exposure,
                )

                if (
                    exact_context is None
                    or not isinstance(exact_context.result, ExactEvidenceResult)
                    or not exact_context.evidence
                ):
                    raise ValueError("exact evidence context is unavailable")
                exposure_decision = plan_evidence_gated_exposure(
                    state,
                    exact_context.result,
                    exact_context.evidence,
                    current_turn=turn,
                    requested_top_k=result_count,
                    retrieval_fault_or_fallback=retrieval_fault_or_fallback,
                    require_initial_explicit_buying=(
                        self.evidence_exposure_policy
                        in {
                            BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
                            BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
                            BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY,
                        }
                    ),
                    question_prefix_limit=(
                        3
                        if self.evidence_exposure_policy
                        in {
                            BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
                            BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY,
                        }
                        else 0
                    ),
                    initial_ambiguous_prefix_limit=(
                        1
                        if self.evidence_exposure_policy
                        is BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY
                        else 0
                    ),
                    protocol_resolution=(
                        protocol_resolution
                        if self.evidence_exposure_policy
                        in {
                            PROTOCOL_POSTERIOR_EXPOSURE_POLICY,
                            PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
                            PROTOCOL_REPLY_TREE_EXPOSURE_POLICY,
                        }
                        else None
                    ),
                    metric_aware_protocol_enumeration=(
                        self.evidence_exposure_policy
                        in {
                            PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
                            PROTOCOL_REPLY_TREE_EXPOSURE_POLICY,
                        }
                    ),
                    reply_tree_protocol_planning=(
                        self.evidence_exposure_policy
                        is PROTOCOL_REPLY_TREE_EXPOSURE_POLICY
                    ),
                )
                exposure_decision = self._validate_evidence_exposure_decision(
                    exposure_decision,
                    ranked_ids=full_ranked_ids,
                    requested_top_k=result_count,
                    current_turn=turn,
                )
            except Exception:
                self._record_evidence_exposure_status(
                    EvidenceExposureStatus.EVIDENCE_FAIL_OPEN,
                    validation_fallback=True,
                )
            else:
                self._record_evidence_exposure_status(exposure_decision.status)
                if exposure_decision.status is EvidenceExposureStatus.TOP3_CONFIDENT:
                    full_ranked_ids = exposure_decision.presentation_ids
                    parent_asins = full_ranked_ids
                    output_count = exposure_decision.width
                    planned_question = None
                    exposure_applied = True
                elif exposure_decision.status in {
                    EvidenceExposureStatus.POSTERIOR_SINGLETON,
                    EvidenceExposureStatus.POSTERIOR_PROBE,
                    EvidenceExposureStatus.POSTERIOR_REPLY_TREE,
                }:
                    full_ranked_ids = exposure_decision.presentation_ids
                    parent_asins = full_ranked_ids
                    output_count = exposure_decision.width
                    planned_question = exposure_decision.question
                    exposure_applied = True
                elif (
                    exposure_decision.status
                    is EvidenceExposureStatus.QUESTION_WITHHELD
                ):
                    output_count = 0
                    planned_question = exposure_decision.question
                    exposure_applied = True
                    exposure_withheld = True
                elif (
                    exposure_decision.status
                    in {
                        EvidenceExposureStatus.QUESTION_WITH_PREFIX,
                        EvidenceExposureStatus.AMBIGUOUS_TOP1_PREVIEW,
                    }
                ):
                    full_ranked_ids = exposure_decision.presentation_ids
                    parent_asins = full_ranked_ids
                    output_count = exposure_decision.width
                    planned_question = exposure_decision.question
                    exposure_applied = True
                elif exposure_decision.status in {
                    EvidenceExposureStatus.NO_INFORMATIVE_QUESTION,
                    EvidenceExposureStatus.FINAL_TURN,
                    EvidenceExposureStatus.POSTERIOR_BATCH,
                    EvidenceExposureStatus.POSTERIOR_ENUMERATION,
                }:
                    output_count = exposure_decision.width
                    planned_question = None
                    exposure_applied = True

        next_slate_state = prior_slate_state
        slate_trace = None
        if full_ranked_ids is not None and output_count > 0:
            self._slate_attempts += 1
            try:
                signature = ranking_signature(
                    state,
                    dense_query,
                    lexical_query,
                    route_weights,
                    self._ranking_policy.value,
                    full_ranked_ids,
                    result_count,
                )
                if self._slate_policy is INTENT_EPOCH_NOVELTY_SLATE_POLICY:
                    try:
                        epoch_selection = (
                            select_slate_with_intent_epoch_novelty(
                                prior_slate_state,
                                signature,
                                full_ranked_ids,
                                output_count,
                            )
                        )
                        self._validate_intent_epoch_slate_selection(
                            epoch_selection,
                            prior_state=prior_slate_state,
                            signature=signature,
                            ranked_ids=full_ranked_ids,
                            limit=output_count,
                        )
                    except Exception:
                        self._record_intent_epoch_slate_status(
                            IntentEpochSlateStatus.VALIDATION_FALLBACK,
                            eligible_prior_shown=0,
                        )
                        selection = select_slate(
                            STAGNATION_AWARE_SLATE_POLICY,
                            prior_slate_state,
                            signature,
                            full_ranked_ids,
                            output_count,
                        )
                    else:
                        selection = epoch_selection.selection
                        self._record_intent_epoch_slate_status(
                            epoch_selection.status,
                            eligible_prior_shown=(
                                epoch_selection.eligible_prior_shown
                            ),
                        )
                else:
                    selection = select_slate(
                        self._slate_policy,
                        prior_slate_state,
                        signature,
                        full_ranked_ids,
                        output_count,
                    )
            except Exception:
                self._slate_failures += 1
                recommendations = self._sanitize(parent_asins, output_count)
            else:
                self._slate_successes += 1
                recommendations = list(selection.selected_ids)
                next_slate_state = selection.state
                slate_trace = selection.trace
        else:
            recommendations = self._sanitize(parent_asins, output_count)

        ask_attribute = (
            planned_question
            if protocol_applied or exposure_applied
            else None if turn >= 10 else self._question_policy.choose(state)
        )
        if ask_attribute is not None:
            state = record_question(state, ask_attribute)
            if exposure_withheld:
                message = (
                    "I need one more detail before recommending. "
                    + QUESTION_TEXT[ask_attribute]
                )
            else:
                message = (
                    "Here are the closest matches so far. "
                    + QUESTION_TEXT[ask_attribute]
                )
        else:
            message = "Here are the closest matches based on your current preferences."
        self._sessions[session_id] = state
        self._slates[session_id] = next_slate_state
        if self.protocol_catalog_policy is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY:
            self._protocol_pending_ids[session_id] = tuple(recommendations)
            self._protocol_pending_refutable[session_id] = bool(
                protocol_resolution is not None
                and protocol_resolution.exact
                and self._protocol_consistency.get(session_id, False)
                and not self._protocol_override_pending.get(session_id, False)
            )
        if (
            self.decision_policy in PROTOCOL_DECISION_POLICIES
            or self.protocol_catalog_policy
            is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
        ):
            if self._protocol_consistency.get(session_id, False):
                prior_protocol_shown = self._protocol_shown_ids.get(
                    session_id,
                    (),
                )
                self._protocol_shown_ids[session_id] = tuple(
                    dict.fromkeys((*prior_protocol_shown, *recommendations))
                )
            self._record_protocol_decision_outcome(
                protocol_outcome or "candidate_or_evidence_error",
                requested_count=result_count,
                presented_count=len(recommendations),
                question=ask_attribute,
            )
            self._protocol_action_traces[session_id] = {
                "protocol_mode": (
                    "applied"
                    if protocol_applied
                    else "eligible_fail_open"
                    if protocol_turn_eligible
                    else "free_form_fail_open"
                ),
                "retrieval_action": decision.action.value,
                "retrieval_reason": decision.reason,
                "dense_policy": dense_retrieval_policy,
                "derived_track": protocol_route_mode,
                "bm25_only_conditions": (
                    protocol_route_conditions.as_dict()
                    if protocol_route_conditions is not None
                    else {}
                ),
                "bm25_status": (
                    retrieval.trace.bm25_status
                    if retrieval is not None
                    else "not_executed"
                ),
                "dense_status": (
                    retrieval.trace.dense_status
                    if retrieval is not None
                    else "not_executed"
                ),
                "planner_outcome": (
                    protocol_outcome or "candidate_or_evidence_error"
                ),
                "question": ask_attribute,
                "requested_width": result_count,
                "presented_width": len(recommendations),
                "retrieval_fallback": bool(
                    retrieval is not None and retrieval.trace.used_fallback
                ),
            }
            if (
                self.decision_policy is EXPECTED_UTILITY_DECISION_POLICY
                and protocol_decision is not None
            ):
                self._protocol_action_traces[session_id].update(
                    {
                        "world_mode": protocol_decision.trace.protocol_mode,
                        "protocol_confidence": (
                            protocol_decision.trace.confidence
                        ),
                        "out_of_pool_probability": (
                            protocol_decision.trace.out_of_pool_probability
                        ),
                        "expected_utility": protocol_decision.value,
                        "immediate_utility": (
                            protocol_decision.immediate_value
                        ),
                        "continuation_utility": (
                            protocol_decision.continuation_value
                        ),
                        "continuation_retrieval": (
                            protocol_decision.retrieval.value
                        ),
                        "runner_up_question": (
                            protocol_decision.runner_up_question
                        ),
                        "runner_up_width": protocol_decision.runner_up_width,
                        "runner_up_utility": protocol_decision.runner_up_value,
                        "simulated_reply_partitions": (
                            protocol_decision.trace.simulated_partition_count
                        ),
                        "pruned_questions": (
                            protocol_decision.trace.pruned_question_count
                        ),
                        "planner_fallback_reason": (
                            protocol_decision.trace.fallback_reason
                        ),
                    }
                )
        if slate_trace is not None:
            if slate_trace.signature_changed:
                if prior_slate_state.signature is None:
                    self._slate_initializations += 1
                else:
                    self._slate_ranking_resets += 1
            if slate_trace.stagnant_turn:
                self._slate_stagnant_turns += 1
                self._slate_unseen_selected_on_stagnant += (
                    slate_trace.unseen_selected
                )
            self._slate_repeat_backfills += slate_trace.repeat_backfills

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin} for parent_asin in recommendations
            ],
            "usage": {
                "prompt_tokens": local_prompt_tokens,
                "completion_tokens": local_completion_tokens,
            },
        }

    def _observe_expected_protocol_turn(
        self,
        session_id: str,
        state: IntentState,
        user_message: str,
        turn: int,
        *,
        intent_cacheable: bool,
    ) -> tuple[bool, str | None]:
        """Stage one recognized turn for the Phase 4 world model.

        The event log contains only observable protocol messages.  Unsupported
        prose remains on the normal broad retrieval path and cannot partially
        mutate a previously valid transcript.
        """

        try:
            from conversational_search.decision import (
                ProtocolObservation,
                parse_protocol_event,
                protocol_state_is_consistent,
                recognize_protocol_observation,
            )
            from conversational_search.protocol import ProtocolEventKind
        except Exception:
            self._protocol_consistency[session_id] = False
            return False, "candidate_or_evidence_error"

        try:
            observation = recognize_protocol_observation(user_message, turn)
        except Exception:
            observation = ProtocolObservation.UNSUPPORTED
        if observation is ProtocolObservation.UNSUPPORTED:
            self._protocol_consistency[session_id] = False
            return False, "unsupported_or_disabled"

        try:
            event = parse_protocol_event(
                user_message,
                observation,
                turn,
                asked_attribute=state.last_asked_attribute,
            )
            prior_events = self._protocol_events.get(session_id, ())
            if prior_events and event.turn <= prior_events[-1].turn:
                raise ValueError("protocol events must advance by turn")
            next_events = (*prior_events, event)
        except Exception:
            self._protocol_consistency[session_id] = False
            return False, "unsupported_or_disabled"

        self._protocol_events[session_id] = next_events
        if event.kind is ProtocolEventKind.INITIAL_TENTATIVE:
            self._protocol_override_pending[session_id] = True
        elif event.kind is ProtocolEventKind.OVERRIDE:
            self._protocol_override_pending[session_id] = False

        try:
            capable = (
                getattr(self._retriever, "protocol_evidence_capability")
                is PROTOCOL_EVIDENCE_CAPABILITY
            )
        except Exception:
            capable = False
        if not capable:
            self._protocol_consistency[session_id] = False
            return False, "capability_unavailable"

        try:
            if (
                self.protocol_catalog_policy
                is FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY
            ):
                # Full-catalog replay validates the visible transcript directly.
                # The ordinary slot-state guard rejects legal repeated ``other``
                # replies because they share one attribute, so it is deliberately
                # not the authority for this exact protocol path.
                consistent = bool(
                    self._protocol_consistency.get(session_id, False)
                    and intent_cacheable
                    and state.category
                    and state.last_turn == turn
                )
            else:
                consistent = (
                    self._protocol_consistency.get(session_id, False)
                    and intent_cacheable
                    and protocol_state_is_consistent(state, next_events, turn)
                )
        except Exception:
            consistent = False
        if not consistent:
            self._protocol_consistency[session_id] = False
            return False, "fail_open_validation"
        return True, None

    @staticmethod
    def _sanitize(items: object, limit: int) -> list[str]:
        if not isinstance(items, (list, tuple)) or limit <= 0:
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            if isinstance(item, dict):
                value = item.get("parent_asin")
            else:
                value = getattr(item, "parent_asin", item)
            if not isinstance(value, str):
                continue
            parent_asin = value.strip()
            if not parent_asin or parent_asin in seen:
                continue
            seen.add(parent_asin)
            result.append(parent_asin)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _validate_importance_aware_ranking(
        result: object,
        *,
        expected_ids: Sequence[str],
    ) -> object:
        """Validate one complete ordinal ranking before it affects output."""

        from conversational_search.requirement_satisfaction import (
            MAX_SATISFACTION_CANDIDATES,
            MAX_SATISFACTION_REQUIREMENTS,
            ImportanceAwareResult,
            ImportanceAwareStatus,
            RequirementSatisfaction,
        )
        from conversational_search.intent import RequirementImportance

        if not isinstance(result, ImportanceAwareResult):
            raise TypeError(
                "importance-aware ranking must return ImportanceAwareResult"
            )
        if not isinstance(result.status, ImportanceAwareStatus):
            raise TypeError("importance-aware ranking status is invalid")
        expected = tuple(expected_ids)
        if (
            not expected
            or len(expected) > MAX_SATISFACTION_CANDIDATES
            or len(expected) != len(set(expected))
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in expected
            )
        ):
            raise ValueError("expected importance-aware IDs are invalid")
        if (
            len(result.ranked_ids) != len(expected)
            or len(set(result.ranked_ids)) != len(expected)
            or set(result.ranked_ids) != set(expected)
        ):
            raise ValueError(
                "importance-aware ranking must be a complete input permutation"
            )
        if (
            len(result.requirements) > MAX_SATISFACTION_REQUIREMENTS
            or len(result.assessments) != len(expected)
            or tuple(item.parent_asin for item in result.assessments) != expected
        ):
            raise ValueError("importance-aware evidence alignment drifted")
        if not set(result.fully_satisfied_best_ids).issubset(expected):
            raise ValueError("importance-aware support is outside the input pool")
        for assessment in result.assessments:
            if (
                len(assessment.satisfactions) != len(result.requirements)
                or len(assessment.requirement_tier) != 9
                or any(
                    not isinstance(status, RequirementSatisfaction)
                    for status in assessment.satisfactions
                )
                or type(assessment.exact_affinity) is not int
                or not 0 <= assessment.exact_affinity <= len(result.requirements)
                or any(
                    type(value) is not int
                    or not 0 <= value <= 3 * len(result.requirements) + 3
                    for value in assessment.requirement_tier
                )
                or any(
                    assessment.requirement_tier[index] not in {0, 1}
                    for index in (0, 3, 6)
                )
                or any(
                    not 0 <= assessment.requirement_tier[index] <= 3
                    for index in (1, 4, 7)
                )
            ):
                raise ValueError("importance-aware assessment is invalid")

        trace = result.trace
        if (
            trace.candidate_count != len(expected)
            or trace.requirement_count != len(result.requirements)
            or trace.requirement_count
            != (
                trace.must_requirement_count
                + trace.should_requirement_count
                + trace.prefer_requirement_count
            )
        ):
            raise ValueError("importance-aware trace cardinality drifted")
        candidate_counts = (
            trace.must_violation_candidate_count,
            trace.must_unknown_candidate_count,
            trace.all_must_full_candidate_count,
            trace.best_requirement_tier_count,
            trace.exact_affinity_candidate_count,
        )
        requirement_counts = (
            trace.must_requirement_count,
            trace.should_requirement_count,
            trace.prefer_requirement_count,
            trace.exclusion_requirement_count,
            trace.budget_requirement_count,
            trace.profile_preference_count,
        )
        if any(
            type(value) is not int or not 0 <= value <= len(expected)
            for value in candidate_counts
        ) or any(
            type(value) is not int
            or not 0 <= value <= len(result.requirements)
            for value in requirement_counts
        ):
            raise ValueError("importance-aware trace count is out of bounds")
        if trace.best_requirement_tier_count < 1:
            raise ValueError("importance-aware best tier cannot be empty")
        must_indexes = tuple(
            index
            for index, requirement in enumerate(result.requirements)
            if requirement.importance is RequirementImportance.MUST
        )
        calculated_must_violations = sum(
            any(
                assessment.satisfactions[index]
                is RequirementSatisfaction.VIOLATED
                for index in must_indexes
            )
            for assessment in result.assessments
        )
        calculated_must_unknown = sum(
            any(
                assessment.satisfactions[index]
                is RequirementSatisfaction.UNKNOWN
                for index in must_indexes
            )
            for assessment in result.assessments
        )
        calculated_all_must_full = sum(
            all(
                assessment.satisfactions[index]
                is RequirementSatisfaction.FULL
                for index in must_indexes
            )
            for assessment in result.assessments
        )
        best_tier = max(
            assessment.requirement_tier for assessment in result.assessments
        )
        calculated_best_count = sum(
            assessment.requirement_tier == best_tier
            for assessment in result.assessments
        )
        assessment_by_id = {
            assessment.parent_asin: assessment
            for assessment in result.assessments
        }
        calculated_support = tuple(
            parent_asin
            for parent_asin in result.ranked_ids
            if assessment_by_id[parent_asin].requirement_tier == best_tier
            and all(
                assessment_by_id[parent_asin].satisfactions[index]
                is RequirementSatisfaction.FULL
                for index in must_indexes
            )
        )
        if (
            trace.must_violation_candidate_count
            != calculated_must_violations
            or trace.must_unknown_candidate_count != calculated_must_unknown
            or trace.all_must_full_candidate_count
            != calculated_all_must_full
            or trace.best_requirement_tier_count != calculated_best_count
            or trace.exact_affinity_candidate_count
            != sum(
                assessment.exact_affinity > 0
                for assessment in result.assessments
            )
            or trace.must_requirement_count != len(must_indexes)
            or trace.should_requirement_count
            != sum(
                requirement.importance is RequirementImportance.SHOULD
                for requirement in result.requirements
            )
            or trace.prefer_requirement_count
            != sum(
                requirement.importance is RequirementImportance.PREFER
                for requirement in result.requirements
            )
            or trace.exclusion_requirement_count
            != sum(
                requirement.kind == "exclusion"
                for requirement in result.requirements
            )
            or trace.budget_requirement_count
            != sum(
                requirement.attribute == "budget"
                for requirement in result.requirements
            )
            or trace.profile_preference_count
            != sum(
                requirement.kind == "profile"
                for requirement in result.requirements
            )
            or result.fully_satisfied_best_ids != calculated_support
        ):
            raise ValueError("importance-aware trace evidence drifted")
        if result.status is ImportanceAwareStatus.NO_REQUIREMENTS:
            if result.requirements or result.ranked_ids != expected:
                raise ValueError("no-requirement ranking must preserve base order")
        elif not result.requirements:
            raise ValueError("applied importance-aware ranking needs requirements")
        return result

    @staticmethod
    def _validate_exact_evidence_ranking(
        result: object,
        *,
        expected_ids: Sequence[str],
    ) -> ExactEvidenceResult:
        """Validate a complete, bounded permutation before it affects output."""

        from conversational_search.exact_evidence import (
            ExactEvidenceResult,
            ExactEvidenceStatus,
        )

        if not isinstance(result, ExactEvidenceResult):
            raise TypeError("exact evidence must return ExactEvidenceResult")
        if not isinstance(result.status, ExactEvidenceStatus):
            raise TypeError("exact evidence status is invalid")
        expected = tuple(expected_ids)
        if (
            not expected
            or len(expected) > MAX_CANDIDATE_DOCUMENTS
            or len(expected) != len(set(expected))
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in expected
            )
        ):
            raise ValueError("expected exact-evidence IDs are invalid")
        if (
            len(result.ranked_ids) != len(expected)
            or len(set(result.ranked_ids)) != len(expected)
            or set(result.ranked_ids) != set(expected)
        ):
            raise ValueError(
                "exact-evidence ranking must be a complete input permutation"
            )
        if not set(result.consistent_support_ids).issubset(expected):
            raise ValueError("exact-evidence support is outside the input pool")
        if result.trace.candidate_count != len(expected):
            raise ValueError("exact-evidence trace candidate count drifted")
        if any(
            value < 0 or value > len(expected)
            for value in (
                result.trace.category_compatible_count,
                result.trace.reply_consistent_count,
                result.trace.exclusion_violation_count,
                result.trace.consistent_support_count,
                result.trace.best_tier_count,
                result.trace.exact_phrase_candidate_count,
                result.trace.budget_compatible_count,
            )
        ):
            raise ValueError("exact-evidence trace count is out of bounds")
        if result.trace.consistent_support_count != len(
            result.consistent_support_ids
        ):
            raise ValueError("exact-evidence support count drifted")
        if result.status is ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT:
            if result.ranked_ids != expected or result.consistent_support_ids:
                raise ValueError("zero-support exact evidence must fail open")
        elif not result.consistent_support_ids:
            raise ValueError("applied exact evidence requires consistent support")
        return result

    def _apply_field_semantic_ranking(
        self,
        state: IntentState,
        base_ranked_ids: tuple[str, ...],
    ) -> tuple[tuple[str, ...], bool]:
        """Apply bounded card-atom semantics after Stage A and fail open."""

        candidate_ids = base_ranked_ids[:MAX_FIELD_SEMANTIC_CANDIDATES]
        typed_requirements = tuple(
            (requirement.attribute, requirement.value)
            for requirement in state.requirements
            if requirement.source == "free_text"
            and requirement.attribute is not None
        )[-MAX_FIELD_SEMANTIC_REQUIREMENTS:]
        exclusions = tuple(state.excluded)[-MAX_FIELD_SEMANTIC_REQUIREMENTS:]
        category = state.category
        if not typed_requirements and not exclusions and not category:
            self._record_field_semantic_outcome(
                "no_signal",
                candidate_count=len(candidate_ids),
            )
            return base_ranked_ids, True

        try:
            capable = (
                getattr(self._retriever, "field_semantic_capability")
                is FIELD_SEMANTIC_CAPABILITY
            )
        except Exception:
            capable = False
        if not capable:
            self._record_field_semantic_outcome(
                "capability_unavailable",
                candidate_count=len(candidate_ids),
            )
            return base_ranked_ids, False

        try:
            from conversational_search.field_semantic import (
                FieldSemanticResult,
                rank_field_semantic,
            )

            assessments = tuple(
                self._retriever.candidate_field_semantic_assessments(
                    candidate_ids,
                    typed_requirements,
                    exclusions,
                    category,
                )
            )
            result = rank_field_semantic(candidate_ids, assessments)
            if not isinstance(result, FieldSemanticResult):
                raise TypeError("field-semantic ranking returned an invalid result")
            if (
                len(result.ranked_ids) != len(candidate_ids)
                or len(set(result.ranked_ids)) != len(candidate_ids)
                or set(result.ranked_ids) != set(candidate_ids)
            ):
                raise ValueError(
                    "field-semantic ranking must preserve the candidate prefix"
                )
        except Exception:
            self._record_field_semantic_outcome(
                "scoring_error",
                candidate_count=len(candidate_ids),
            )
            return base_ranked_ids, False

        outcome = (
            "reordered"
            if result.status is FieldSemanticStatus.REORDERED
            else "no_signal"
            if result.status is FieldSemanticStatus.NO_SIGNAL
            else "unchanged"
        )
        self._record_field_semantic_outcome(
            outcome,
            candidate_count=len(candidate_ids),
        )
        return (*result.ranked_ids, *base_ranked_ids[len(candidate_ids) :]), True

    def _apply_exact_evidence_ranking(
        self,
        state: IntentState,
        base_ranked_ids: tuple[str, ...],
        *,
        dense_ids: Sequence[str] = (),
        dense_scores: Sequence[float] = (),
        protocol_events: Sequence[object] = (),
    ) -> _ExactRankingContext:
        """Apply one bounded exact pass before committing the ranking cache.

        The candidate set is immutable here. Any missing capability, malformed
        metadata, or invalid result fails open to the complete Stage-A order and
        disables cache storage for that turn. A validated zero-support result is
        safe to cache because it is defined to preserve the input permutation.
        """

        candidate_count = len(base_ranked_ids)
        consistent_count = 0
        try:
            exact_capable = (
                getattr(self._retriever, "protocol_evidence_capability")
                is PROTOCOL_EVIDENCE_CAPABILITY
            )
        except Exception:
            exact_capable = False
        if not exact_capable:
            self._record_exact_evidence_outcome(
                "capability_unavailable",
                candidate_count=candidate_count,
                consistent_count=0,
            )
            return _ExactRankingContext(
                base_ranked_ids,
                (),
                None,
                base_ranked_ids,
                False,
            )

        try:
            evidence = tuple(
                self._retriever.candidate_protocol_evidence(base_ranked_ids)
            )
        except Exception:
            self._record_exact_evidence_outcome(
                "evidence_error",
                candidate_count=candidate_count,
                consistent_count=0,
            )
            return _ExactRankingContext(
                base_ranked_ids,
                (),
                None,
                base_ranked_ids,
                False,
            )

        try:
            from conversational_search.exact_evidence import (
                ExactEvidenceStatus,
                rank_exact_evidence,
            )

            exact_ranking = self._validate_exact_evidence_ranking(
                rank_exact_evidence(
                    base_ranked_ids,
                    evidence,
                    state,
                    protocol_events=protocol_events,
                ),
                expected_ids=base_ranked_ids,
            )
        except Exception:
            self._record_exact_evidence_outcome(
                "validation_error",
                candidate_count=candidate_count,
                consistent_count=0,
            )
            return _ExactRankingContext(
                base_ranked_ids,
                evidence,
                None,
                base_ranked_ids,
                False,
            )

        if self.semantic_tiebreak_policy is not DISABLED_SEMANTIC_TIEBREAK_POLICY:
            baseline_exact_ranking = exact_ranking
            try:
                from conversational_search.exact_evidence import (
                    apply_dense_best_tier_tiebreak,
                )

                semantic_result = apply_dense_best_tier_tiebreak(
                    exact_ranking,
                    dense_ids,
                    dense_scores=dense_scores,
                    policy=self.semantic_tiebreak_policy,
                )
                exact_ranking = self._validate_exact_evidence_ranking(
                    semantic_result.ranking,
                    expected_ids=base_ranked_ids,
                )
                self._record_semantic_tiebreak_status(semantic_result.status)
            except Exception:
                exact_ranking = baseline_exact_ranking
                self._record_semantic_tiebreak_status(
                    None,
                    validation_fallback=True,
                )
                semantic_cacheable = False
            else:
                semantic_cacheable = True
        else:
            semantic_cacheable = True

        consistent_count = len(exact_ranking.consistent_support_ids)
        if exact_ranking.status is ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT:
            self._record_exact_evidence_outcome(
                "zero_support_fail_open",
                candidate_count=candidate_count,
                consistent_count=consistent_count,
            )
            return _ExactRankingContext(
                base_ranked_ids,
                evidence,
                exact_ranking,
                base_ranked_ids,
                semantic_cacheable,
            )

        self._record_exact_evidence_outcome(
            (
                "applied_unchanged"
                if exact_ranking.ranked_ids == base_ranked_ids
                else "applied_reordered"
            ),
            candidate_count=candidate_count,
            consistent_count=consistent_count,
        )
        return _ExactRankingContext(
            base_ranked_ids,
            evidence,
            exact_ranking,
            exact_ranking.ranked_ids,
            semantic_cacheable,
        )

    def _apply_importance_aware_ranking(
        self,
        state: IntentState,
        base_ranked_ids: tuple[str, ...],
        *,
        bm25_ids: Sequence[str],
        documents: Sequence[CandidateDocument],
        profile_prior: ProfilePrior,
    ) -> tuple[tuple[str, ...], bool]:
        """Apply one bounded ordinal pass, failing open to the Stage-A order."""

        candidate_count = len(base_ranked_ids)
        try:
            capable = (
                getattr(self._retriever, "protocol_evidence_capability")
                is PROTOCOL_EVIDENCE_CAPABILITY
            )
        except Exception:
            capable = False
        if not capable:
            self._record_importance_satisfaction_outcome(
                "capability_unavailable",
                candidate_count=candidate_count,
            )
            return base_ranked_ids, False

        try:
            evidence = tuple(
                self._retriever.candidate_protocol_evidence(base_ranked_ids)
            )
        except Exception:
            self._record_importance_satisfaction_outcome(
                "evidence_error",
                candidate_count=candidate_count,
            )
            return base_ranked_ids, False

        try:
            from conversational_search.requirement_satisfaction import (
                ImportanceAwareStatus,
                rank_importance_aware_satisfaction,
            )

            result = self._validate_importance_aware_ranking(
                rank_importance_aware_satisfaction(
                    base_ranked_ids,
                    bm25_ids,
                    evidence,
                    documents,
                    state,
                    profile_prior=profile_prior,
                ),
                expected_ids=base_ranked_ids,
            )
        except Exception:
            self._record_importance_satisfaction_outcome(
                "validation_error",
                candidate_count=candidate_count,
            )
            return base_ranked_ids, False

        outcome = (
            "no_requirements"
            if result.status is ImportanceAwareStatus.NO_REQUIREMENTS
            else "applied_unchanged"
            if result.ranked_ids == base_ranked_ids
            else "applied_reordered"
        )
        self._record_importance_satisfaction_outcome(
            outcome,
            candidate_count=candidate_count,
            requirement_count=result.trace.requirement_count,
            must_violation_count=result.trace.must_violation_candidate_count,
            must_unknown_count=result.trace.must_unknown_candidate_count,
            all_must_full_count=result.trace.all_must_full_candidate_count,
            best_tier_count=result.trace.best_requirement_tier_count,
            budget_requirement_count=result.trace.budget_requirement_count,
            profile_preference_count=result.trace.profile_preference_count,
        )
        return result.ranked_ids, True

    @staticmethod
    def _validate_profile_ranking(result: object) -> ProfileRankingResult:
        if not isinstance(result, ProfileRankingResult):
            raise TypeError("profile reranker must return ProfileRankingResult")
        if not isinstance(result.ranking, RankingResult):
            raise TypeError("profile reranker must contain RankingResult")
        if not isinstance(result.status, ProfileResidualStatus):
            raise TypeError("profile reranker must contain ProfileResidualStatus")
        return result

    @staticmethod
    def _validate_bm25_rescue_ranking(
        result: object,
        *,
        state: IntentState,
        fused_ids: Sequence[str],
    ) -> Bm25RescueRankingResult:
        if not isinstance(result, Bm25RescueRankingResult):
            raise TypeError(
                "BM25 rescue reranker must return Bm25RescueRankingResult"
            )
        if not isinstance(result.ranking, RankingResult):
            raise TypeError("BM25 rescue reranker must contain RankingResult")
        if not isinstance(result.status, Bm25RescueStatus):
            raise TypeError("BM25 rescue reranker must contain Bm25RescueStatus")
        if not isinstance(result.profile_status, ProfileResidualStatus):
            raise TypeError(
                "BM25 rescue reranker must contain ProfileResidualStatus"
            )
        counts = (result.requested_theme_count, result.represented_theme_count)
        if any(type(value) is not int or not 0 <= value <= 10 for value in counts):
            raise ValueError("BM25 rescue profile counts must be integers in [0, 10]")
        if result.represented_theme_count > result.requested_theme_count:
            raise ValueError("represented profile themes cannot exceed requested themes")

        expected_ids = tuple(fused_ids)
        ranking = result.ranking
        trace = ranking.trace
        if not isinstance(trace, RankingTrace):
            raise TypeError("BM25 rescue reranker must contain RankingTrace")
        if (
            not expected_ids
            or len(set(expected_ids)) != len(expected_ids)
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in expected_ids
            )
        ):
            raise ValueError("expected fused IDs must be non-empty and unique")
        if (
            trace.input_ids != expected_ids
            or trace.output_ids != ranking.ranked_ids
            or len(ranking.ranked_ids) != len(expected_ids)
            or len(set(ranking.ranked_ids)) != len(expected_ids)
            or set(ranking.ranked_ids) != set(expected_ids)
        ):
            raise ValueError(
                "BM25 rescue ranking must be a complete fused-ID permutation"
            )

        completeness = intent_completeness(state)
        expected_beta = 0.20 + 0.25 * completeness
        if (
            isinstance(trace.beta, bool)
            or not isinstance(trace.beta, (int, float))
            or not math.isfinite(trace.beta)
            or trace.beta != expected_beta
        ):
            raise ValueError("BM25 rescue beta must match intent completeness")
        if (
            type(trace.observable_clause_count) is not int
            or not 0 <= trace.observable_clause_count <= MAX_CLAUSES
        ):
            raise ValueError("BM25 rescue clause count is out of bounds")
        return result

    @staticmethod
    def _validate_route_redundancy_ranking(
        result: object,
        *,
        state: IntentState,
        fused_ids: Sequence[str],
    ) -> RouteRedundancyRankingResult:
        if not isinstance(result, RouteRedundancyRankingResult):
            raise TypeError(
                "route-redundancy reranker must return "
                "RouteRedundancyRankingResult"
            )
        if not isinstance(result.ranking, RankingResult):
            raise TypeError(
                "route-redundancy reranker must contain RankingResult"
            )
        if not isinstance(result.status, RouteRedundancyStatus):
            raise TypeError(
                "route-redundancy reranker must contain RouteRedundancyStatus"
            )
        if not isinstance(result.profile_status, ProfileResidualStatus):
            raise TypeError(
                "route-redundancy reranker must contain ProfileResidualStatus"
            )
        counts = (result.requested_theme_count, result.represented_theme_count)
        if any(type(value) is not int or not 0 <= value <= 10 for value in counts):
            raise ValueError(
                "route-redundancy profile counts must be integers in [0, 10]"
            )
        if result.represented_theme_count > result.requested_theme_count:
            raise ValueError(
                "represented profile themes cannot exceed requested themes"
            )

        expected_ids = tuple(fused_ids)
        ranking = result.ranking
        trace = ranking.trace
        if not isinstance(trace, RankingTrace):
            raise TypeError(
                "route-redundancy reranker must contain RankingTrace"
            )
        if (
            not expected_ids
            or len(set(expected_ids)) != len(expected_ids)
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in expected_ids
            )
        ):
            raise ValueError("expected fused IDs must be non-empty and unique")
        if (
            trace.input_ids != expected_ids
            or trace.output_ids != ranking.ranked_ids
            or len(ranking.ranked_ids) != len(expected_ids)
            or len(set(ranking.ranked_ids)) != len(expected_ids)
            or set(ranking.ranked_ids) != set(expected_ids)
        ):
            raise ValueError(
                "route-redundancy ranking must be a complete fused-ID permutation"
            )

        expected_beta = 0.20 + 0.25 * intent_completeness(state)
        if (
            isinstance(trace.beta, bool)
            or not isinstance(trace.beta, (int, float))
            or not math.isfinite(trace.beta)
            or trace.beta != expected_beta
        ):
            raise ValueError(
                "route-redundancy beta must match intent completeness"
            )
        if (
            type(trace.observable_clause_count) is not int
            or not 0 <= trace.observable_clause_count <= MAX_CLAUSES
        ):
            raise ValueError(
                "route-redundancy clause count is out of bounds"
            )
        return result

    @staticmethod
    def _validate_intent_epoch_slate_selection(
        result: object,
        *,
        prior_state: SlateState,
        signature: tuple[object, ...],
        ranked_ids: Sequence[str],
        limit: int,
    ) -> IntentEpochSlateSelection:
        if not isinstance(result, IntentEpochSlateSelection):
            raise TypeError(
                "intent-epoch selector must return IntentEpochSlateSelection"
            )
        if not isinstance(result.selection, SlateSelection):
            raise TypeError(
                "intent-epoch selector must contain SlateSelection"
            )
        if not isinstance(result.status, IntentEpochSlateStatus):
            raise TypeError(
                "intent-epoch selector must contain IntentEpochSlateStatus"
            )
        if (
            type(result.eligible_prior_shown) is not int
            or not 0 <= result.eligible_prior_shown <= MAX_SLATE_CANDIDATES
        ):
            raise ValueError("eligible prior shown count is out of bounds")
        if not isinstance(prior_state, SlateState):
            raise TypeError("prior_state must be SlateState")
        if not isinstance(signature, tuple):
            raise TypeError("signature must be a tuple")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("limit must be an integer from 1 through 10")

        pool = tuple(ranked_ids)
        if (
            not pool
            or len(pool) > MAX_SLATE_CANDIDATES
            or len(pool) != len(set(pool))
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in pool
            )
        ):
            raise ValueError("ranked_ids must be a non-empty unique pool")
        selection = result.selection
        selected = selection.selected_ids
        expected_count = min(limit, len(pool))
        if (
            len(selected) != expected_count
            or len(selected) != len(set(selected))
            or not set(selected).issubset(pool)
        ):
            raise ValueError(
                "intent-epoch slate must be a complete unique pool subset"
            )
        if not isinstance(selection.state, SlateState):
            raise TypeError("intent-epoch selection state must be SlateState")
        shown = selection.state.shown_ids
        if (
            selection.state.signature != signature
            or len(shown) > MAX_SLATE_CANDIDATES
            or len(shown) != len(set(shown))
            or not set(shown).issubset(pool)
        ):
            raise ValueError("intent-epoch state is invalid or unbounded")
        trace = selection.trace
        if not isinstance(trace, SlateTrace):
            raise TypeError("intent-epoch selection trace must be SlateTrace")
        signature_changed = prior_state.signature != signature
        if (
            trace.signature_changed is not signature_changed
            or trace.stagnant_turn is signature_changed
            or type(trace.unseen_selected) is not int
            or type(trace.repeat_backfills) is not int
            or trace.unseen_selected < 0
            or trace.repeat_backfills < 0
            or trace.unseen_selected + trace.repeat_backfills
            != len(selected)
        ):
            raise ValueError("intent-epoch trace is inconsistent")

        baseline = select_slate(
            STAGNATION_AWARE_SLATE_POLICY,
            prior_state,
            signature,
            pool,
            limit,
        )
        if result.status is not IntentEpochSlateStatus.CARRIED:
            if selection != baseline:
                raise ValueError(
                    "neutral intent-epoch outcome must equal protected slate"
                )
            if prior_state.signature is None:
                expected_status = IntentEpochSlateStatus.FIRST
            elif prior_state.signature == signature:
                expected_status = IntentEpochSlateStatus.UNCHANGED
            else:
                prior_epoch = (
                    prior_state.signature[0]
                    if prior_state.signature
                    else None
                )
                current_epoch = signature[0] if signature else None
                if (
                    type(prior_epoch) is not int
                    or prior_epoch < 0
                    or type(current_epoch) is not int
                    or current_epoch < 0
                ):
                    expected_status = (
                        IntentEpochSlateStatus.VALIDATION_FALLBACK
                    )
                elif prior_epoch != current_epoch:
                    expected_status = IntentEpochSlateStatus.EPOCH_RESET
                else:
                    expected_status = IntentEpochSlateStatus.CARRIED
            if result.status is not expected_status:
                raise ValueError("intent-epoch status does not match transition")
            return result

        if prior_state.signature is None or not signature_changed:
            raise ValueError("history carry requires a changed prior signature")
        prior_epoch = prior_state.signature[0]
        current_epoch = signature[0] if signature else None
        if (
            type(prior_epoch) is not int
            or prior_epoch < 0
            or type(current_epoch) is not int
            or current_epoch < 0
            or prior_epoch != current_epoch
        ):
            raise ValueError("history carry requires one valid intent epoch")

        pool_set = frozenset(pool)
        prior_shown = tuple(
            dict.fromkeys(
                parent_asin
                for parent_asin in prior_state.shown_ids
                if parent_asin in pool_set
            )
        )
        if result.eligible_prior_shown != len(prior_shown):
            raise ValueError("eligible prior shown count drifted")
        prior_set = frozenset(prior_shown)
        unseen = tuple(
            parent_asin for parent_asin in pool if parent_asin not in prior_set
        )
        expected_selected = list(unseen[:limit])
        expected_unseen = len(expected_selected)
        if len(expected_selected) < limit:
            expected_set = frozenset(expected_selected)
            expected_selected.extend(
                parent_asin
                for parent_asin in pool
                if parent_asin not in expected_set
            )
            del expected_selected[limit:]
        expected_shown = tuple(
            dict.fromkeys((*prior_shown, *expected_selected))
        )
        if (
            selected != tuple(expected_selected)
            or shown != expected_shown
            or trace.unseen_selected != expected_unseen
            or trace.repeat_backfills != len(expected_selected) - expected_unseen
        ):
            raise ValueError("intent-epoch candidate selection drifted")
        return result

    @staticmethod
    def _validate_semantic_lexical_retrieval(
        result: object,
    ) -> SemanticLexicalRetrievalResult:
        """Validate that private dense evidence cannot enter an exposed route."""

        if not isinstance(result, SemanticLexicalRetrievalResult):
            raise TypeError(
                "semantic rescue must return SemanticLexicalRetrievalResult"
            )
        if not isinstance(result.trace, RetrievalTrace):
            raise TypeError("semantic rescue must contain RetrievalTrace")
        semantic_trace = result.semantic_trace
        if not isinstance(semantic_trace, SemanticLexicalRescueTrace):
            raise TypeError("semantic rescue must contain its bounded trace")
        if not isinstance(
            semantic_trace.status,
            SemanticLexicalRescueStatus,
        ):
            raise TypeError("semantic rescue status is invalid")
        routes = (
            result.trace.bm25_ids,
            result.trace.dense_ids,
            result.trace.fused_ids,
            semantic_trace.base_bm25_ids,
            semantic_trace.retry_bm25_ids,
        )
        if any(
            not isinstance(route, tuple)
            or len(route) != len(set(route))
            or any(not isinstance(value, str) or not value for value in route)
            for route in routes
        ):
            raise ValueError("semantic rescue routes must be unique ID tuples")
        if (
            result.trace.dense_ids
            or result.trace.dense_status != "skipped"
            or result.trace.fused_ids != result.trace.bm25_ids
            or len(result.trace.bm25_ids) > ROUTE_LIMIT
            or len(semantic_trace.base_bm25_ids) > ROUTE_LIMIT
            or len(semantic_trace.retry_bm25_ids) > ROUTE_LIMIT
        ):
            raise ValueError("semantic rescue exposed a non-lexical route")
        counts = (
            semantic_trace.private_dense_candidate_count,
            semantic_trace.compatible_dense_candidate_count,
            semantic_trace.expansion_term_count,
            semantic_trace.retry_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("semantic rescue counts must be non-negative integers")
        if (
            semantic_trace.private_dense_candidate_count > ROUTE_LIMIT
            or semantic_trace.compatible_dense_candidate_count
            > semantic_trace.private_dense_candidate_count
            or semantic_trace.expansion_term_count
            > MAX_SEMANTIC_EXPANSION_TERMS
            or semantic_trace.retry_count not in {0, 1}
        ):
            raise ValueError("semantic rescue count exceeds its bound")
        if result.trace.used_fallback:
            if result.trace.fused_ids:
                raise ValueError("semantic fallback must have an empty lexical route")
        elif result.recommendations != result.trace.fused_ids[
            : len(result.recommendations)
        ]:
            raise ValueError("recommendations must be a lexical-route prefix")
        if semantic_trace.status is SemanticLexicalRescueStatus.APPLIED:
            if (
                semantic_trace.retry_count != 1
                or semantic_trace.retry_bm25_status != "ok"
                or not semantic_trace.retry_bm25_ids
                or result.trace.bm25_ids != semantic_trace.retry_bm25_ids
            ):
                raise ValueError("applied semantic rescue is internally inconsistent")
        if semantic_trace.status is SemanticLexicalRescueStatus.NOT_NEEDED:
            if (
                semantic_trace.private_dense_candidate_count
                or semantic_trace.retry_count
                or result.trace.bm25_ids != semantic_trace.base_bm25_ids
            ):
                raise ValueError("neutral semantic rescue executed an extra route")
        return result

    @staticmethod
    def _validate_evidence_exposure_decision(
        result: object,
        *,
        ranked_ids: tuple[str, ...],
        requested_top_k: int,
        current_turn: int,
    ) -> EvidenceExposureDecision:
        if not isinstance(result, EvidenceExposureDecision):
            raise TypeError("exposure planner must return EvidenceExposureDecision")
        if not isinstance(result.status, EvidenceExposureStatus):
            raise TypeError("exposure decision status is invalid")
        if (
            not isinstance(result.presentation_ids, tuple)
            or len(result.presentation_ids) != len(set(result.presentation_ids))
            or not set(result.presentation_ids).issubset(ranked_ids)
            or type(result.width) is not int
            or not 0 <= result.width <= requested_top_k
            or result.width > len(result.presentation_ids)
            or type(result.plausible_count) is not int
            or result.plausible_count < 0
        ):
            raise ValueError("exposure decision is outside its bounds")
        if result.question is not None and result.question not in QUESTION_TEXT:
            raise ValueError("exposure question is unsupported")
        if result.status is EvidenceExposureStatus.TOP3_CONFIDENT:
            if (
                not result.presentation_ids
                or result.presentation_ids
                != ranked_ids[: len(result.presentation_ids)]
                or result.width != len(result.presentation_ids)
                or result.width > 3
                or result.question is not None
            ):
                raise ValueError("confident exposure does not fit its prefix")
        elif result.status is EvidenceExposureStatus.QUESTION_WITH_PREFIX:
            expected_width = min(3, requested_top_k, result.plausible_count)
            if (
                result.plausible_count <= 3
                and result.plausible_count <= requested_top_k
            ) or (
                not result.presentation_ids
                or result.presentation_ids != ranked_ids[:expected_width]
                or result.width != expected_width
                or result.width != len(result.presentation_ids)
                or result.question is None
                or current_turn >= 10
            ):
                raise ValueError("question prefix is outside its safe bound")
        elif result.status is EvidenceExposureStatus.AMBIGUOUS_TOP1_PREVIEW:
            if (
                not result.presentation_ids
                or result.presentation_ids != ranked_ids[:1]
                or result.width != 1
                or len(result.presentation_ids) != 1
                or result.question != "other"
                or result.plausible_count < 1
                or current_turn != 1
            ):
                raise ValueError("ambiguous preview is outside its safe bound")
        elif result.status is EvidenceExposureStatus.POSTERIOR_SINGLETON:
            if (
                result.presentation_ids != ranked_ids[:1]
                or result.width != 1
                or result.plausible_count != 1
                or result.question is not None
            ):
                raise ValueError("posterior singleton is outside its safe bound")
        elif result.status is EvidenceExposureStatus.POSTERIOR_PROBE:
            if (
                result.presentation_ids != ranked_ids[:1]
                or result.width != 1
                or result.plausible_count <= 1
                or result.question != "other"
                or current_turn >= 10
            ):
                raise ValueError("posterior probe is outside its safe bound")
        elif result.status is EvidenceExposureStatus.POSTERIOR_REPLY_TREE:
            if (
                result.presentation_ids != ranked_ids
                or not 1 <= result.width <= min(
                    requested_top_k,
                    len(ranked_ids),
                    result.plausible_count,
                )
                or result.question != "other"
                or current_turn >= 10
            ):
                raise ValueError("reply-tree exposure is outside its safe bound")
        elif result.status is EvidenceExposureStatus.POSTERIOR_BATCH:
            if (
                result.presentation_ids != ranked_ids
                or result.width != min(requested_top_k, len(ranked_ids))
                or result.question is not None
            ):
                raise ValueError("posterior batch is outside its safe bound")
        elif result.status is EvidenceExposureStatus.POSTERIOR_ENUMERATION:
            if (
                result.presentation_ids != ranked_ids
                or not 1 <= result.width <= min(
                    requested_top_k,
                    len(ranked_ids),
                    result.plausible_count,
                )
                or result.question is not None
                or current_turn >= 10
            ):
                raise ValueError(
                    "posterior enumeration is outside its safe bound"
                )
        else:
            if result.presentation_ids != ranked_ids:
                raise ValueError("non-confident exposure changed the ranking pool")
        if result.status is EvidenceExposureStatus.QUESTION_WITHHELD:
            if result.width != 0 or result.question is None or current_turn >= 10:
                raise ValueError("withheld exposure requires a live question")
        elif (
            result.status
            not in {
                EvidenceExposureStatus.QUESTION_WITH_PREFIX,
                EvidenceExposureStatus.AMBIGUOUS_TOP1_PREVIEW,
                EvidenceExposureStatus.POSTERIOR_PROBE,
                EvidenceExposureStatus.POSTERIOR_REPLY_TREE,
            }
            and result.question is not None
        ):
            raise ValueError("only question exposure may force a question")
        if (
            result.status is EvidenceExposureStatus.FINAL_TURN
            and current_turn < 10
        ):
            raise ValueError("final-turn exposure was selected too early")
        return result

    @staticmethod
    def _validate_requirement_probe_retrieval(
        result: object,
    ) -> RetrievalResult:
        """Validate the bounded additive contract before trusting probe output."""

        if not isinstance(result, RequirementProbeRetrievalResult):
            raise TypeError(
                "candidate search must return RequirementProbeRetrievalResult"
            )
        trace = result.trace
        if not isinstance(trace, RetrievalTrace):
            raise TypeError("retrieval result must contain RetrievalTrace")
        probe_trace = result.probe_trace
        if not isinstance(probe_trace, RequirementProbeTrace):
            raise TypeError("candidate result must contain RequirementProbeTrace")
        status = probe_trace.status
        if status not in REQUIREMENT_PROBE_STATUSES or status == "disabled":
            raise ValueError("candidate retrieval returned an invalid probe status")
        query_count = probe_trace.query_count
        if (
            type(query_count) is not int
            or not 0 <= query_count <= MAX_REQUIREMENT_PROBES
        ):
            raise ValueError("probe query count is out of bounds")

        routes = (
            probe_trace.base_bm25_ids,
            probe_trace.supplemental_ids,
            trace.bm25_ids,
            trace.dense_ids,
            trace.fused_ids,
        )
        if any(
            not isinstance(route, tuple)
            or len(route) != len(set(route))
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in route
            )
            for route in routes
        ):
            raise ValueError("probe retrieval routes must be unique ID tuples")
        base_bm25_ids = probe_trace.base_bm25_ids
        probe_ids = probe_trace.supplemental_ids
        if (
            len(base_bm25_ids) > ROUTE_LIMIT
            or len(trace.dense_ids) > ROUTE_LIMIT
            or len(trace.bm25_ids) > MAX_CANDIDATE_DOCUMENTS
            or len(trace.fused_ids) > MAX_CANDIDATE_DOCUMENTS
        ):
            raise ValueError("probe retrieval route exceeds its bound")

        incumbent = frozenset((*base_bm25_ids, *trace.dense_ids))
        if (
            set(probe_ids) & incumbent
            or len(probe_ids) > MAX_CANDIDATE_DOCUMENTS - len(incumbent)
        ):
            raise ValueError("probe supplements overlap or exceed capacity")
        if status == "ok":
            if (
                not probe_ids
                or not 1 <= query_count <= MAX_REQUIREMENT_PROBES
                or trace.bm25_ids != (*base_bm25_ids, *probe_ids)
            ):
                raise ValueError("successful probe output is inconsistent")
        else:
            if probe_ids or trace.bm25_ids != base_bm25_ids:
                raise ValueError("neutral probe output changed protected BM25")
            if status in {"empty", "no_additions"}:
                if not 1 <= query_count <= MAX_REQUIREMENT_PROBES:
                    raise ValueError("executed probe count is inconsistent")
            elif status in {"no_eligible", "capacity", "unavailable"}:
                if query_count != 0:
                    raise ValueError("neutral probe must not report executions")

        effective_union = frozenset((*trace.bm25_ids, *trace.dense_ids))
        route_statuses = {
            "bm25": (trace.bm25_status, trace.bm25_ids),
            "dense": (trace.dense_status, trace.dense_ids),
        }
        for route_name, (route_status, route_ids) in route_statuses.items():
            if route_status == "ok":
                if not route_ids:
                    raise ValueError(f"{route_name} ok status has an empty route")
            elif route_status in {"empty", "unavailable", "error", "skipped"}:
                if route_ids:
                    raise ValueError(
                        f"{route_name} non-ok status has a non-empty route"
                    )
            else:
                raise ValueError(f"{route_name} status is invalid")
        if status == "ok" and trace.bm25_status != "ok":
            raise ValueError("successful supplements require effective BM25 ok")
        if trace.used_fallback:
            if effective_union or trace.fused_ids:
                raise ValueError("retrieval fallback contains ranked routes")
        elif frozenset(trace.fused_ids) != effective_union:
            raise ValueError("fused route does not equal the effective union")
        if trace.fused_ids:
            recommendations = result.recommendations
            if (
                not isinstance(recommendations, tuple)
                or recommendations
                != trace.fused_ids[: len(recommendations)]
            ):
                raise ValueError("recommendations are not a fused-route prefix")
        return result

    def _record_requirement_probe_status(
        self,
        status: str,
        *,
        query_count: int = 0,
        additions: int = 0,
        validation_fallback: bool = False,
    ) -> None:
        if status not in REQUIREMENT_PROBE_STATUSES:
            raise ValueError("unknown requirement-probe status")
        if (
            type(query_count) is not int
            or not 0 <= query_count <= MAX_REQUIREMENT_PROBES
            or type(additions) is not int
            or not 0 <= additions <= MAX_CANDIDATE_DOCUMENTS
            or not isinstance(validation_fallback, bool)
        ):
            raise ValueError("requirement-probe telemetry is out of bounds")
        counts = getattr(self, "_requirement_probe_counts", None)
        if (
            type(counts) is not list
            or len(counts) != 12
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("requirement-probe counter state is invalid")
        counts[0] += 1
        counts[REQUIREMENT_PROBE_STATUSES.index(status) + 1] += 1
        counts[9] += query_count
        counts[10] += additions
        counts[11] += int(validation_fallback)

    def _record_semantic_rescue_status(
        self,
        status: SemanticLexicalRescueStatus | None,
        *,
        validation_fallback: bool = False,
    ) -> None:
        if status is not None and not isinstance(
            status,
            SemanticLexicalRescueStatus,
        ):
            raise TypeError("semantic rescue status is invalid")
        if type(validation_fallback) is not bool:
            raise TypeError("validation_fallback must be a boolean")
        counts = getattr(self, "_semantic_rescue_counts", None)
        expected_length = len(SemanticLexicalRescueStatus) + 2
        if (
            type(counts) is not list
            or len(counts) != expected_length
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("semantic rescue counter state is invalid")
        counts[0] += 1
        if status is not None:
            counts[list(SemanticLexicalRescueStatus).index(status) + 1] += 1
        counts[-1] += int(validation_fallback)

    def _record_evidence_exposure_status(
        self,
        status: EvidenceExposureStatus,
        *,
        validation_fallback: bool = False,
    ) -> None:
        if not isinstance(status, EvidenceExposureStatus):
            raise TypeError("evidence exposure status is invalid")
        if type(validation_fallback) is not bool:
            raise TypeError("validation_fallback must be a boolean")
        counts = getattr(self, "_evidence_exposure_counts", None)
        expected_length = len(EvidenceExposureStatus) + 3
        if (
            type(counts) is not list
            or len(counts) != expected_length
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("evidence exposure counter state is invalid")
        counts[0] += 1
        counts[list(EvidenceExposureStatus).index(status) + 1] += 1
        counts[-2] += int(
            status is EvidenceExposureStatus.QUESTION_WITHHELD
        )
        counts[-1] += int(validation_fallback)

    def _record_exact_evidence_outcome(
        self,
        outcome: str,
        *,
        candidate_count: int,
        consistent_count: int,
    ) -> None:
        if outcome not in EXACT_EVIDENCE_OUTCOMES:
            raise ValueError("unknown exact-evidence outcome")
        if (
            type(candidate_count) is not int
            or not 0 <= candidate_count <= MAX_CANDIDATE_DOCUMENTS
            or type(consistent_count) is not int
            or not 0 <= consistent_count <= candidate_count
        ):
            raise ValueError("exact-evidence counts are out of bounds")
        counts = getattr(self, "_exact_evidence_counts", None)
        if (
            type(counts) is not list
            or len(counts) != 9
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("exact-evidence counter state is invalid")
        counts[0] += 1
        counts[EXACT_EVIDENCE_OUTCOMES.index(outcome) + 1] += 1
        counts[7] += candidate_count
        counts[8] += consistent_count

    def _record_field_semantic_outcome(
        self,
        outcome: str,
        *,
        candidate_count: int,
    ) -> None:
        outcomes = (
            "reordered",
            "unchanged",
            "no_signal",
            "capability_unavailable",
            "scoring_error",
        )
        if outcome not in outcomes:
            raise ValueError("unknown field-semantic outcome")
        if (
            type(candidate_count) is not int
            or not 0 <= candidate_count <= MAX_FIELD_SEMANTIC_CANDIDATES
        ):
            raise ValueError("field-semantic candidate count is out of bounds")
        counts = getattr(self, "_field_semantic_counts", None)
        if (
            type(counts) is not list
            or len(counts) != 7
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("field-semantic counter state is invalid")
        counts[0] += 1
        counts[outcomes.index(outcome) + 1] += 1
        counts[6] += candidate_count

    def _record_semantic_tiebreak_status(
        self,
        status: SemanticTieBreakStatus | None,
        *,
        validation_fallback: bool = False,
    ) -> None:
        if status is not None and not isinstance(status, SemanticTieBreakStatus):
            raise TypeError("semantic tie-break status is invalid")
        if type(validation_fallback) is not bool:
            raise TypeError("validation_fallback must be a boolean")
        counts = getattr(self, "_semantic_tiebreak_counts", None)
        expected_length = len(SemanticTieBreakStatus) + 2
        if (
            type(counts) is not list
            or len(counts) != expected_length
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("semantic tie-break counter state is invalid")
        counts[0] += 1
        if status is not None:
            counts[list(SemanticTieBreakStatus).index(status) + 1] += 1
        counts[-1] += int(validation_fallback)

    def _record_importance_satisfaction_outcome(
        self,
        outcome: str,
        *,
        candidate_count: int,
        requirement_count: int = 0,
        must_violation_count: int = 0,
        must_unknown_count: int = 0,
        all_must_full_count: int = 0,
        best_tier_count: int = 0,
        budget_requirement_count: int = 0,
        profile_preference_count: int = 0,
    ) -> None:
        """Record fixed-cardinality, aggregate-only ordinal evidence health."""

        if outcome not in IMPORTANCE_SATISFACTION_OUTCOMES:
            raise ValueError("unknown importance-satisfaction outcome")
        candidate_counts = (
            candidate_count,
            must_violation_count,
            must_unknown_count,
            all_must_full_count,
            best_tier_count,
        )
        requirement_counts = (
            requirement_count,
            budget_requirement_count,
            profile_preference_count,
        )
        if any(
            type(value) is not int
            or not 0 <= value <= MAX_CANDIDATE_DOCUMENTS
            for value in candidate_counts
        ) or any(
            type(value) is not int or not 0 <= value <= 64
            for value in requirement_counts
        ):
            raise ValueError("importance-satisfaction counts are out of bounds")
        counts = getattr(self, "_importance_satisfaction_counts", None)
        if (
            type(counts) is not list
            or len(counts) != 15
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("importance-satisfaction counter state is invalid")
        counts[0] += 1
        counts[IMPORTANCE_SATISFACTION_OUTCOMES.index(outcome) + 1] += 1
        counts[7] += candidate_count
        counts[8] += requirement_count
        counts[9] += must_violation_count
        counts[10] += must_unknown_count
        counts[11] += all_must_full_count
        counts[12] += best_tier_count
        counts[13] += budget_requirement_count
        counts[14] += profile_preference_count

    def _record_protocol_decision_outcome(
        self,
        outcome: str,
        *,
        requested_count: int,
        presented_count: int,
        question: str | None,
    ) -> None:
        """Record one fixed-cardinality, ID-free candidate-policy outcome."""

        if outcome not in PROTOCOL_DECISION_OUTCOMES:
            raise ValueError("unknown protocol-decision outcome")
        if (
            type(requested_count) is not int
            or not 0 <= requested_count <= 10
            or type(presented_count) is not int
            or not 0 <= presented_count <= requested_count
        ):
            raise ValueError("protocol-decision widths are out of bounds")
        question_key = "none" if question is None else question
        if question_key not in PROTOCOL_QUESTION_ACTIONS:
            raise ValueError("protocol-decision question is invalid")
        counts = getattr(self, "_protocol_decision_counts", None)
        if (
            type(counts) is not list
            or len(counts) != 9
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("protocol-decision counter state is invalid")
        counts[PROTOCOL_DECISION_OUTCOMES.index(outcome) + 1] += 1
        question_counts = getattr(self, "_protocol_question_counts", None)
        width_counts = getattr(self, "_protocol_width_counts", None)
        if (
            type(question_counts) is not list
            or len(question_counts) != len(PROTOCOL_QUESTION_ACTIONS)
            or any(type(value) is not int or value < 0 for value in question_counts)
            or type(width_counts) is not list
            or len(width_counts) != 11
            or any(type(value) is not int or value < 0 for value in width_counts)
        ):
            raise RuntimeError("protocol-decision action counters are invalid")
        question_counts[PROTOCOL_QUESTION_ACTIONS.index(question_key)] += 1
        width_counts[presented_count] += 1
        self._protocol_requested_total += requested_count
        self._protocol_presented_total += presented_count

    def _rerank_exact_phase9(
        self,
        state: IntentState,
        documents: Sequence[CandidateDocument],
        retrieval: RetrievalResult,
        route_weights: RouteWeights,
        profile_prior: ProfilePrior,
    ) -> tuple[RankingResult, bool]:
        """Return exact Phase 9 ordering and whether it is safe to cache."""

        trace = retrieval.trace
        if self._profile_policy is ProfilePolicy.DISABLED:
            return (
                rerank_stage_a(
                    state,
                    documents,
                    bm25_ids=trace.bm25_ids,
                    dense_ids=trace.dense_ids,
                    fused_ids=trace.fused_ids,
                    route_weights=route_weights,
                ),
                True,
            )
        try:
            profile_ranking = self._validate_profile_ranking(
                rerank_stage_a_with_profile(
                    state,
                    documents,
                    bm25_ids=trace.bm25_ids,
                    dense_ids=trace.dense_ids,
                    fused_ids=trace.fused_ids,
                    route_weights=route_weights,
                    profile_prior=profile_prior,
                    profile_policy=self._profile_policy,
                )
            )
        except Exception:
            self._parsing_or_scoring_fallbacks += 1
            return (
                rerank_stage_a(
                    state,
                    documents,
                    bm25_ids=trace.bm25_ids,
                    dense_ids=trace.dense_ids,
                    fused_ids=trace.fused_ids,
                    route_weights=route_weights,
                ),
                False,
            )

        self._record_profile_status(profile_ranking.status)
        return (
            profile_ranking.ranking,
            profile_ranking.status is not ProfileResidualStatus.SCORING_FALLBACK,
        )

    def _record_bm25_rescue_status(self, status: Bm25RescueStatus) -> None:
        if status is Bm25RescueStatus.ZERO_COMPLETENESS:
            self._bm25_rescue_zero_completeness += 1
        elif status is Bm25RescueStatus.EMPTY_BM25:
            self._bm25_rescue_unavailable_or_empty += 1
        elif status is Bm25RescueStatus.NO_POSITIVE_UPLIFT:
            self._bm25_rescue_no_positive_uplift += 1
        elif status is Bm25RescueStatus.CONSTANT_UPLIFT:
            self._bm25_rescue_constant_uplift += 1
        elif status is Bm25RescueStatus.UNCHANGED_ORDER:
            self._bm25_rescue_unchanged_order += 1
        elif status is Bm25RescueStatus.REORDERED:
            self._bm25_rescue_successful_reorders += 1
        elif status is Bm25RescueStatus.SCORING_FALLBACK:
            self._bm25_rescue_fallbacks += 1
        else:
            raise TypeError("status must be Bm25RescueStatus")

    def _record_route_redundancy_status(
        self,
        status: RouteRedundancyStatus,
    ) -> None:
        counts = getattr(self, "_route_redundancy_counts", None)
        if counts is None:
            # Allocate one fixed seven-counter vector lazily so both protected
            # and candidate constructors retain the exact Phase 9 shape/cost.
            counts = [0] * 7
            self._route_redundancy_counts = counts
        if (
            type(counts) is not list
            or len(counts) != 7
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("route-redundancy counter state is invalid")
        counts[0] += 1
        if status is RouteRedundancyStatus.EMPTY:
            counts[1] += 1
        elif status is RouteRedundancyStatus.SINGLE_ROUTE:
            counts[2] += 1
        elif status is RouteRedundancyStatus.DISJOINT:
            counts[3] += 1
        elif status is RouteRedundancyStatus.IDENTICAL:
            counts[4] += 1
        elif status is RouteRedundancyStatus.APPLIED:
            counts[5] += 1
        elif status is RouteRedundancyStatus.SCORING_FALLBACK:
            counts[6] += 1
        else:
            raise TypeError("status must be RouteRedundancyStatus")

    def _record_intent_epoch_slate_status(
        self,
        status: IntentEpochSlateStatus,
        *,
        eligible_prior_shown: int,
    ) -> None:
        if (
            type(eligible_prior_shown) is not int
            or not 0 <= eligible_prior_shown <= MAX_SLATE_CANDIDATES
        ):
            raise ValueError("eligible_prior_shown is out of bounds")
        counts = getattr(self, "_intent_epoch_slate_counts", None)
        if counts is None:
            # Allocate fixed aggregate telemetry lazily so both protected and
            # candidate constructors retain the exact Phase 9 shape/cost.
            counts = [0] * 8
            self._intent_epoch_slate_counts = counts
        if (
            type(counts) is not list
            or len(counts) != 8
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("intent-epoch slate counter state is invalid")
        counts[0] += 1
        if status is IntentEpochSlateStatus.EMPTY:
            counts[1] += 1
        elif status is IntentEpochSlateStatus.FIRST:
            counts[2] += 1
        elif status is IntentEpochSlateStatus.UNCHANGED:
            counts[3] += 1
        elif status is IntentEpochSlateStatus.EPOCH_RESET:
            counts[4] += 1
        elif status is IntentEpochSlateStatus.CARRIED:
            counts[5] += 1
        elif status is IntentEpochSlateStatus.VALIDATION_FALLBACK:
            counts[6] += 1
        else:
            raise TypeError("status must be IntentEpochSlateStatus")
        counts[7] += eligible_prior_shown

    def _record_profile_status(self, status: ProfileResidualStatus) -> None:
        if status is ProfileResidualStatus.NO_REPRESENTED_THEME:
            self._empty_represented_theme_fallbacks += 1
        elif status is ProfileResidualStatus.CONSTANT_SCORE:
            self._constant_score_neutral_fallbacks += 1
        elif status is ProfileResidualStatus.APPLIED:
            self._successful_residual_applications += 1
        elif status is ProfileResidualStatus.SCORING_FALLBACK:
            self._parsing_or_scoring_fallbacks += 1

    def session_state(self, session_id: str) -> IntentState:
        """Read-only diagnostic hook used by tests and local analysis."""

        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id]

    def slate_state(self, session_id: str) -> SlateState:
        """Read-only diagnostic hook for exact behavior-equivalence tests."""

        if session_id not in self._slates:
            raise KeyError(session_id)
        return self._slates[session_id]

    def last_action_trace(self, session_id: str) -> dict[str, object] | None:
        """Return one sanitized opt-in decision trace for demos and audits."""

        if session_id not in self._sessions:
            raise KeyError(session_id)
        traces = getattr(self, "_protocol_action_traces", None)
        if not isinstance(traces, dict):
            return None
        trace = traces.get(session_id)
        return dict(trace) if isinstance(trace, dict) else None

    @property
    def retrieval_backend(self) -> object:
        """Share the immutable/runtime retrieval assets with sequential experiments."""

        return self._retriever

    @property
    def intent_policy(self) -> IntentParsingPolicy:
        return self._intent_policy

    def _record_retrieval_route_plan(
        self,
        plan: RetrievalRoutePlan,
    ) -> None:
        """Record one bounded route plan without retaining intent or IDs."""

        if not isinstance(plan, RetrievalRoutePlan):
            raise TypeError("route plan must be RetrievalRoutePlan")
        reasons = tuple(RetrievalRouteReason)
        self._retrieval_route_reason_counts[reasons.index(plan.reason)] += 1
        if plan.mode is RetrievalRouteMode.HYBRID:
            self._retrieval_route_outcome_counts[0] += 1
        else:
            self._retrieval_route_outcome_counts[1] += 1

    def _record_retrieval_route_outcome(
        self,
        action: QueryAction,
        retrieval: RetrievalResult | None,
    ) -> None:
        """Record the executed route using aggregate trace statuses only."""

        if action is QueryAction.REUSE:
            self._retrieval_route_outcome_counts[3] += 1
            return
        if action is QueryAction.SKIP:
            self._retrieval_route_outcome_counts[4] += 1
            return
        self._retrieval_route_outcome_counts[2] += 1
        if retrieval is None:
            self._retrieval_route_outcome_counts[8] += 1
            return
        trace = retrieval.trace
        if trace.used_fallback:
            self._retrieval_route_outcome_counts[8] += 1
        elif trace.bm25_status == "ok" and trace.dense_status == "skipped":
            self._retrieval_route_outcome_counts[5] += 1
        elif (
            trace.bm25_status in {"unavailable", "error", "empty"}
            and trace.dense_status == "ok"
        ):
            self._retrieval_route_outcome_counts[6] += 1
        elif trace.bm25_status == "ok" and trace.dense_status == "ok":
            self._retrieval_route_outcome_counts[7] += 1
        else:
            self._retrieval_route_outcome_counts[9] += 1

    @property
    def retrieval_routing_policy(self) -> RetrievalRoutingPolicy:
        policy = getattr(
            self,
            "_retrieval_routing_policy",
            ALWAYS_HYBRID_RETRIEVAL_ROUTING_POLICY,
        )
        if not isinstance(policy, RetrievalRoutingPolicy):
            raise RuntimeError("retrieval routing policy state is invalid")
        return policy

    @property
    def retrieval_routing_health(self) -> dict[str, object]:
        """Return aggregate-only smart-routing decisions and outcomes."""

        raw_reasons = getattr(self, "_retrieval_route_reason_counts", None)
        reason_counts = (
            tuple(raw_reasons)
            if type(raw_reasons) is list
            and len(raw_reasons) == len(RetrievalRouteReason)
            and all(type(value) is int and value >= 0 for value in raw_reasons)
            else (0,) * len(RetrievalRouteReason)
        )
        raw_outcomes = getattr(self, "_retrieval_route_outcome_counts", None)
        outcomes = (
            tuple(raw_outcomes)
            if type(raw_outcomes) is list
            and len(raw_outcomes) == 10
            and all(type(value) is int and value >= 0 for value in raw_outcomes)
            else (0,) * 10
        )
        return {
            "policy": self.retrieval_routing_policy.value,
            "decisions": outcomes[0] + outcomes[1],
            "planned_hybrid": outcomes[0],
            "planned_bm25_first": outcomes[1],
            "reasons": {
                reason.value: reason_counts[index]
                for index, reason in enumerate(RetrievalRouteReason)
            },
            "searches": outcomes[2],
            "reuses": outcomes[3],
            "skips": outcomes[4],
            "executed_bm25_only": outcomes[5],
            "executed_dense_rescue": outcomes[6],
            "executed_hybrid": outcomes[7],
            "fallbacks_or_execution_errors": outcomes[8],
            "degraded_route_outcomes": outcomes[9],
        }

    @property
    def local_intent_health(self) -> dict[str, int | str | bool | None]:
        """Return aggregate-only local parser outcomes without message content."""

        return {
            "enabled": self._local_intent_parser is not None,
            "initialization_error": self.local_intent_initialization_error,
            "attempts": self._local_intent_attempts,
            "applied": self._local_intent_applied,
            "no_delta": self._local_intent_no_delta,
            "failures": self._local_intent_failures,
            "free_text_attempts": self._local_intent_free_text_attempts,
            "complex_language_attempts": self._local_intent_complex_attempts,
            "prompt_tokens": self._local_intent_prompt_tokens,
            "completion_tokens": self._local_intent_completion_tokens,
        }

    @property
    def field_semantic_policy(self) -> FieldSemanticPolicy:
        policy = getattr(
            self,
            "_field_semantic_policy",
            DISABLED_FIELD_SEMANTIC_POLICY,
        )
        if not isinstance(policy, FieldSemanticPolicy):
            raise RuntimeError("field-semantic policy state is invalid")
        return policy

    @property
    def field_semantic_health(self) -> dict[str, int | str]:
        """Return aggregate-only local-intent semantic scoring outcomes."""

        raw_counts = getattr(self, "_field_semantic_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == 7
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * 7
        )
        return {
            "policy": self.field_semantic_policy.value,
            "attempts": counts[0],
            "reordered": counts[1],
            "unchanged": counts[2],
            "no_signal": counts[3],
            "capability_unavailable": counts[4],
            "scoring_errors": counts[5],
            "candidate_ids_examined": counts[6],
        }

    @property
    def decision_policy(self) -> DecisionPolicy:
        policy = getattr(self, "_decision_policy", PROTECTED_DECISION_POLICY)
        if not isinstance(policy, DecisionPolicy):
            raise RuntimeError("decision policy state is invalid")
        return policy

    @property
    def protocol_catalog_policy(self) -> ProtocolCatalogPolicy:
        policy = getattr(
            self,
            "_protocol_catalog_policy",
            DISABLED_PROTOCOL_CATALOG_POLICY,
        )
        if not isinstance(policy, ProtocolCatalogPolicy):
            raise RuntimeError("protocol catalog policy state is invalid")
        return policy

    @property
    def protocol_refutation_policy(self) -> ProtocolRefutationPolicy:
        policy = getattr(
            self,
            "_protocol_refutation_policy",
            DISABLED_PROTOCOL_REFUTATION_POLICY,
        )
        if not isinstance(policy, ProtocolRefutationPolicy):
            raise RuntimeError("protocol refutation policy state is invalid")
        return policy

    @property
    def exact_evidence_health(self) -> dict[str, int | str]:
        """Return aggregate-only post-Stage-A evidence outcomes."""

        raw_counts = getattr(self, "_exact_evidence_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == 9
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * 9
        )
        return {
            "policy": self._ranking_policy.value,
            "attempts": counts[0],
            "applied": counts[1] + counts[2],
            "applied_reordered": counts[1],
            "applied_unchanged": counts[2],
            "zero_support_fail_open": counts[3],
            "capability_unavailable": counts[4],
            "evidence_errors": counts[5],
            "validation_errors": counts[6],
            "candidate_ids_examined": counts[7],
            "consistent_support_ids": counts[8],
        }

    @property
    def semantic_tiebreak_policy(self) -> SemanticTieBreakPolicy:
        policy = getattr(
            self,
            "_semantic_tiebreak_policy",
            DISABLED_SEMANTIC_TIEBREAK_POLICY,
        )
        if not isinstance(policy, SemanticTieBreakPolicy):
            raise RuntimeError("semantic tie-break policy state is invalid")
        return policy

    @property
    def semantic_tiebreak_health(self) -> dict[str, int | str]:
        """Return aggregate-only dense best-tier reranking outcomes."""

        statuses = tuple(SemanticTieBreakStatus)
        expected_length = len(statuses) + 2
        raw_counts = getattr(self, "_semantic_tiebreak_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == expected_length
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * expected_length
        )
        return {
            "policy": self.semantic_tiebreak_policy.value,
            "attempts": counts[0],
            **{
                status.value: counts[index]
                for index, status in enumerate(statuses, start=1)
            },
            "validation_or_execution_fallbacks": counts[-1],
        }

    @property
    def importance_satisfaction_health(self) -> dict[str, int | str]:
        """Return aggregate-only ordinal reranker outcomes."""

        raw_counts = getattr(self, "_importance_satisfaction_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == 15
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * 15
        )
        return {
            "policy": self._ranking_policy.value,
            "attempts": counts[0],
            "applied": counts[1] + counts[2],
            "applied_reordered": counts[1],
            "applied_unchanged": counts[2],
            "no_requirements": counts[3],
            "capability_unavailable": counts[4],
            "evidence_errors": counts[5],
            "validation_errors": counts[6],
            "candidate_ids_examined": counts[7],
            "requirements_evaluated": counts[8],
            "must_violation_candidates": counts[9],
            "must_unknown_candidates": counts[10],
            "all_must_full_candidates": counts[11],
            "best_tier_candidates": counts[12],
            "budget_requirements": counts[13],
            "profile_preferences": counts[14],
        }

    @property
    def protocol_decision_health(self) -> dict[str, object]:
        """Return aggregate-only outcomes for the inactive Phase 15 candidate."""

        raw_counts = getattr(self, "_protocol_decision_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == 9
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * 9
        )
        raw_question_counts = getattr(self, "_protocol_question_counts", None)
        question_counts = (
            tuple(raw_question_counts)
            if type(raw_question_counts) is list
            and len(raw_question_counts) == len(PROTOCOL_QUESTION_ACTIONS)
            and all(
                type(value) is int and value >= 0
                for value in raw_question_counts
            )
            else (0,) * len(PROTOCOL_QUESTION_ACTIONS)
        )
        raw_width_counts = getattr(self, "_protocol_width_counts", None)
        width_counts = (
            tuple(raw_width_counts)
            if type(raw_width_counts) is list
            and len(raw_width_counts) == 11
            and all(type(value) is int and value >= 0 for value in raw_width_counts)
            else (0,) * 11
        )
        return {
            "policy": self.decision_policy.value,
            "turns": counts[0],
            **{
                outcome: counts[index]
                for index, outcome in enumerate(
                    PROTOCOL_DECISION_OUTCOMES,
                    start=1,
                )
            },
            "question_action_counts": {
                question: question_counts[index]
                for index, question in enumerate(PROTOCOL_QUESTION_ACTIONS)
            },
            "width_action_counts": {
                str(width): width_counts[width] for width in range(11)
            },
            "requested_total": getattr(self, "_protocol_requested_total", 0),
            "presented_total": getattr(self, "_protocol_presented_total", 0),
        }

    @property
    def semantic_lexical_rescue_policy(self) -> SemanticLexicalRescuePolicy:
        policy = getattr(
            self,
            "_semantic_lexical_rescue_policy",
            DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
        )
        if not isinstance(policy, SemanticLexicalRescuePolicy):
            raise RuntimeError("semantic rescue policy state is invalid")
        return policy

    @property
    def semantic_lexical_rescue_health(self) -> dict[str, int | str]:
        statuses = tuple(SemanticLexicalRescueStatus)
        expected_length = len(statuses) + 2
        raw_counts = getattr(self, "_semantic_rescue_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == expected_length
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * expected_length
        )
        return {
            "policy": self.semantic_lexical_rescue_policy.value,
            "attempts": counts[0],
            **{
                status.value: counts[index]
                for index, status in enumerate(statuses, start=1)
            },
            "validation_or_execution_fallbacks": counts[-1],
        }

    @property
    def evidence_exposure_policy(self) -> EvidenceExposurePolicy:
        policy = getattr(
            self,
            "_evidence_exposure_policy",
            DISABLED_EVIDENCE_EXPOSURE_POLICY,
        )
        if not isinstance(policy, EvidenceExposurePolicy):
            raise RuntimeError("evidence exposure policy state is invalid")
        return policy

    @property
    def evidence_exposure_health(self) -> dict[str, int | str]:
        statuses = tuple(EvidenceExposureStatus)
        expected_length = len(statuses) + 3
        raw_counts = getattr(self, "_evidence_exposure_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == expected_length
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * expected_length
        )
        return {
            "policy": self.evidence_exposure_policy.value,
            "attempts": counts[0],
            **{
                status.value: counts[index]
                for index, status in enumerate(statuses, start=1)
            },
            "withheld_turns": counts[-2],
            "validation_fallbacks": counts[-1],
        }

    @property
    def requirement_probe_policy(self) -> RequirementProbePolicy:
        policy = getattr(
            self,
            "_requirement_probe_policy",
            DISABLED_REQUIREMENT_PROBE_POLICY,
        )
        if not isinstance(policy, RequirementProbePolicy):
            raise RuntimeError("requirement-probe policy state is invalid")
        return policy

    @property
    def requirement_probe_health(self) -> dict[str, int | str]:
        """Return fixed-cardinality aggregate probe counters without query text."""

        raw_counts = getattr(self, "_requirement_probe_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == 12
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * 12
        )
        if (
            len(counts) != 12
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            raise RuntimeError("requirement-probe counter state is invalid")
        return {
            "policy": self.requirement_probe_policy.value,
            "attempts": counts[0],
            "disabled": counts[1],
            "no_eligible": counts[2],
            "capacity": counts[3],
            "successful_supplements": counts[4],
            "empty_routes": counts[5],
            "no_additions": counts[6],
            "unavailable": counts[7],
            "errors": counts[8],
            "selected_probe_queries": counts[9],
            "supplemental_ids": counts[10],
            "validation_or_execution_fallbacks": counts[11],
        }

    @property
    def orchestration_policy(self) -> OrchestrationPolicy:
        return self._orchestrator.policy

    @property
    def orchestration_health(self) -> dict[str, object]:
        """Return aggregate cache/action counters without queries or user data."""

        return self._orchestrator.health

    @property
    def ranking_health(self) -> dict[str, int | str]:
        """Return aggregate label-free health counters for local experiments."""

        return {
            "policy": self._ranking_policy.value,
            "attempts": self._reranking_attempts,
            "successes": self._reranking_successes,
            "failures": self._reranking_failures,
            "unavailable_skips": self._reranking_unavailable_skips,
        }

    @property
    def rescue_health(self) -> dict[str, int | str]:
        """Return fixed-cardinality aggregate BM25-rescue counters only."""

        return {
            "policy": self._ranking_policy.value,
            "attempts": self._bm25_rescue_attempts,
            "zero_completeness_neutral": (
                self._bm25_rescue_zero_completeness
            ),
            "bm25_unavailable_or_empty_neutral": (
                self._bm25_rescue_unavailable_or_empty
            ),
            "no_positive_uplift_neutral": (
                self._bm25_rescue_no_positive_uplift
            ),
            "constant_uplift_neutral": self._bm25_rescue_constant_uplift,
            "unchanged_order_neutral": self._bm25_rescue_unchanged_order,
            "successful_reorders": self._bm25_rescue_successful_reorders,
            "validation_or_scoring_fallbacks": self._bm25_rescue_fallbacks,
        }

    @property
    def route_redundancy_health(self) -> dict[str, int | str]:
        """Return fixed-cardinality aggregate Phase 12 counters only."""

        raw_counts = getattr(self, "_route_redundancy_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == 7
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * 7
        )
        return {
            "policy": self._ranking_policy.value,
            "attempts": counts[0],
            "empty_exact_baseline": counts[1],
            "single_route_exact_baseline": counts[2],
            "disjoint_exact_baseline": counts[3],
            "identical_order_exact_baseline": counts[4],
            "correction_applied": counts[5],
            "validation_or_scoring_fallbacks": counts[6],
        }

    @property
    def profile_health(self) -> dict[str, int | str]:
        """Return fixed-cardinality aggregate profile counters only."""

        session_entries = len(self._profile_priors)
        return {
            "policy": self._profile_policy.value,
            "session_entries": session_entries,
            "logical_profile_bytes": session_entries * PROFILE_THEME_MASK_BYTES,
            "profiles_reset": self._profiles_reset,
            "zero_mask_profiles": self._zero_mask_profiles,
            "nonzero_mask_profiles": self._nonzero_mask_profiles,
            "recognized_theme_count": self._recognized_theme_count,
            "turns_disabled_by_active_requirements": (
                self._turns_disabled_by_active_requirements
            ),
            "eligible_stage_a_attempts": self._eligible_stage_a_attempts,
            "empty_represented_theme_fallbacks": (
                self._empty_represented_theme_fallbacks
            ),
            "constant_score_neutral_fallbacks": (
                self._constant_score_neutral_fallbacks
            ),
            "successful_residual_applications": (
                self._successful_residual_applications
            ),
            "parsing_or_scoring_fallbacks": (
                self._parsing_or_scoring_fallbacks
            ),
        }

    @property
    def slate_health(self) -> dict[str, int | str]:
        """Return aggregate label-free exploration counters for local experiments."""

        return {
            "policy": self._slate_policy.value,
            "attempts": self._slate_attempts,
            "successes": self._slate_successes,
            "failures": self._slate_failures,
            "initializations": self._slate_initializations,
            "ranking_resets": self._slate_ranking_resets,
            "stagnant_turns": self._slate_stagnant_turns,
            "unseen_selected_on_stagnant": (
                self._slate_unseen_selected_on_stagnant
            ),
            "repeat_backfills": self._slate_repeat_backfills,
        }

    @property
    def intent_epoch_slate_health(self) -> dict[str, int | str]:
        """Return fixed-cardinality aggregate Phase 13 counters only."""

        raw_counts = getattr(self, "_intent_epoch_slate_counts", None)
        counts = (
            tuple(raw_counts)
            if type(raw_counts) is list
            and len(raw_counts) == 8
            and all(type(value) is int and value >= 0 for value in raw_counts)
            else (0,) * 8
        )
        return {
            "policy": self._slate_policy.value,
            "attempts": counts[0],
            "empty_exact_baseline": counts[1],
            "first_slate_exact_baseline": counts[2],
            "unchanged_signature_exact_baseline": counts[3],
            "changed_epoch_exact_baseline": counts[4],
            "same_epoch_history_carried": counts[5],
            "validation_fallbacks": counts[6],
            "eligible_prior_shown_total": counts[7],
        }
