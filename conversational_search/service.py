from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

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
    MAX_CANDIDATE_DOCUMENTS,
    MAX_REQUIREMENT_PROBES,
    REQUIREMENT_PROBE_CAPABILITY,
    ROUTE_LIMIT,
    HybridRetriever,
    RequirementProbeRetrievalResult,
    RequirementProbePolicy,
    RequirementProbeTrace,
    RetrievalResult,
    RetrievalTrace,
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ASSETS = REPOSITORY_ROOT / "assets" / "bge-small-en-v1.5-int8"
DEFAULT_DENSE_INDEX = (
    REPOSITORY_ROOT / "assets" / "search-index-bge-small-en-v1.5-v2"
)
CACHEABLE_ROUTE_STATUSES = frozenset({"ok", "empty"})
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
    }
)


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

    keys = (
        "model_id",
        "revision",
        "model_sha256",
        "source_model_sha256",
        "asset_manifest_sha256",
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
    from preprocessing.encoder import OnnxBgeEncoder
    from starter.dense import ShardedDenseIndex

    encoder = OnnxBgeEncoder(model_assets, threads=1)
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
        ranking_policy: RankingPolicy = STAGE_A_RANKING_POLICY,
        profile_policy: ProfilePolicy = BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy: SlatePolicy = STAGNATION_AWARE_SLATE_POLICY,
        intent_policy: IntentParsingPolicy = ROBUST_INTENT_POLICY,
        requirement_probe_policy: RequirementProbePolicy = (
            DISABLED_REQUIREMENT_PROBE_POLICY
        ),
        orchestration_policy: OrchestrationPolicy = (
            EXACT_RANKING_REUSE_ORCHESTRATION_POLICY
        ),
        ranking_cache_capacity: int = DEFAULT_RANKING_CACHE_CAPACITY,
        model_assets: str | Path = DEFAULT_MODEL_ASSETS,
        dense_index_path: str | Path = DEFAULT_DENSE_INDEX,
    ) -> None:
        if not isinstance(question_policy, QuestionPolicy):
            raise TypeError("question_policy must be a QuestionPolicy")
        if not isinstance(fusion_policy, FusionPolicy):
            raise TypeError("fusion_policy must be a FusionPolicy")
        if not isinstance(ranking_policy, RankingPolicy):
            raise TypeError("ranking_policy must be a RankingPolicy")
        if not isinstance(profile_policy, ProfilePolicy):
            raise TypeError("profile_policy must be a ProfilePolicy")
        if not isinstance(slate_policy, SlatePolicy):
            raise TypeError("slate_policy must be a SlatePolicy")
        if not isinstance(intent_policy, IntentParsingPolicy):
            raise TypeError("intent_policy must be an IntentParsingPolicy")
        if not isinstance(requirement_probe_policy, RequirementProbePolicy):
            raise TypeError(
                "requirement_probe_policy must be a RequirementProbePolicy"
            )
        if (
            requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY
            and ranking_policy is not RankingPolicy.STAGE_A
        ):
            raise ValueError(
                "requirement probes are supported only with the Stage-A policy"
            )
        if not isinstance(orchestration_policy, OrchestrationPolicy):
            raise TypeError("orchestration_policy must be an OrchestrationPolicy")
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
            )
        self._retriever = retriever
        self._question_policy = question_policy
        self._fusion_policy = fusion_policy
        self._ranking_policy = ranking_policy
        self._profile_policy = profile_policy
        self._slate_policy = slate_policy
        self._intent_policy = intent_policy
        if requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY:
            self._requirement_probe_policy = requirement_probe_policy
            self._requirement_probe_counts = [0] * 12
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

        intent_cacheable = True
        if self._intent_policy is LOSSLESS_MULTI_SLOT_INTENT_POLICY:
            reduction = apply_user_message_with_trace(
                self._sessions[session_id],
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
                self._sessions[session_id],
                user_message,
                turn,
                policy=self._intent_policy,
            )
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
        route_weights = self._fusion_policy.choose(state)
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
            retrieval_policy=(
                self.requirement_probe_policy.value
                if probe_enabled
                else None
            ),
        )

        retrieval: RetrievalResult | None = None
        probe_result_cacheable = True
        parent_asins: object = ()
        full_ranked_ids: tuple[str, ...] | None = None
        if decision.action is QueryAction.REUSE:
            if self._slate_policy is REPEAT_TOP_SLATE_POLICY:
                parent_asins = decision.cached_ranked_ids[:result_count]
            else:
                full_ranked_ids = decision.cached_ranked_ids
                parent_asins = full_ranked_ids
        elif decision.action is QueryAction.SEARCH:
            if not probe_enabled:
                try:
                    retrieval = self._retriever.search_with_trace(
                        dense_query,
                        lexical_query,
                        top_k=result_count,
                        route_weights=route_weights,
                    )
                    if not isinstance(retrieval, RetrievalResult):
                        raise TypeError(
                            "search_with_trace must return RetrievalResult"
                        )
                except Exception:
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
                        route_is_cacheable = (
                            retrieval.trace.bm25_status
                            in CACHEABLE_ROUTE_STATUSES
                            and retrieval.trace.dense_status
                            in CACHEABLE_ROUTE_STATUSES
                        )
                        complete_ranking = (
                            bool(sanitized_ranked_ids)
                            and sanitized_ranked_ids == ranked.ranked_ids
                        )
                        if (
                            route_is_cacheable
                            and complete_ranking
                            and ranking_cacheable
                            and intent_cacheable
                            and probe_result_cacheable
                        ):
                            self._orchestrator.commit(
                                session_id,
                                decision,
                                backend_snapshot_token,
                                sanitized_ranked_ids,
                            )
                        if self._slate_policy is REPEAT_TOP_SLATE_POLICY:
                            parent_asins = sanitized_ranked_ids[:result_count]
                        else:
                            full_ranked_ids = sanitized_ranked_ids
                            parent_asins = full_ranked_ids

        prior_slate_state = self._slates[session_id]
        next_slate_state = prior_slate_state
        slate_trace = None
        if full_ranked_ids is not None and result_count > 0:
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
                                result_count,
                            )
                        )
                        self._validate_intent_epoch_slate_selection(
                            epoch_selection,
                            prior_state=prior_slate_state,
                            signature=signature,
                            ranked_ids=full_ranked_ids,
                            limit=result_count,
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
                            result_count,
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
                        result_count,
                    )
            except Exception:
                self._slate_failures += 1
                recommendations = self._sanitize(parent_asins, result_count)
            else:
                self._slate_successes += 1
                recommendations = list(selection.selected_ids)
                next_slate_state = selection.state
                slate_trace = selection.trace
        else:
            recommendations = self._sanitize(parent_asins, result_count)

        ask_attribute = (
            None if turn >= 10 else self._question_policy.choose(state)
        )
        if ask_attribute is not None:
            state = record_question(state, ask_attribute)
            message = (
                "Here are the closest matches so far. "
                + QUESTION_TEXT[ask_attribute]
            )
        else:
            message = "Here are the closest matches based on your current preferences."
        self._sessions[session_id] = state
        self._slates[session_id] = next_slate_state
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
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

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

    @property
    def retrieval_backend(self) -> object:
        """Share the immutable/runtime retrieval assets with sequential experiments."""

        return self._retriever

    @property
    def intent_policy(self) -> IntentParsingPolicy:
        return self._intent_policy

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
