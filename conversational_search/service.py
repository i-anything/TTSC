from __future__ import annotations

from pathlib import Path

from conversational_search.intent import (
    ROBUST_INTENT_POLICY,
    IntentState,
    IntentParsingPolicy,
    apply_user_message,
    record_question,
    render_dense_query,
    render_lexical_query,
)
from conversational_search.orchestration import (
    BackendSnapshotToken,
    DEFAULT_RANKING_CACHE_CAPACITY,
    EXACT_RANKING_CACHE_CAPABILITY,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    OrchestrationPlanner,
    OrchestrationPolicy,
    QueryAction,
)
from conversational_search.questions import (
    CONSERVATIVE_EARLY_OTHER_POLICY,
    QUESTION_TEXT,
    QuestionPolicy,
)
from conversational_search.ranking import (
    STAGE_A_RANKING_POLICY,
    RankingPolicy,
    rerank_stage_a,
)
from conversational_search.retrieval import HybridRetriever, RetrievalResult
from conversational_search.slates import (
    MAX_SLATE_CANDIDATES,
    REPEAT_TOP_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
    SlatePolicy,
    SlateState,
    ranking_signature,
    select_slate,
)
from conversational_search.strategy import (
    COMPLETENESS_ADAPTIVE_RRF_POLICY,
    FusionPolicy,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ASSETS = REPOSITORY_ROOT / "assets" / "bge-small-en-v1.5-int8"
DEFAULT_DENSE_INDEX = (
    REPOSITORY_ROOT / "assets" / "search-index-bge-small-en-v1.5-v2"
)
CACHEABLE_ROUTE_STATUSES = frozenset({"ok", "empty"})

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
        slate_policy: SlatePolicy = STAGNATION_AWARE_SLATE_POLICY,
        intent_policy: IntentParsingPolicy = ROBUST_INTENT_POLICY,
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
        if not isinstance(slate_policy, SlatePolicy):
            raise TypeError("slate_policy must be a SlatePolicy")
        if not isinstance(intent_policy, IntentParsingPolicy):
            raise TypeError("intent_policy must be an IntentParsingPolicy")
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
        self._slate_policy = slate_policy
        self._intent_policy = intent_policy
        self._orchestrator = OrchestrationPlanner(
            orchestration_policy,
            capacity=ranking_cache_capacity,
        )
        self._reranking_attempts = 0
        self._reranking_successes = 0
        self._reranking_failures = 0
        self._reranking_unavailable_skips = 0
        self._sessions: dict[str, IntentState] = {}
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
        # Phase 1 deliberately leaves profile blending disabled until an ablation
        # shows that it improves target recall without overpowering explicit intent.
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

        state = apply_user_message(
            self._sessions[session_id],
            user_message,
            turn,
            policy=self._intent_policy,
        )
        dense_query = render_dense_query(state)
        lexical_query = render_lexical_query(state)
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
            self._ranking_policy is RankingPolicy.STAGE_A
            and backend_snapshot_token is not None
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
        )

        retrieval: RetrievalResult | None = None
        parent_asins: object = ()
        full_ranked_ids: tuple[str, ...] | None = None
        if decision.action is QueryAction.REUSE:
            if self._slate_policy is REPEAT_TOP_SLATE_POLICY:
                parent_asins = decision.cached_ranked_ids[:result_count]
            else:
                full_ranked_ids = decision.cached_ranked_ids
                parent_asins = full_ranked_ids
        elif decision.action is QueryAction.SEARCH:
            try:
                retrieval = self._retriever.search_with_trace(
                    dense_query,
                    lexical_query,
                    top_k=result_count,
                    route_weights=route_weights,
                )
                if not isinstance(retrieval, RetrievalResult):
                    raise TypeError("search_with_trace must return RetrievalResult")
            except Exception:
                retrieval = None
            parent_asins = () if retrieval is None else retrieval.recommendations

            if (
                retrieval is not None
                and self._ranking_policy is RankingPolicy.STAGE_A
                and retrieval.trace.fused_ids
                and not retrieval.trace.used_fallback
            ):
                self._reranking_attempts += 1
                try:
                    documents = self._retriever.candidate_documents(
                        retrieval.trace.fused_ids
                    )
                    ranked = (
                        rerank_stage_a(
                            state,
                            documents,
                            bm25_ids=retrieval.trace.bm25_ids,
                            dense_ids=retrieval.trace.dense_ids,
                            fused_ids=retrieval.trace.fused_ids,
                            route_weights=route_weights,
                        )
                        if documents
                        else None
                    )
                except Exception:
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
                        if route_is_cacheable and complete_ranking:
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
