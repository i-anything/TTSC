"""Label-free one-step planning over a question and recommendation width."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite


MAX_TURN = 10


@dataclass(frozen=True, slots=True)
class CandidateHypothesis:
    """One ranked target hypothesis and its deterministic possible replies."""

    candidate_id: str
    rank: int
    weight: float
    answer_signatures: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PlannedAction:
    """The deterministic maximum-utility action for the current turn."""

    question: str | None
    width: int
    value: float


class RetrievalChoice(str, Enum):
    """Bounded next-turn retrieval choices after the modeled response."""

    REUSE = "reuse"
    RERETRIEVE = "reretrieve"


@dataclass(frozen=True, slots=True)
class ExpectedUtilityCandidate:
    """One current target belief plus counterfactual post-reply ranks."""

    candidate_id: str
    current_rank: int
    probability: float


@dataclass(frozen=True, slots=True)
class SimulatedQuestion:
    """Per-candidate ranks after a modeled ordinary or shared reply."""

    question: str
    ordinary_post_ranks: tuple[tuple[str, int | None], ...]
    shared_post_ranks: tuple[tuple[str, int | None], ...] = ()
    shared_reply_probability: float = 0.0
    ordinary_post_ranks_by_width: tuple[
        tuple[int, tuple[tuple[str, int | None], ...]], ...
    ] = ()
    shared_post_ranks_by_width: tuple[
        tuple[int, tuple[tuple[str, int | None], ...]], ...
    ] = ()


@dataclass(frozen=True, slots=True)
class ExpectedUtilityAction:
    """One fully scored question, width, and retrieval combination."""

    question: str | None
    width: int
    retrieval: RetrievalChoice
    expected_reward: float
    immediate_reward: float
    continuation_reward: float
    computation_cost: float
    value: float
    residual_risk: float


@dataclass(frozen=True, slots=True)
class ExpectedUtilityPlan:
    """Selected action plus an aggregate-safe runner-up for diagnostics."""

    selected: ExpectedUtilityAction
    runner_up: ExpectedUtilityAction | None
    protocol_confidence: float
    fallback_reason: str | None = None


def hit_utility(turn: int, rank: int) -> float:
    """Return the official utility for a hit at ``turn`` and displayed ``rank``."""

    if isinstance(turn, bool) or not isinstance(turn, int) or not 1 <= turn <= MAX_TURN:
        raise ValueError("turn must be an integer from 1 through 10")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    return 0.5 + (0.3 / rank) + (0.02 * (11 - turn))


def plan_expected_utility(
    candidates: Sequence[ExpectedUtilityCandidate],
    questions: Sequence[SimulatedQuestion],
    *,
    current_turn: int,
    top_k: int,
    widths: Sequence[int],
    retrieval_choices: Sequence[RetrievalChoice],
    out_of_pool_probability: float,
    protocol_confidence: float,
    protocol_locked: bool = False,
    allow_zero_width: bool = False,
    reretrieve_recovery_probability: float = 0.0,
    reretrieve_recovery_rank: int | None = None,
    reretrieve_computation_cost: float = 0.0,
    no_question_post_ranks_by_width: tuple[
        tuple[int, tuple[tuple[str, int | None], ...]], ...
    ] = (),
    fallback_reason: str | None = None,
) -> ExpectedUtilityPlan:
    """Choose the highest exact-score expectation over bounded actions.

    Candidate probabilities are not renormalized: they must sum with the
    explicit out-of-pool probability to one.  Each modeled reply supplies the
    ranks obtained by actually updating state and reranking the cached pool.
    Protocol uncertainty is represented explicitly by shared-reply and
    out-of-pool probability mass. The confidence score is retained for gates
    and diagnostics rather than multiplying those probabilities a second
    time. The official per-session score is the sole reward; an optional
    nonnegative computation cost is reported separately and subtracted only
    when comparing an otherwise valid reretrieval action.
    """

    (
        ordered,
        question_models,
        legal_widths,
        legal_retrievals,
        residual,
        confidence,
        recovery_probability,
        recovery_rank,
        reretrieve_cost,
    ) = _validated_expected_utility_inputs(
        candidates,
        questions,
        current_turn=current_turn,
        top_k=top_k,
        widths=widths,
        retrieval_choices=retrieval_choices,
        out_of_pool_probability=out_of_pool_probability,
        protocol_confidence=protocol_confidence,
        protocol_locked=protocol_locked,
        allow_zero_width=allow_zero_width,
        reretrieve_recovery_probability=reretrieve_recovery_probability,
        reretrieve_recovery_rank=reretrieve_recovery_rank,
        reretrieve_computation_cost=reretrieve_computation_cost,
        no_question_post_ranks_by_width=no_question_post_ranks_by_width,
    )
    no_question_by_width = {
        width: dict(ranks)
        for width, ranks in no_question_post_ranks_by_width
    }
    current_rank_map = {
        candidate.candidate_id: candidate.current_rank
        for candidate in ordered
    }
    question_options: tuple[SimulatedQuestion | None, ...] = (
        (None,)
        if current_turn == MAX_TURN
        else (*question_models, None)
    )
    effective_recovery_probability = (
        recovery_probability if current_turn < MAX_TURN else 0.0
    )
    actions: list[ExpectedUtilityAction] = []
    for retrieval in legal_retrievals:
        computation_cost = (
            reretrieve_cost
            if retrieval is RetrievalChoice.RERETRIEVE
            else 0.0
        )
        for question in question_options:
            default_ordinary_ranks = (
                current_rank_map
                if question is None
                else dict(question.ordinary_post_ranks)
            )
            default_shared_ranks = (
                default_ordinary_ranks
                if question is None or question.shared_reply_probability == 0.0
                else dict(question.shared_post_ranks)
            )
            ordinary_by_width = (
                no_question_by_width
                if question is None
                else {
                    width: dict(ranks)
                    for width, ranks in question.ordinary_post_ranks_by_width
                }
            )
            shared_by_width = (
                {}
                if question is None
                else {
                    width: dict(ranks)
                    for width, ranks in question.shared_post_ranks_by_width
                }
            )
            shared_probability = (
                0.0 if question is None else question.shared_reply_probability
            )
            for width in legal_widths:
                ordinary_ranks = ordinary_by_width.get(
                    width,
                    default_ordinary_ranks,
                )
                shared_ranks = shared_by_width.get(
                    width,
                    default_shared_ranks,
                )
                immediate = 0.0
                continuation = 0.0
                for candidate in ordered:
                    probability = float(candidate.probability)
                    exposed = (
                        not protocol_locked and candidate.current_rank <= width
                    )
                    if exposed:
                        immediate += probability * hit_utility(
                            current_turn,
                            candidate.current_rank,
                        )
                        continue
                    if current_turn == MAX_TURN:
                        continue
                    ordinary_rank = ordinary_ranks[candidate.candidate_id]
                    shared_rank = shared_ranks[candidate.candidate_id]
                    ordinary_reward = _bounded_future_reward(
                        current_turn + 1,
                        ordinary_rank,
                        top_k,
                    )
                    shared_reward = _bounded_future_reward(
                        current_turn + 1,
                        shared_rank,
                        top_k,
                    )
                    modeled_reward = (
                        (1.0 - shared_probability)
                        * ordinary_reward
                        + shared_probability * shared_reward
                    )
                    continuation += probability * modeled_reward

                recovered_unknown = 0.0
                if (
                    retrieval is RetrievalChoice.RERETRIEVE
                    and current_turn < MAX_TURN
                    and effective_recovery_probability > 0.0
                    and recovery_rank is not None
                ):
                    recovered_unknown = (
                        residual
                        * effective_recovery_probability
                        * _bounded_future_reward(
                            current_turn + 1,
                            recovery_rank,
                            top_k,
                        )
                    )
                    continuation += recovered_unknown
                expected_reward = immediate + continuation
                residual_risk = residual * (
                    1.0 - effective_recovery_probability
                    if retrieval is RetrievalChoice.RERETRIEVE
                    else 1.0
                )
                actions.append(
                    ExpectedUtilityAction(
                        question=(None if question is None else question.question),
                        width=width,
                        retrieval=retrieval,
                        expected_reward=expected_reward,
                        immediate_reward=immediate,
                        continuation_reward=continuation,
                        computation_cost=computation_cost,
                        value=expected_reward - computation_cost,
                        residual_risk=residual_risk,
                    )
                )

    ranked_actions = sorted(
        enumerate(actions),
        key=lambda item: (
            -item[1].value,
            -item[1].expected_reward,
            item[1].residual_risk,
            item[1].computation_cost,
            item[1].question is not None,
            item[1].retrieval is not RetrievalChoice.REUSE,
            item[1].width if protocol_locked else -item[1].width,
            item[0],
        ),
    )
    selected = ranked_actions[0][1]
    runner_up = ranked_actions[1][1] if len(ranked_actions) > 1 else None
    return ExpectedUtilityPlan(
        selected=selected,
        runner_up=runner_up,
        protocol_confidence=confidence,
        fallback_reason=fallback_reason,
    )


def _bounded_future_reward(turn: int, rank: int | None, top_k: int) -> float:
    if rank is None or rank < 1 or rank > top_k:
        return 0.0
    return hit_utility(turn, rank)


def _validated_expected_utility_inputs(
    candidates: Sequence[ExpectedUtilityCandidate],
    questions: Sequence[SimulatedQuestion],
    *,
    current_turn: int,
    top_k: int,
    widths: Sequence[int],
    retrieval_choices: Sequence[RetrievalChoice],
    out_of_pool_probability: float,
    protocol_confidence: float,
    protocol_locked: bool,
    allow_zero_width: bool,
    reretrieve_recovery_probability: float,
    reretrieve_recovery_rank: int | None,
    reretrieve_computation_cost: float,
    no_question_post_ranks_by_width: tuple[
        tuple[int, tuple[tuple[str, int | None], ...]], ...
    ],
) -> tuple[
    tuple[ExpectedUtilityCandidate, ...],
    tuple[SimulatedQuestion, ...],
    tuple[int, ...],
    tuple[RetrievalChoice, ...],
    float,
    float,
    float,
    int | None,
    float,
]:
    if isinstance(current_turn, bool) or not isinstance(current_turn, int):
        raise ValueError("current_turn must be an integer")
    if not 1 <= current_turn <= MAX_TURN:
        raise ValueError("current_turn must be from 1 through 10")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 10
    ):
        raise ValueError("top_k must be an integer from one through ten")
    if not isinstance(protocol_locked, bool):
        raise TypeError("protocol_locked must be a boolean")
    if not isinstance(allow_zero_width, bool):
        raise TypeError("allow_zero_width must be a boolean")

    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")
    ordered = tuple(candidates)
    if not ordered or any(
        not isinstance(candidate, ExpectedUtilityCandidate)
        for candidate in ordered
    ):
        raise ValueError("candidates must contain ExpectedUtilityCandidate values")
    identifiers: set[str] = set()
    ranks: list[int] = []
    candidate_probability = 0.0
    for candidate in ordered:
        if (
            not isinstance(candidate.candidate_id, str)
            or not candidate.candidate_id
            or candidate.candidate_id != candidate.candidate_id.strip()
            or candidate.candidate_id in identifiers
        ):
            raise ValueError("candidate IDs must be unique normalized strings")
        identifiers.add(candidate.candidate_id)
        if (
            isinstance(candidate.current_rank, bool)
            or not isinstance(candidate.current_rank, int)
            or candidate.current_rank < 1
        ):
            raise ValueError("candidate ranks must be positive integers")
        ranks.append(candidate.current_rank)
        if isinstance(candidate.probability, bool) or not isinstance(
            candidate.probability,
            (int, float),
        ):
            raise TypeError("candidate probabilities must be numeric")
        probability = float(candidate.probability)
        if not isfinite(probability) or probability < 0.0:
            raise ValueError("candidate probabilities must be finite and nonnegative")
        candidate_probability += probability
    if len(set(ranks)) != len(ranks):
        raise ValueError("candidate ranks must be unique")
    if tuple(ranks) != tuple(sorted(ranks)):
        raise ValueError("candidates must be supplied in current rank order")

    residual = _finite_unit_interval(
        out_of_pool_probability,
        "out_of_pool_probability",
    )
    if not isclose(
        candidate_probability + residual,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate and out-of-pool probabilities must sum to one")
    confidence = _finite_unit_interval(
        protocol_confidence,
        "protocol_confidence",
    )
    recovery_probability = _finite_unit_interval(
        reretrieve_recovery_probability,
        "reretrieve_recovery_probability",
    )
    if recovery_probability > 0.0:
        if (
            isinstance(reretrieve_recovery_rank, bool)
            or not isinstance(reretrieve_recovery_rank, int)
            or not 1 <= reretrieve_recovery_rank <= top_k
        ):
            raise ValueError(
                "a positive recovery probability requires a rank within top_k"
            )
        recovery_rank = reretrieve_recovery_rank
    else:
        if reretrieve_recovery_rank is not None and (
            isinstance(reretrieve_recovery_rank, bool)
            or not isinstance(reretrieve_recovery_rank, int)
            or not 1 <= reretrieve_recovery_rank <= top_k
        ):
            raise ValueError("reretrieve_recovery_rank must be within top_k or None")
        recovery_rank = reretrieve_recovery_rank
    if isinstance(reretrieve_computation_cost, bool) or not isinstance(
        reretrieve_computation_cost,
        (int, float),
    ):
        raise TypeError("reretrieve_computation_cost must be numeric")
    reretrieve_cost = float(reretrieve_computation_cost)
    if not isfinite(reretrieve_cost) or reretrieve_cost < 0.0:
        raise ValueError("reretrieve_computation_cost must be finite and nonnegative")

    if isinstance(widths, (str, bytes)) or not isinstance(widths, Sequence):
        raise TypeError("widths must be a sequence")
    legal_widths = tuple(widths)
    if not legal_widths or len(set(legal_widths)) != len(legal_widths):
        raise ValueError("widths must be non-empty and unique")
    if any(
        isinstance(width, bool)
        or not isinstance(width, int)
        or not 0 <= width <= top_k
        for width in legal_widths
    ):
        raise ValueError("widths must be integers from zero through top_k")
    if (
        0 in legal_widths
        and not protocol_locked
        and not (allow_zero_width and confidence == 1.0)
    ):
        raise ValueError(
            "zero width requires a protocol lock or explicit exact-protocol permission"
        )
    if current_turn == MAX_TURN and legal_widths != (top_k,):
        raise ValueError("the final turn must expose exactly top_k")
    _validated_rank_maps_by_width(
        no_question_post_ranks_by_width,
        identifiers,
        legal_widths,
        "no_question_post_ranks_by_width",
        unique_non_none_ranks=True,
    )

    if isinstance(retrieval_choices, (str, bytes)) or not isinstance(
        retrieval_choices,
        Sequence,
    ):
        raise TypeError("retrieval_choices must be a sequence")
    legal_retrievals = tuple(retrieval_choices)
    if not legal_retrievals or len(set(legal_retrievals)) != len(
        legal_retrievals
    ):
        raise ValueError("retrieval choices must be non-empty and unique")
    if any(
        not isinstance(retrieval, RetrievalChoice)
        for retrieval in legal_retrievals
    ):
        raise TypeError("retrieval choices must contain RetrievalChoice values")

    if isinstance(questions, (str, bytes)) or not isinstance(questions, Sequence):
        raise TypeError("questions must be a sequence")
    question_models = tuple(questions)
    if any(not isinstance(question, SimulatedQuestion) for question in question_models):
        raise TypeError("questions must contain SimulatedQuestion values")
    question_names = tuple(question.question for question in question_models)
    if any(
        not isinstance(question, str)
        or not question
        or question != question.strip()
        for question in question_names
    ) or len(set(question_names)) != len(question_names):
        raise ValueError("question names must be unique normalized strings")
    for question in question_models:
        ordinary = _validated_rank_map(
            question.ordinary_post_ranks,
            identifiers,
            "ordinary_post_ranks",
        )
        shared_probability = _finite_unit_interval(
            question.shared_reply_probability,
            "shared_reply_probability",
        )
        if shared_probability > 0.0:
            _validated_rank_map(
                question.shared_post_ranks,
                identifiers,
                "shared_post_ranks",
                unique_non_none_ranks=True,
            )
        elif question.shared_post_ranks:
            _validated_rank_map(
                question.shared_post_ranks,
                identifiers,
                "shared_post_ranks",
                unique_non_none_ranks=True,
            )
        _validated_rank_maps_by_width(
            question.ordinary_post_ranks_by_width,
            identifiers,
            legal_widths,
            "ordinary_post_ranks_by_width",
        )
        _validated_rank_maps_by_width(
            question.shared_post_ranks_by_width,
            identifiers,
            legal_widths,
            "shared_post_ranks_by_width",
            unique_non_none_ranks=True,
        )
        if set(ordinary) != identifiers:  # Defensive; helper already enforces it.
            raise ValueError("ordinary reply ranks must cover every candidate")
    return (
        ordered,
        question_models,
        legal_widths,
        legal_retrievals,
        residual,
        confidence,
        recovery_probability,
        recovery_rank,
        reretrieve_cost,
    )


def _validated_rank_map(
    values: tuple[tuple[str, int | None], ...],
    identifiers: set[str],
    name: str,
    *,
    unique_non_none_ranks: bool = False,
) -> dict[str, int | None]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    result: dict[str, int | None] = {}
    for pair in values:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{name} must contain (candidate, rank) pairs")
        candidate_id, rank = pair
        if candidate_id in result or candidate_id not in identifiers:
            raise ValueError(f"{name} contains duplicate or unknown candidates")
        if rank is not None and (
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
        ):
            raise ValueError(f"{name} ranks must be positive integers or None")
        result[candidate_id] = rank
    if set(result) != identifiers:
        raise ValueError(f"{name} must cover every candidate exactly once")
    concrete_ranks = tuple(rank for rank in result.values() if rank is not None)
    if unique_non_none_ranks and len(set(concrete_ranks)) != len(concrete_ranks):
        raise ValueError(f"{name} must describe one common ranking")
    return result


def _validated_rank_maps_by_width(
    values: tuple[tuple[int, tuple[tuple[str, int | None], ...]], ...],
    identifiers: set[str],
    legal_widths: tuple[int, ...],
    name: str,
    *,
    unique_non_none_ranks: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    seen_widths: set[int] = set()
    allowed_widths = frozenset(legal_widths)
    for item in values:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{name} must contain (width, rank-map) pairs")
        width, rank_map = item
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width not in allowed_widths
            or width in seen_widths
        ):
            raise ValueError(
                f"{name} widths must be unique members of the legal widths"
            )
        seen_widths.add(width)
        _validated_rank_map(
            rank_map,
            identifiers,
            f"{name}[{width}]",
            unique_non_none_ranks=unique_non_none_ranks,
        )


def _finite_unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


def plan_one_step_action(
    hypotheses: Sequence[CandidateHypothesis],
    questions: Sequence[str],
    *,
    current_turn: int,
    top_k: int,
    protocol_locked: bool = False,
    shared_reply_probability: float = 0.0,
) -> PlannedAction:
    """Choose a question and width by exact one-step expected utility.

    The current slate earns :func:`hit_utility` for hypotheses it exposes.
    Every non-exposed hypothesis deterministically enters its answer partition,
    where the next turn preserves the current order and shows the full ``top_k``.
    A protocol lock means that no current presentation can terminate, so all
    hypotheses survive and width zero becomes legal.

    ``shared_reply_probability`` models a protocol world where the next reply
    is candidate-independent, such as the still-possible first boundary
    decline after an initial browsing message. Exact-value ties preserve the
    caller's predeclared question order and then prefer the smallest width.
    On turn 10 there is no continuation: the full width is returned with no
    question.
    """

    ordered, ordered_questions, signature_maps = _validated_inputs(
        hypotheses,
        questions,
        current_turn=current_turn,
        top_k=top_k,
        protocol_locked=protocol_locked,
    )
    if isinstance(shared_reply_probability, bool) or not isinstance(
        shared_reply_probability,
        (int, float),
    ):
        raise ValueError("shared_reply_probability must be numeric")
    shared_probability = float(shared_reply_probability)
    if not isfinite(shared_probability) or not 0.0 <= shared_probability <= 1.0:
        raise ValueError("shared_reply_probability must be from zero through one")
    total_weight = sum(float(hypothesis.weight) for hypothesis in ordered)
    probabilities = tuple(
        float(hypothesis.weight) / total_weight for hypothesis in ordered
    )

    if current_turn == MAX_TURN:
        value = 0.0
        if not protocol_locked:
            value = sum(
                probability * hit_utility(current_turn, hypothesis.rank)
                for hypothesis, probability in zip(
                    ordered[:top_k],
                    probabilities[:top_k],
                )
            )
        return PlannedAction(question=None, width=top_k, value=value)

    widths = range(0 if protocol_locked else 1, top_k + 1)
    best: PlannedAction | None = None
    for question in ordered_questions:
        for width in widths:
            value = _action_value(
                ordered,
                probabilities,
                signature_maps,
                question=question,
                width=width,
                current_turn=current_turn,
                top_k=top_k,
                protocol_locked=protocol_locked,
                shared_reply_probability=shared_probability,
            )
            action = PlannedAction(question=question, width=width, value=value)
            if best is None or action.value > best.value:
                best = action

    if best is None:  # Defensive: validation requires at least one question here.
        raise ValueError("no valid action")
    return best


def _action_value(
    hypotheses: tuple[CandidateHypothesis, ...],
    probabilities: tuple[float, ...],
    signature_maps: tuple[dict[str, str], ...],
    *,
    question: str,
    width: int,
    current_turn: int,
    top_k: int,
    protocol_locked: bool,
    shared_reply_probability: float,
) -> float:
    immediate_value = 0.0
    survivors: list[tuple[int, float]] = []
    for index, (hypothesis, probability) in enumerate(zip(hypotheses, probabilities)):
        exposed = not protocol_locked and hypothesis.rank <= width
        if exposed:
            immediate_value += probability * hit_utility(
                current_turn,
                hypothesis.rank,
            )
        else:
            survivors.append((index, probability))

    informative_continuation = 0.0
    partition_ranks: dict[str, int] = {}
    for index, probability in survivors:
        signature = signature_maps[index][question]
        continuation_rank = partition_ranks.get(signature, 0) + 1
        partition_ranks[signature] = continuation_rank
        if continuation_rank <= top_k:
            informative_continuation += probability * hit_utility(
                current_turn + 1,
                continuation_rank,
            )

    shared_continuation = sum(
        probability * hit_utility(current_turn + 1, continuation_rank)
        for continuation_rank, (_, probability) in enumerate(
            survivors[:top_k],
            start=1,
        )
    )
    continuation = (
        (1.0 - shared_reply_probability) * informative_continuation
        + shared_reply_probability * shared_continuation
    )
    return immediate_value + continuation


def _validated_inputs(
    hypotheses: Sequence[CandidateHypothesis],
    questions: Sequence[str],
    *,
    current_turn: int,
    top_k: int,
    protocol_locked: bool,
) -> tuple[
    tuple[CandidateHypothesis, ...],
    tuple[str, ...],
    tuple[dict[str, str], ...],
]:
    if isinstance(current_turn, bool) or not isinstance(current_turn, int):
        raise ValueError("current_turn must be an integer")
    if not 1 <= current_turn <= MAX_TURN:
        raise ValueError("current_turn must be from 1 through 10")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(protocol_locked, bool):
        raise TypeError("protocol_locked must be a boolean")
    if isinstance(hypotheses, (str, bytes)) or not isinstance(hypotheses, Sequence):
        raise TypeError("hypotheses must be a sequence")
    if not hypotheses:
        raise ValueError("hypotheses must not be empty")
    if top_k > len(hypotheses):
        raise ValueError("top_k must not exceed the hypothesis count")
    if isinstance(questions, (str, bytes)) or not isinstance(questions, Sequence):
        raise TypeError("questions must be a sequence")

    question_tuple = tuple(questions)
    for question in question_tuple:
        if (
            not isinstance(question, str)
            or not question
            or question.strip() != question
        ):
            raise ValueError("questions must be non-empty normalized strings")
    if len(set(question_tuple)) != len(question_tuple):
        raise ValueError("questions must be unique")
    if current_turn < MAX_TURN and not question_tuple:
        raise ValueError("at least one question is required before the final turn")
    ordered_questions = question_tuple

    hypothesis_tuple = tuple(hypotheses)
    if any(
        not isinstance(hypothesis, CandidateHypothesis)
        for hypothesis in hypothesis_tuple
    ):
        raise TypeError("every hypothesis must be a CandidateHypothesis")
    ordered = tuple(sorted(hypothesis_tuple, key=lambda hypothesis: hypothesis.rank))
    expected_ranks = tuple(range(1, len(ordered) + 1))
    actual_ranks: list[int] = []
    candidate_ids: set[str] = set()
    signature_maps: list[dict[str, str]] = []
    total_weight = 0.0
    for hypothesis in ordered:
        if (
            not isinstance(hypothesis.candidate_id, str)
            or not hypothesis.candidate_id
            or hypothesis.candidate_id.strip() != hypothesis.candidate_id
        ):
            raise ValueError("candidate IDs must be non-empty normalized strings")
        if hypothesis.candidate_id in candidate_ids:
            raise ValueError("candidate IDs must be unique")
        candidate_ids.add(hypothesis.candidate_id)
        if isinstance(hypothesis.rank, bool) or not isinstance(hypothesis.rank, int):
            raise ValueError("hypothesis ranks must be integers")
        actual_ranks.append(hypothesis.rank)
        if isinstance(hypothesis.weight, bool) or not isinstance(
            hypothesis.weight,
            (int, float),
        ):
            raise ValueError("hypothesis weights must be numeric")
        weight = float(hypothesis.weight)
        if not isfinite(weight) or weight < 0.0:
            raise ValueError("hypothesis weights must be finite and nonnegative")
        total_weight += weight
        if not isinstance(hypothesis.answer_signatures, tuple):
            raise TypeError("answer_signatures must be an immutable tuple")
        answer_map: dict[str, str] = {}
        for pair in hypothesis.answer_signatures:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    "answer signatures must be (question, signature) pairs"
                )
            question, signature = pair
            if (
                not isinstance(question, str)
                or not isinstance(signature, str)
                or not signature
            ):
                raise ValueError(
                    "answer signature pairs must contain non-empty strings"
                )
            if question in answer_map:
                raise ValueError("a hypothesis cannot repeat a question signature")
            answer_map[question] = signature
        if set(answer_map) != set(question_tuple):
            raise ValueError(
                "each hypothesis must answer every allowed question exactly once"
            )
        signature_maps.append(answer_map)

    if tuple(actual_ranks) != expected_ranks:
        raise ValueError("hypothesis ranks must be unique and contiguous from one")
    if not isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("hypothesis weights must have a finite positive sum")
    return ordered, ordered_questions, tuple(signature_maps)
