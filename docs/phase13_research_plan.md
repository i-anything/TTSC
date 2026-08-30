# Phase 13 research plan

Status: frozen before candidate implementation, generator evaluation,
validation, or public confirmation.

## 1. Exact protected baseline

The protected submission remains `phase9-bounded-profile-residual-v1`.
`starter/agent.py` has SHA-256
`e533227f7467cb10f583519fdf281e253c64d0cf351d7b5194fd1381b5485b82`
and selects robust intent parsing, conservative clarification,
completeness-adaptive fusion, deterministic Stage A, the bounded Phase 9
profile residual, stagnation-aware slates, and exact ranking reuse. Rejected
Phase 10, 11, and 12 mechanisms remain unreachable from `starter.Agent`.

The final Phase 12 tree passes 412 tests, and the exact 1,200-case Phase 9
ranking oracle has digest
`853f33454db9e3ce8c468a0b7ead525a174217c565e6a8a60ef65faf915476e1`.
Official frozen metrics remain HR@10 `0.990000`, MRR `0.529558`, MTTC
`3.065000`, and TechnicalScore `0.812567`. On the deduplicated,
public-excluded 996-case development generator, the same protected baseline is
HR@10 `0.991968`, MRR `0.534880`, MTTC `3.058233`, and TechnicalScore
`0.815283`.

The complete source, asset, runtime, metric, verification, and rollback
manifest is `docs/phase13_baseline_lock.json`. No public row or prior failed
row was opened during this audit.

## 2. Bottleneck audit and ranked backlog

The protected slate remembers shown products only while the entire ranking
signature is unchanged. An ordinary answer that adds a requirement changes
the query and ranking signature, resets that memory, and permits products just
shown on the preceding turn to consume the new top-10 again. The intent reducer
already exposes a stricter semantic boundary: `intent_version` increments on a
recognized replacement or override, but remains stable for ordinary
refinement. This boundary can preserve novelty without another search, model,
score, product field, or learned parameter.

Scores use 1 (low) through 5 (high), frozen before Phase 13 code.

| Rank | Direction | Upside | Hidden robustness | Risk | Runtime cost | Reversible | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Intent-epoch continuation novelty | 4 | 5 | 2 | 1 | 5 | Sole Phase 13 candidate |
| 2 | Statistically regularized reranking | 3 | 3 | 4 | 3 | 4 | Defer: reliable calibration is still unidentified |
| 3 | Smoothed decision-theoretic clarification | 3 | 3 | 4 | 2 | 4 | Defer: customer answerability remains unidentified |
| 4 | Search-versus-ask orchestration | 2 | 4 | 3 | 1 | 4 | Defer: recommendations are required on ordinary turns |
| 5 | Worst-group robust optimization | 2 | 4 | 4 | 3 | 3 | Evaluation method, not a runtime mechanism |

The candidate is not a repair of Phase 12 and does not use its row, rank, or
subgroup outcomes. Phase 12's formula remains closed. A fixed anchor quota,
score-margin threshold, probabilistic exploration rate, and catalog-facet
diversification were rejected before implementation because each introduces an
unidentified parameter or proxy. Exactly one parameter-free candidate remains.

## 3. Single hypothesis and equations

### Hypothesis

Carrying the bounded set of previously shown products across ranking changes
within the same explicit intent epoch will reduce repeated exposure and improve
turn utility without reducing rank quality or hits. A recognized intent
replacement increments `intent_version` and resets history exactly, so products
shown for an obsolete intent become eligible again.

### Frozen selection rule

At turn `t`, let `P_t` be the unique ordered ranked pool, `k` the requested
slate size, `e_t = state.intent_version`, and `H_(t-1)` the bounded shown-ID
state. Let `e_(t-1)` be the first field of the prior canonical ranking
signature. Define the eligible prior history

```text
A_t = empty                              if no prior signature exists
A_t = empty                              if e_t != e_(t-1)
A_t = H_(t-1) intersect P_t             otherwise, preserving H order
```

Define unseen candidates in ranked order and deterministic backfill:

```text
U_t = [p in P_t where p not in A_t]
L_t = first k unique IDs from (U_t concatenated with P_t)
H_t = unique(A_t concatenated with L_t)
```

The existing 200-candidate and 10-result bounds remain unchanged. Exact Phase
9 slate behavior is used for an empty pool, non-positive limit, first slate,
unchanged signature, changed intent epoch, or malformed candidate-only epoch
metadata. The only treatment case is a changed ranking signature with the same
valid intent epoch.

### Safety argument

Under the conversational evaluation contract, reaching another eligible turn
means the active target was not in an earlier accepted slate. Before an
override, recommendations are deliberately ineligible; the canonical override
increments `intent_version`, causing an exact reset. Therefore, whenever
history is carried for an eligible target `g`, `g` is not in `A_t`. Removing
members of `A_t` ahead of `g` can only preserve or improve `g`'s visible rank,
and deterministic backfill preserves slate cardinality. This is a conditional
dominance property, not a claim that every real customer rejects every shown
item; the hidden-data gates remain conjunctive.

### Assumptions and limits

- `intent_version` is the authoritative replacement boundary already used by
  the active intent state; no message text is reparsed by the slate layer.
- User continuation is weak negative feedback against immediate repetition,
  not a product relevance label and not retained beyond the session.
- The mechanism cannot help a target absent from the fused ranked pool and does
  not change retrieval, ranking scores, questions, or query construction.
- Shown state remains a unique subset of the current pool with at most 200 IDs.
- Any candidate-only validation failure returns exact stagnation-aware output
  and is ineligible for promotion.

## 4. Frozen tests, costs, evaluation, and rollback

The machine-readable contract is `docs/phase13_experiment_contract.json`.
Implementation is restricted to one non-default slate policy and the minimum
service, test, oracle, and sequential harness plumbing. `starter.Agent` remains
byte-identical until every data gate passes.

Required deterministic checks include:

- exhaustive bounded small-pool sequences and at least 30,000 fixed-seed
  randomized state/signature/pool transitions;
- exact incumbent equivalence for empty, zero-limit, first-turn,
  unchanged-signature, changed-epoch, disabled-policy, malformed, and
  fail-closed cases;
- no duplicate recommendations, complete deterministic backfill, state bounds,
  epoch reset, cross-refinement novelty, and abstract continuation dominance;
- unchanged retrieval, ranking, profile, question, query, cache, and reset
  semantics, including interleaved sessions and fault injection;
- one BM25, dense, candidate-document, and Stage-A call per fresh search, with
  exact reuse making none;
- exact replay, independent explicit-policy construction, the complete unit
  suite, and protected Phase 7 and 9 oracles.

The additional slate computation is `O(|P_t| + |H_(t-1)|)` over two bounded
200-ID sequences, adds no model/API/embedding/search/document call, and retains
no field beyond the existing `SlateState`. Candidate warm p95, total wall time,
and startup time must each be at most `1.05x` baseline; additional startup RSS
must be at most 1 MiB and additional retained agent state at most 2 MiB. The
retained-state cap covers at most 200 existing ID references per evaluated
session without authorizing a new state field.

Evaluation order is fixed: one label-free backend warm-up, candidate on the
deduplicated 996-case non-public development generator, protected baseline,
exact candidate replay, and independent explicit-policy candidate. Only a
fully passing source may access the sealed 1,000-case independent validation
generator; only a candidate passing both may access the 200-case public
confirmation once. All actions are sequential under the six one-thread
environment variables.

For every suite, HR@10 and MRR may not fall, MTTC may not rise, TechnicalScore
must rise strictly, MRR or MTTC must improve strictly, baseline-hit to
candidate-miss transitions must be zero, and the fixed scenario-stratified
10,000-replicate paired-bootstrap 95% lower bound for per-session utility delta
must be non-negative. Replay, call, fault, fallback, token, privacy, source-lock,
latency, startup, memory, and generator-separation gates are conjunctive.

Rollback is the default outcome. Any failed gate rejects the candidate, leaves
`starter/agent.py` at its protected SHA-256, skips all later suites, never
inspects a failed row, and appends aggregate-only evidence. Promotion requires
one explicit starter slate-policy change after all three suites, followed by an
independent-starter exactness gate. No push is authorized.
