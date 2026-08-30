# Phase 12 research plan

Status: frozen before candidate implementation, synthetic evaluation, validation,
or public confirmation.

## 1. Exact protected baseline

The protected submission remains `phase9-bounded-profile-residual-v1`.
`starter/agent.py` has SHA-256
`e533227f7467cb10f583519fdf281e253c64d0cf351d7b5194fd1381b5485b82`
and selects robust single-slot intent, conservative-early-other clarification,
completeness-adaptive route weights, deterministic Stage A, the bounded Phase 9
profile residual, stagnation-aware slates, and exact ranking reuse.

The presence of rejected Phase 10 and Phase 11 research code does not make it
reachable from `starter.Agent`. The current tree passes 391 tests and the exact
1,200-case Phase 9 ranking oracle has digest
`853f33454db9e3ce8c468a0b7ead525a174217c565e6a8a60ef65faf915476e1`.
Official frozen metrics are HR@10 `0.990000`, MRR `0.529558`, MTTC `3.065000`,
and TechnicalScore `0.812567`. On the deduplicated, public-excluded 996-case
development generator the same baseline recorded HR@10 `0.991968`, MRR
`0.534880`, MTTC `3.058233`, and TechnicalScore `0.815283`.

The complete per-file source manifest, immutable model/index checksums, runtime,
test commands, aggregate health, and rollback target are in
`docs/phase12_baseline_lock.json`. No public row was opened during this audit.

## Bottleneck audit

The active system's remaining headroom is rank and turn quality, not gross
recall. Its Stage-A retrieval term treats evidence from BM25 and dense retrieval
as additive even when the two routes are strongly redundant. An item appearing
in both routes receives both contributions in full; a high-ranked item from one
route can therefore be displaced by routine correlated agreement. The candidate
union is already complete and bounded, so this can be tested without another
search, embedding, document fetch, model, index, or persistent structure.

The prior Phase 11 parser candidate is closed: it improved all point estimates
but failed its frozen paired-bootstrap lower-bound gate. It will not be retuned
or rerun. The prior Phase 8 question-order candidate is also closed: candidate
facet separation was not a reliable proxy for whether the customer could reveal
a useful preference. The prior Phase 10 BM25 rescue is consumed and excluded
from Phase 12; neither its formula nor a repair of its public outcome is a Phase
12 alternative.

## 2. Ranked research backlog

Scores use 1 (low) through 5 (high). Decisions are frozen before Phase 12 code.

| Rank | Direction | Upside | Hidden robustness | Risk | Runtime cost | Reversible | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Conservative multi-slot intent reduction | 5 | 5 | 2 | 1 | 5 | Phase 11 consumed and rejected; no retry |
| 2 | Smoothed decision-theoretic clarification | 3 | 3 | 4 | 2 | 4 | Defer: answerability is unidentified and Phase 8 falsified the available proxy |
| 3 | Rank-only retrieval redundancy correction | 4 | 5 | 2 | 1 | 5 | Sole Phase 12 candidate |
| 4 | Statistically regularized reranking | 3 | 3 | 4 | 3 | 4 | Defer until fusion evidence is reliable |
| 5 | Confidence-preserving sequential slate optimization | 3 | 4 | 3 | 2 | 4 | Defer; current slate already protects recall |
| 6 | Search-versus-ask orchestration | 2 | 4 | 3 | 1 | 4 | Defer; ordinary turns must still recommend and exact reuse already avoids unchanged work |
| 7 | Worst-group robust optimization | 2 | 4 | 4 | 3 | 3 | Evaluation method, not the next runtime mechanism |

Three alternatives were considered conceptually. A candidate-facet expected
information-gain selector was rejected because product diversity does not
identify customer answer probability. Raw BM25/cosine entropy weighting was
rejected because route score scales are not calibrated and would widen the
device/runtime contract. Re-running or numerically repairing Phase 10 was
rejected because that public-tested candidate is consumed. Exactly one candidate
remains.

## 3. Single first hypothesis

### Hypothesis

Correcting the additive Stage-A route prior for observable BM25/dense route
redundancy will improve early rank and turn utility across generator-separated
data without losing baseline hits. The correction uses only the two already
available ordered route-ID lists and their frozen completeness-adaptive weights.
It is symmetric, rank-only, parameter-free with respect to outcomes, bounded,
and exact baseline when one route is empty, the routes are disjoint, or the two
routes are identical in order after normalization.

### Equations

Let `B` and `D` be the unique BM25 and dense Top-100 lists, with lengths `n_B`
and `n_D`, and let `K=60`. Define the route redundancy coefficient

```text
rho = 0                                      if min(n_B, n_D) = 0
rho = |B intersect D| / min(n_B, n_D)       otherwise
```

For candidate `i`, missing route evidence is zero and present evidence is

```text
b_i = (K + 1) / (K + rank_B(i))
d_i = (K + 1) / (K + rank_D(i))
x_i = w_B * b_i
y_i = w_D * d_i
```

where `w_B + w_D = 1` are the exact protected completeness-adaptive weights.
The incumbent additive evidence is proportional to `x_i + y_i`. The candidate
uses a redundancy-discounted submodular sum

```text
u_i = x_i + y_i - rho * min(x_i, y_i)
R_i = u_i / max_j(u_j)
```

and changes only the Stage-A retrieval term:

```text
beta = 0.20 + 0.25 * intent_completeness(state)
F_i = (1 - beta) * R_i + beta * requirement_satisfaction_i
```

The existing input order breaks exact score ties. Since `0 <= rho <= 1`,
`max(x_i,y_i) <= u_i <= x_i+y_i <= 1`; therefore all candidate evidence and
final scores are finite and bounded when inputs satisfy the frozen contract.
The negative cross-route interaction represents diminishing value from
correlated evidence, not a calibrated target probability.

### Assumptions and limits

- Route membership and rank are useful ordinal evidence; raw route scores are
  deliberately not assumed comparable.
- Route overlap measures redundancy, not correctness or target probability.
- Only complete, unique, bounded route lists whose union exactly equals the
  supplied fused candidates are eligible.
- The formula cannot recover a target absent from both Top-100 routes and does
  not change query construction, route depth, clause matching, or the catalog.
- Correlated agreement can sometimes be genuinely useful. The conjunctive
  paired-safety and confidence gates are therefore mandatory.

## 4. Frozen contract, tests, costs, and rollback

The machine-readable contract is `docs/phase12_experiment_contract.json`.
Implementation is restricted to one immutable non-default ranking policy and
the minimum service, test, oracle, and sequential harness plumbing needed to
exercise it. `starter.Agent` remains byte-identical until all promotion gates
pass.

Required deterministic checks include:

- exhaustive small route sets and at least 30,000 fixed-seed route/rank cases;
- coefficient bounds, score bounds, monotonicity, symmetry, deterministic ties,
  complete permutations, and exact replay;
- exact incumbent equivalence for empty, one-route, disjoint, identical-order,
  disabled-policy, malformed, and fail-closed cases;
- unchanged requirement satisfaction, beta, profile semantics, queries,
  questions, slates, cache dependencies, reset, and interleaved sessions;
- one BM25 call, one dense call, one candidate-document call, and one Stage-A
  attempt per fresh search, with exact reuse causing none;
- route, ranking, profile, slate, cache, fallback, and policy fault injection;
- the complete unit suite and the protected Phase 7/9 exact oracles.

The added computation is `O(n_B + n_D + n_F)` over at most 100 + 100 + 200
IDs, uses transient maps only, adds no model/API/embedding/search/document call,
and may retain no per-session route data. Candidate warm p95 and total wall time
must each be at most `1.05x` baseline; startup time must be at most `1.05x`,
additional startup RSS at most 1 MiB, and retained agent state at most 64 KiB.

Evaluation order is fixed: one unlabeled backend warm-up, candidate on the
deduplicated 996-case non-public development generator, protected baseline,
exact candidate replay, and independent explicit-policy candidate. Only a
fully passing frozen source may access the sealed 1,000-case independent
validation generator. Only a candidate passing both may access the 200-case
public confirmation once. Every model and evaluator action is sequential under
the six one-thread environment variables.

For every suite, HR@10 and MRR may not fall, MTTC may not rise, TechnicalScore
must rise strictly, MRR or MTTC must improve strictly, baseline-hit to
candidate-miss transitions must be zero, and the fixed scenario-stratified
10,000-replicate paired-bootstrap 95% lower bound for per-session utility delta
must be non-negative. Replay, API, call, fault, fallback, privacy, source-lock,
latency, startup, and memory gates are conjunctive.

Rollback is the default outcome: any failed check rejects the candidate, leaves
`starter/agent.py` at its protected SHA-256, does not inspect failed rows or
rerun that suite, and appends aggregate-only evidence. Promotion requires a
single explicit starter policy change followed by the independent-starter
exactness gate. No push is authorized.
