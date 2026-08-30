# Phase 11 research plan

Status: planning freeze complete; candidate implementation had not started when
this document and its experiment contract were written.

## 1. Exact protected baseline

The protected submission is `phase9-bounded-profile-residual-v1`. The public
adapter is byte-for-byte the Phase 9 adapter (`starter/agent.py` SHA-256
`e533227f7467cb10f583519fdf281e253c64d0cf351d7b5194fd1381b5485b82`) and
does not select the rejected Phase 10 rescue policy.

| Component | Protected policy |
|---|---|
| Intent reducer | `robust` |
| Clarification | `conservative_early_other` |
| Fusion | `completeness_adaptive` |
| Ranking | deterministic Stage A |
| Profile | `phase9-bounded-profile-residual-v1` |
| Slate | `stagnation_aware` |
| Orchestration | `exact_ranking_reuse` |
| Encoder/index | pinned INT8 BGE-small, 384 dimensions |

The frozen public metrics are HR@10 `0.990`, MRR `0.529558`, MTTC `3.065`,
efficiency `0.7935`, and TechnicalScore `0.812567`, with zero reported tokens.
The current single-thread suite passes 339 tests, and the 1,200-case Phase 9
oracle retains digest
`853f33454db9e3ce8c468a0b7ead525a174217c565e6a8a60ef65faf915476e1`.
The complete source, asset, runtime, and health lock is
`docs/phase11_baseline_lock.json`.

The two supplied 1,200-row files each contain the same 200 released public
rows. Content-level fingerprinting also found four internal duplicate cases in
the plain generator. After public exclusion and deterministic deduplication,
the development generator has 996 cases, the sealed scenario-aware generator
has 1,000 cases, and their non-public overlap is zero. No row content was
printed or manually inspected. See `docs/phase11_dataset_audit.json`.

## Current bottleneck audit

The strongest justified bottleneck is the intent reducer, before retrieval:

- `IntentState` and both query renderers already support multiple active
  `Requirement` objects and typed material, color, size, style, brand, budget,
  use-case, feature, and residual clues.
- `classify_requirement()` nevertheless maps one entire incoming value to one
  attribute using first-match priority. A compound value can therefore be
  stored under only one query section and treated as one indivisible Stage-A
  clause.
- Unmatched follow-ups become one whole free-text requirement. Explicit
  negatives are not populated into `IntentState.excluded` and can therefore be
  embedded as if they were positive evidence.
- The downstream fusion, Stage-A, cache dependency, and slate code already
  react correctly to multiple requirements; no new model, index, catalog pass,
  or candidate-side representation is needed to test a better reducer.
- Phase 4 showed that better structured requirement satisfaction can materially
  improve ranking. Phase 8 showed that changing question order before the
  underlying state is more reliable can regress quality. Phase 10 showed
  lexical-ranking headroom but failed its operational gates. Neither rejected
  outcome is used to tune the parser proposed here.

The remaining public headroom is primarily rank and turn quality rather than
gross recall: HR@10 is already `0.990`, while MRR is `0.529558` and MTTC is
`3.065`. A parser improvement is valuable only if it improves those measures
without weakening the protected single-slot behavior.

## 2. Ranked research backlog

Scores are qualitative and fixed before Phase 11 implementation. Upside,
hidden robustness, and reversibility use 1 (low) to 5 (high); implementation
risk and runtime cost use 1 (low) to 5 (high).

| Rank | Direction | Upside | Hidden robustness | Risk | Runtime cost | Reversibility | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Lossless conservative multi-slot intent reduction | 5 | 5 | 2 | 1 | 5 | Phase 11 |
| 2 | Retrieval-uncertainty signals with bounded monotonic fusion | 4 | 4 | 3 | 1 | 4 | Next only after parser |
| 3 | Confidence-preserving submodular/MMR slate selection | 3 | 4 | 3 | 2 | 4 | Defer; protect top candidate |
| 4 | Smoothed decision-theoretic clarification | 3 | 3 | 4 | 2 | 4 | Defer after Phase 8 rejection |
| 5 | Regularized synthetic-only reranking | 3 | 3 | 4 | 3 | 4 | Defer; greater validation burden |
| 6 | Search-versus-ask value orchestration | 2 | 4 | 3 | 1 | 4 | Defer; exact reuse is already strong |
| 7 | Robust worst-group optimization | 2 | 4 | 4 | 3 | 3 | Evaluation method, not first runtime change |

Dataset deduplication, generator separation, paired confidence intervals, and
McNemar transitions are mandatory experiment infrastructure, not a candidate
feature and not counted as an intelligence improvement.

## 3. Recommended first hypothesis

### Hypothesis

When one user turn contains two or more conservatively separable constraints,
a lossless span ledger that stores each unambiguous positive, exclusion,
attribute-specific no-preference, or explicit correction as its own state delta
will improve retrieval/ranking and reduce wasted clarification turns across two
independent synthetic generators, while returning the exact Phase 9 state for
every single-slot, ambiguous, oversized, or malformed message.

The candidate is parameter-free with respect to outcomes. It uses only fixed
language structure and the already frozen attribute cue sets. It does not use
the catalog, profiles, candidates, route scores, scenario labels, targets, or
evaluation results.

### Decision rule

Let `B(s,m,t)` be the exact protected Phase 9 reducer. For each bounded message
payload, the candidate constructs ordered atoms

```text
g_j = (operation_j, attribute_j, value_j, source_j, span_j)
operation in {ADD, EXCLUDE, CLEAR}
attribute in allowed attributes or untyped residual
```

For an unlabeled span `x`, let

```text
D(x) = {a : the frozen high-confidence cue detector for attribute a matches x}
```

An explicit allowed label fixes its attribute. Otherwise a span is typed only
when `|D(x)| = 1`; `|D(x)| > 1` is ambiguous, and `|D(x)| = 0` is preserved as
an untyped residual rather than discarded.

The span ledger is eligible only when every alphanumeric character of the
payload belongs to a captured value or a fixed control/separator span, at least
two semantic atoms survive, all bounds and contradiction checks pass, and the
result differs from an unsafe one-slot interpretation. The state transition is

```text
T(s,m,t) = fold_left(apply_atom, s, g_1..g_n)   if Eligible(m,s,t)
         = B(s,m,t)                              otherwise
```

Atoms are applied in source order, so the latest explicit statement in the
same message wins. Negated values are removed from positive requirements and
stored only in `excluded`; they are never rendered into the positive dense or
BM25 query. A destructive correction, exclusion, clear, or full override
increments `intent_version` exactly once. The existing renderer, completeness
formula, fusion, ranking, profile, slate, and exact-cache dependency then operate
unchanged on the improved state.

This proposal deliberately does not add negative candidate penalties. Catalog
metadata missingness makes such a penalty unsafe as part of a parser experiment.
Phase 11 only prevents explicit negatives from being treated as positive query
evidence and preserves them for a separately tested future policy.

## 4. Frozen experiment, tests, and rollback

The machine-readable contract is
`docs/phase11_experiment_contract.json`. Its key controls are:

- Maximum 2,048 input characters, eight atoms per turn, 256 characters per
  atom, 24 active requirements, and 16 exclusions. Any candidate-bound breach
  returns exact Phase 9.
- No changes to questions, retrieval, fusion, ranking, profiles, slates,
  orchestration, model assets, index artifacts, runtime requirements, or the
  evaluator.
- Existing canonical and robust single-slot families must be state/query exact
  under the candidate.
- Synthetic tests cover multi-slot positives, safe conjunctions, negation,
  exclusions, corrections, no-preference, overrides, pronouns, residual text,
  malformed inputs, interleaved sessions, reset, search/reuse, fault injection,
  and exact baseline fallback.
- A separately expressed oracle exhausts small atom partitions and at least
  20,000 fixed-seed valid/malformed messages. All candidate outputs and traces
  must replay exactly.
- The 996-case plain-generator development suite is evaluated first. Only a
  fully locked passing candidate may access the 1,000-case sealed
  scenario-aware validation suite, and only a candidate passing both may run
  once on the 200-case public confirmation.
- Each suite is candidate, protected baseline, exact candidate replay, and an
  independent explicit-policy `starter.Agent` instance, strictly sequential and
  single-threaded. The submitted starter default remains Phase 9 until every
  promotion gate passes.
- For each suite: HR@10 and MRR cannot decrease, MTTC cannot increase,
  TechnicalScore must improve strictly, at least one of MRR/MTTC must improve,
  baseline-hit to candidate-miss transitions must be zero, and a fixed-seed
  scenario-stratified paired bootstrap must have a non-negative 95% lower bound
  for per-session TechnicalScore utility delta.
- Replay, calls, faults, API contract, privacy, warm p95, wall time, startup,
  and retained-memory bounds are conjunctive gates.

If any test or suite gate fails, the Phase 11 candidate is rejected. The public
starter is restored to the exact locked Phase 9 default, no public failure rows
are inspected, no formula or cue is repaired from outcomes, no failed suite is
rerun, and only aggregate rejection evidence is added. If promoted, the policy
switch is one explicit default change after all data gates; otherwise
`benchmarks/phase9.json` remains official.
