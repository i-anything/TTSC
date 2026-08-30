# Phase 15 v2: Decision-Aware Evidence Acquisition

## Decision record

The fixed-width Phase 15 v1 candidate is superseded before any catalog suite,
development suite, validation suite, or public-set execution. Its uncommitted
opt-in source and tests were removed before v2 implementation; its
one-`other`/four-result policy is not promotable and cannot be run as the Phase
15 hypothesis. Phase 13 remains the protected reference baseline. The live
worktree's `starter/agent.py` currently enables a separate, unpromoted
exact-evidence ranking experiment and does not match the locked Phase 13
starter hash; it must be restored or deliberately relocked before any Phase 15
data execution.

Phase 15 v2 now exists as an opt-in implementation candidate, but it remains
inactive, unhashed, unevaluated, and non-promotable. The protected constructor
does not build the protocol index or import the protocol-ranking and planning
modules. Nothing in this document is a score or improvement claim.

This decision follows a larger static audit of the public repository frontier.
The strongest checked-in artifact reports `0.979100`, while several other
repositories report scores from `0.925064` to `0.9627`. These are not an
official leaderboard. Evidence ranges from locally reproduced or checked JSON
artifacts to README-only self-reports, and most leading implementations are
closely coupled to the released simulator/card protocol and tuned on the 200
public sessions. The scores therefore motivate a general decision problem;
they do not supply code, constants, thresholds, training examples, or evidence
about the 800 hidden sessions.

No competitor code or constant is copied. No public row, target, failure,
trajectory, query, or product-level outcome may be inspected. Public aggregate
metrics are comparison-only and cannot be used for policy fitting.

## Frozen research question

At each turn, should the agent expose the next ranked candidate now, or preserve
the opportunity to improve its reciprocal rank by asking for evidence first?

Phase 15 v2 answers this with a one-step, receding-horizon belief-state
controller. It jointly chooses a permitted clarification question and a
recommendation prefix width. It is not a fixed schedule, a learned classifier,
or a target oracle.

## Candidate architecture

### 1. Protocol world model

Independently derive a disclosure transition model from only the official
evaluator, API contract, competition specification, and frozen catalog. For a
candidate product and a legal question, the model constructs the disclosure
signature the official simulator could emit from that product's structured
fields. The implementation must be deterministic, read-only, bounded to the
existing candidate pool, and separately unit-tested against hand-built product
cards.

The model may reason about a candidate as a possible target; it may never read
the evaluation target or any label at runtime. The source allowlist and hashes
are locked before target-disjoint execution.

The candidate retains a bounded, typed protocol event log independently of
`IntentState`. It records initial browsing, explicit and tentative evidence,
disclosures, overrides, no-additional-preference replies, boundary declines,
and need-attribute replies. Replaying that log prevents slot clearing,
deduplication, or reducer rewrites from erasing evidence that the official
protocol already disclosed. The log is capped at ten turn-ordered events, two
values per structured event, and bounded value lengths. A disclosure's
evaluator-visible payload is additionally preserved as one opaque bounded
string and replayed without guessing whether a semicolon separates two values
or occurs inside one catalog value. Overbound event values are rejected rather
than truncated, so lossy parsing can never authorize a dense skip.

`Requirement` also carries an explicit `hard` or `soft` strength independently
of provenance. Existing semantics remain the default: explicit, answer, and
override evidence is hard; tentative and free-text evidence is soft. Strength
participates in ranking, slate, and cache dependencies and is never inferred
from a candidate product's reconstructed hard/soft card fields. Free-text
evidence cannot be promoted to hard.

Once a protocol event log exists, it is authoritative for positive protocol
evidence; mirrored `IntentState` answer slots are not merged back into it. This
prevents an intent parser's semicolon splitting or duplicate-slot behavior from
changing the reconstructed protocol transcript. The initial tentative parser
also uses the first sentence boundary non-greedily, retaining every subsequent
sentence as the tentative payload rather than truncating multi-sentence clues.

### 2. Exact-evidence ranking tier

Keep Phase 13's BM25 and dense routes unchanged for the protected baseline and
for every unsupported, paraphrased, or inconsistent turn. In candidate mode,
BM25 always executes before the dense-route decision. Dense is skipped only
when all six pre-BM25 conditions pass—exact protocol message, consistent state,
exact catalog category, at least one independently verified exact structured
product constraint, no unparsed/free-text requirement, and no
tentative/override/contradiction
latch—and the BM25 pool intersects the bounded structurally valid candidate
set. An unavailable, empty, or failed BM25 route triggers one dense rescue;
healthy BM25 with zero structural support runs dense and fuses both routes. Add
a bounded, candidate-local structural evidence tier ordered by:

1. explicit-negative safety;
2. exact protocol-reply compatibility;
3. hard-budget non-violation, with unknown prices retained;
4. complete active-constraint coverage;
5. at-least-two exact hard-constraint coverage;
6. exact hard-constraint coverage count;
7. category compatibility;
8. exact non-budget field coverage;
9. exact phrase affinity;
10. total multi-constraint coverage;
11. general budget compatibility.

After those structural tiers, ties preserve the selected base-route order:
Phase 13 BM25+dense order on a full hybrid/fail-open turn and deterministic
BM25 order on an intentional dense-skip turn. Popularity remains a bounded
final tuple member only as a theoretical equivalence fallback: because base
ranks are distinct, popularity cannot displace a candidate with a different
base rank. It is not an active reranking signal and cannot override structural
evidence or selected-route order.

A conjunctive lexical route may inject additional candidates into the union,
but it may not hard-filter the broad pool or remove selected base candidates. Exact
evidence must not override an explicit negative constraint, a category
override, or a maximum budget. No weight or threshold may be selected from the
public 200 sessions.

The opt-in retriever builds a compact structured protocol table during the
same catalog pass that builds the existing BM25 store. It stores only the
reconstructed card fields, category, price, popularity, and aligned ID; it
does not retain a second raw catalog copy. Candidate evidence is synthesized
only from target category and reconstructed hard and soft card values; it does
not fetch or duplicate full FTS candidate text. Exact conjunctive rescue
matches are appended after the already-computed base ranking, then the union is
capped at 200 candidates. A validated prefix invariant prevents a rescue from
displacing a base candidate while constructing that union; subsequent
structural ranking may promote a rescue that survives the cap. The candidate
executes at most one dense call per search.

The routing decision uses logical invariants, not a fitted numeric threshold.
Its six pre-BM25 condition values, complete bounded protocol transcript, and
at most 200 catalog-derived structural-support IDs form the hashed candidate
routing dependency. The seventh condition is the observed BM25/support intersection and
is derived after BM25 executes. This forces fresh retrieval when
the derived track changes while permitting exact ranking reuse when no
ranking-relevant input changes. An intentional skip is recorded explicitly
instead of being confused with an unavailable or failed encoder.

When there is no product-specific evidence, including an open-ended browsing
start, a boundary decline, or a need-attribute reply, popularity is neutral and
the exact tier preserves the selected base-route order. If exact evidence has
zero consistent support or any candidate metadata is missing or invalid, the
selected base slate remains available unchanged. Only the hybrid/fail-open
route claims exact Phase 13 ordering.

### 3. Belief state

Build a normalized belief over the best protocol-consistent structural
evidence tier. Within that tier, preserve the deterministic exact-evidence
order and assign reciprocal-rank mass `1/r`, normalized by the tier's harmonic
sum. Products outside the best tier receive no planning mass. This fixed,
label-free transform is finite, deterministic, monotone in the resulting
evidence order, and defined for every non-empty consistent tier. It is a
decision distribution, not a trained or calibrated relevance model. The fixed
reciprocal-rank transform and the browsing/boundary prior are never fitted.
Target-disjoint runs may publish aggregate Brier score and expected calibration
error only as diagnostics. Those diagnostics cannot gate promotion, alter a
belief, or trigger any post-outcome calibration. The diagnostic observer must
return the ranking payload unchanged. Any observer failure is retained and
raised only after the official evaluator call, so instrumentation cannot alter
the evaluator-visible recommendation or question.

Each observed answer conditions this belief through the independently derived
disclosure signature. Exposed candidates are represented explicitly so the
planner conditions on every candidate already shown in the session, rather
than only the current intent epoch. Those IDs are excluded from both belief
support and the planned available order, so a surviving protocol session does
not repeat an already exposed candidate.
Contradictions, retractions, and category overrides invalidate stale evidence
before the next decision.

The initial tentative protocol form creates a sticky override lock in session
state. The lock remains active until an explicit override event is observed,
even if the ordinary intent reducer deduplicates the eventual value against an
existing slot. While locked, an applied planner may choose zero-width
questioning. A fail-open response may still expose the protected slate; every
such product is recorded in session-wide shown memory before later planning.

Routing has a separate conservative latch: observing either an initial
tentative form or any override keeps dense retrieval enabled for the rest of
that session, until reset. Clearing the planner's pre-override lock therefore
does not enable BM25-only retrieval after an override.

### 4. Expected-utility planner over question and width

For turn `t`, belief `b`, legal question `q`, and every prefix width
`k in 1..normalized_top_k`, evaluate the official-score utility of the joint
action `(q, k)`. A zero-width action is additionally legal while the protocol
itself prevents a hit, such as before an intent override becomes active. Let
`i_r` be the product at rank `r`, `A(q)` the possible disclosure answers under
the world model, and `R_official(t, r)` the hit utility defined by the checked
official evaluator. The locked one-step value is:

```text
Q_t(b, state, q, k) =
    sum_(r=1..k) b(i_r) * R_official(t, r)
    + sum_(a in A(q)) P(a, target not exposed | b, q, k)
        * U_full_(t+1)(conditioned candidates for a)
```

`U_full` exposes up to the official `top_k` at the next turn in the current
order within each deterministic answer partition. The controller replans from
new evidence on every real turn. The terminal value at the official turn limit
is the immediate full-width exposure value.
The planner uses only candidate hypotheses, observed conversation state, and
official scoring semantics. It cannot access the real target. There is no
catalog-scale rollout, Monte Carlo sampling, external model call, learned value
function, or recursively accumulated model error.

The candidate action set includes every currently legal named question,
and `other` when legal. `other` is first in the frozen question order and wins
only an exact expected-utility tie; a named question wins whenever its value is
strictly greater. Within one question, an exact value tie selects the smallest
legal width. These tie-breaks and degenerate-belief behavior are locked before
data execution. The selected width affects only the returned prefix and
successfully shown-product memory. A positive requested width bounds legal
planner/presentation widths but does not change the complete route pool or
Stage-A computation; a non-positive width skips retrieval entirely.

On turn one after an initial browsing message, two protocol-compatible worlds
remain possible: ordinary candidate-conditioned browsing and a
candidate-independent boundary decline on the next question. The planner uses
a fixed symmetric prior (`0.5` each) and averages those continuation values;
this is a protocol prior, not an estimate fitted from sessions. When a plan is
applied, execution uses the planner's validated available-candidate order, not
the unconditioned exact ranking, so shown candidates cannot re-enter through
the presentation layer.

The value function is deliberately a one-step approximation. It replans after
the next real observation, but it does not solve the complete remaining-turn
partially observable decision process. Multi-step information interactions can
therefore be valued imperfectly; this is a declared residual, not hidden model
accuracy.

### 5. Logical fail-open dual mode

Protocol mode is enabled only while all of the following logical invariants
hold:

- every observed answer is recognized by the official transition model;
- the observation is compatible with at least one unexposed candidate
  hypothesis;
- no unresolved contradiction, retraction, or category override remains;
- the belief is finite, normalized, non-empty, and derived from a complete
  ranked candidate pool;
- the requested width and remaining-turn state satisfy the API contract.

There is no fitted confidence threshold. Any invariant failure detected before
retrieval disables aggressive withholding and selects broad BM25+dense
retrieval. It applies explicit hard constraints conservatively and returns the
full normalized recall-safe slate. A later world-model, planner, or metadata
exception must preserve a valid full-width base result without corrupting
session state. Such a fault is still a promotion failure; it cannot be hidden
as successful orchestration.

Protocol evidence only reorders or narrows output after candidate checks
succeed. Unsupported and paraphrased turns therefore retain the exact Phase 13
ranking, width, and protected question behavior. A high-confidence structural
turn performs no second dense query; if BM25 is empty or errors, its single
allowed dense call is used as a rescue.

Startup and import behavior is fail-open as well. The protected Phase 13
constructor neither builds the protocol table nor imports the protocol world
model, exact-ranking, or planner modules. In opt-in mode, failure while creating
or populating the protocol table or its five explicit indexes disables the
protocol capability while preserving the normal Phase 13 BM25+dense backend.
A later lazy-import, capability, metadata, world-model, or planner failure keeps
the already-computed selected-base response for that turn and must not corrupt
session state. That is exact Phase 13 behavior on a hybrid/fail-open turn and a
full-width BM25 base response after an intentional dense skip. A fresh-process
startup probe asserts that no Phase 15 candidate module is present after
protected construction and that candidate construction does load the protocol
module.

Candidate-only telemetry is fixed-cardinality and aggregate-only: outcome
reason counts, question-action counts (including no question), width counts
for zero through ten, total requested products, and total presented products.
It contains no product IDs, messages, targets, or per-session trajectories.
For local demos only, `last_action_trace(session_id)` exposes the latest
sanitized route/action decision. It is not part of the evaluator response or
published benchmark payload and contains no IDs, candidate lists, query text,
profiles, targets, labels, or scores.

## Compared policies and exactness checks

Only two implemented policies are compared:

- `phase13`: the protected baseline;
- `combined_v2`: exact evidence, belief/planner, and logical fail-open dual
  mode, opt-in and the only potentially promotable Phase 15 policy.

Deterministic replay and independent construction of `combined_v2` are
exactness checks, not additional policies or score-bearing arms. The proposed
`exact_evidence_only` arm is implemented as a separate, non-promotable
ablation and is currently selected by the live uncommitted starter; it cannot
be combined with `combined_v2`. The `planner_only` arm remains deferred and
unimplemented. Neither component arm is score-bearing Phase 15 evidence.

## Target-disjoint validation

Before any candidate execution, freeze generator source, suite hashes, target
fingerprints, source hashes, and the manual-message protocol. The generator
must hash and exclude the union of targets from public, legacy development,
legacy validation, and the frozen Phase 14 fresh suite before selecting any
Phase 15 target. Those exclusion sources may overlap one another and are not
promotion evidence. All six newly generated Phase 15 sources must be mutually
target-disjoint and disjoint from that entire union. Selection is deterministic
textual salted hashing under the frozen
`phase15-protocol-robustness-target-disjoint-v2` contract; there is no numeric
random seed or label-conditioned sampler. Publish only aggregate metrics and
paired aggregate counts; never inspect row-level output.

The execution harness re-hashes the generator, official evaluator, Phase 14
fresh-suite builder, every locked source, and the aggregate-only robustness
manifest. It recomputes case and target fingerprint sets, verifies actual
pairwise target disjointness, and independently cross-checks every generated
output plus the selected and forbidden target proofs in the builder manifest.
Each later gate also binds its prerequisite artifact to the current
implementation-lock hash, suite-lock hash, and expected suite identity.

The ordered gates are:

1. `fresh_exact`;
2. `paraphrase_fail_open`;
3. `card_perturbed`;
4. `scenario_balanced`;
5. newly generated `target_disjoint_development`;
6. newly generated `target_disjoint_validation`;
7. one public confirmation, run once after every internal gate passes.

Every generated suite is balanced across apparel, footwear, and
jewelry/accessories and across catalog-only head, torso, and tail popularity
strata. The four focused robustness sources contain 36 unique targets each;
the generated development and validation sources contain 108 unique targets
each and are scenario-balanced. `paraphrase_fail_open` uses deterministic prose
outside the released templates. `card_perturbed` covers constraint order,
optional soft-card absence, and opaque-semicolon serialization while retaining
bounded target semantics. `scenario_balanced` covers buying, browsing,
boundary, and intent override. The legacy development/validation files remain
hash-locked exclusion and comparison inputs only. Suite files and the
aggregate-only manifest remain pending and may not be generated against real
inputs until all prerequisite source locks are materialized and real-data
execution is explicitly requested.

## Promotion gates

Every suite is one-shot and sequential. A failed gate stops the experiment;
there is no post-outcome threshold, weight, question-order, action-set, or
fallback change.

- no Phase 13 hit becomes a `combined_v2` miss;
- Hit Rate@10 and MRR do not regress;
- MTTC does not increase;
- TechnicalScore does not regress on any promotion suite;
- TechnicalScore strictly improves on fresh exact, target-disjoint development,
  and target-disjoint validation;
- MRR or MTTC strictly improves;
- paired bootstrap 95% lower bound for TechnicalScore delta is non-negative
  on every robustness suite and strictly positive on development and
  validation;
- protocol mode improves the protocol-consistent stratum, while fail-open mode
  does not regress paraphrased or card-perturbed strata;
- replay and independent construction are exact;
- zero response exception, invalid API response, runtime network attempt,
  parser, state, protocol-model, ranking, cache, planner, retrieval, embedding,
  model, or API fault;
- no added external model or API call;
- warm p95, total wall time, startup RSS, and retained session-state limits in
  the experiment contract pass;
- only aggregate counters and metrics are published.

Final promotion authority reloads the seven fixed result paths itself,
revalidates the current implementation and suite locks, binds every result to
its exact generated source fingerprint, and requires the complete predeclared
decision-gate schema. It independently validates and cross-checks the aggregate
metrics, paired transitions, health counters, call accounting, exactness,
calibration, latency, startup, retained-state, and reproducibility sections,
then recomputes every gate. Prerequisite suites pass through the same validator.
A caller-supplied `advance: true` payload is never sufficient.

Aggregate target-disjoint Brier score and expected calibration error may be
reported beside these gates, but neither is a gate and neither may change the
fixed beliefs or policy.

Startup time and RSS are probed in both protected-then-candidate and
candidate-then-protected orders. The gate conservatively uses the slowest and
largest candidate observations against the fastest and smallest protected
observations. Retained candidate-state overhead is measured directly from the
bounded `_protocol_*` session fields; whole-agent retained-size delta remains a
diagnostic and is not allowed to conceal candidate state behind unrelated
baseline variance.

The checked `0.979100` public artifact is a stretch comparison, not a fitting
gate. TechnicalScore is capped at `1.0`, so the largest mathematical margin
over it is `0.020900`; a `0.1` lead is impossible. Hidden-test robustness takes
priority over matching any public protocol-specialized score.

## Thermal-safety constraint

The user authorizes at most three local CPU processes for verification, with
MPS/GPU disabled and per-process numerical-library thread pools capped at one.
That resource allowance does not unlock real-data execution. Until the source,
suite, and manifest locks are materialized, Phase 15 v2 remains limited to
static design, source edits, formatting checks, and focused unit tests. It must
not scan the full catalog, build indexes, run ONNX or local-model inference,
generate catalog-scale signatures, or execute a data suite or public evaluator.

## Predeclared operational ceilings

Before any catalog or session data execution, the maximum acceptable candidate
overheads are frozen at: warm per-turn p95 ratio `1.10`, total wall-time ratio
`1.10`, startup-time ratio `1.10`, additional startup and post-warm peak RSS
`67108864` bytes (64 MiB), and additional retained session state `65536` bytes. These are
rejection ceilings, not expected results. The candidate must also make zero
additional external model/API calls, create no duplicate raw catalog copy, and
complete replay and independent construction exactly.

## Declared residuals before execution

- The receding-horizon value is a one-step approximation, not a solved
  remaining-session policy.
- Candidate startup time and RSS for the protocol table plus five explicit
  indexes are unmeasured and must pass the frozen operational ceilings.
- The selected-base-first union is capped at 200 candidates, so an exact rescue
  beyond that bound cannot enter protocol reasoning.
- The fixed `0.5` browsing/boundary prior and reciprocal-rank belief are
  uncalibrated; aggregate Brier/ECE diagnostics cannot modify them.
- Preserving semicolon payloads opaquely prevents unsafe false splitting, but
  can lose recall when two genuinely separate values would otherwise match
  independently.
- Conditional dense routing is label-free but unpromoted; protected Phase 13
  routing remains unchanged until every quality and resource gate passes.
