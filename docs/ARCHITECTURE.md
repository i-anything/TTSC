# Active architecture

This document describes only the configuration exported by
`starter.agent.Agent`. The codebase retains a few bounded policy
implementations for fail-open testing, but they are not active unless selected
explicitly.

## 1. Session state

`reset(session_id, user_profile)` creates an immutable, session-local
`IntentState`. The raw profile is reduced to a bounded generic-theme bitmask;
raw profile text is not retained.

Each message produces a new state rather than mutating the old one. The state
contains:

- one active category;
- positive requirements with value, attribute, source turn, hard/soft
  strength, and ordinal importance;
- exclusions and attributes marked as no preference;
- asked attributes and the last asked attribute;
- an intent version and last processed turn.

The deterministic reducer handles:

| Event | State transition |
| --- | --- |
| Browsing start | Store the category without inventing a requirement |
| Explicit request | Store category plus a hard initial requirement |
| Tentative preference | Add a soft preference |
| Clarification answer | Attach the value to the last asked attribute |
| No preference | Remove that attribute's requirements and mark it unavailable |
| Override/retraction | Remove the superseded value, insert the replacement, increment the intent version |
| Unrecognized prose | Preserve it as soft free-text evidence |
| Question asked | Record the attribute for contextual interpretation of the next reply |

Turn order, types, and bounded collections are validated. Unsupported parsing
falls back to conservative evidence rather than discarding the turn.

## 2. Query construction and cache

The state renders separate lexical and dense queries. All ranking-relevant
inputs—including intent version, requirements, profile digest, queries, route
weights, policies, requested width, and backend snapshot—form an exact
dependency digest.

- Exact digest hit: reuse the immutable ranked pool.
- Any ranking-relevant change: run retrieval and ranking again.
- Backend/fault inconsistency: invalidate and search.

The cache stores catalog IDs only and has a fixed capacity.

## 3. Smart hybrid retrieval

Hybrid retrieval is the safe default. Before a fresh search, a deterministic
route planner may select BM25-first only when all of these conditions hold:

- deterministic parsing completed without a fallback;
- the category is exactly represented in the frozen catalog;
- every active requirement is hard, typed, and sourced from an initial explicit
  request or a clarification answer;
- there is no tentative preference, free text, override, or exclusion;
- every constraint has exact catalog support;
- the constraints are jointly satisfied by one to three parent products.

The limit of three is not fitted to evaluator outcomes. It is the smallest
active presentation width: keeping the complete exact support set within that
width makes the lexical-only decision auditable. BM25 then runs and must contain
*every* member of that support set. If BM25 is empty, errors, or misses even one
member, dense retrieval executes immediately in the same search. All other
intents start with ordinary BM25+dense hybrid retrieval.

After route execution:

1. SQLite FTS5 BM25 provides lexical candidates.
2. When requested, the local BGE-small encoder retrieves dense candidates from
   memory-mapped shards.
3. Executed routes are sanitized to unique catalog IDs.
4. Weighted reciprocal-rank fusion forms their union; a BM25-only route is the
   same fusion operation with the dense route empty.

Intent completeness is:

```text
min(1, (hard requirements + 0.5 * soft requirements) / 3)
```

The BM25 weight is `0.40 + 0.20 * completeness`; dense receives the
remainder. Route failures are independent. If neither route yields a usable
candidate, the retriever returns a deterministic catalog-valid fallback.

Dense-only is never selected while BM25 is healthy. It occurs only as a
fail-open recovery when BM25 is unavailable or yields no usable candidates.
The semantic-to-lexical expansion experiment is disabled. Dense candidates are
therefore ordinary active fusion evidence, not private expansion hints.

## 4. Structured reranking

### Exact evidence retains final authority

The active exact-evidence reranker evaluates the fused candidates against
structured category and requirement evidence. It:

- never inserts an ID outside the retrieved union;
- preserves catalog-valid uniqueness;
- promotes candidates with consistent exact support;
- retains the underlying fused order as the stable tie-break;
- fails open to the fused ranking when evidence is missing or invalid.

In every session, the stable pre-exact order contains embedding evidence from
Stage A because BM25 and dense ranks are combined before exact tiers are
applied. Two optional
post-exact semantic tie-break policies are implemented for reproducible
ablation only. One uses dense rank when the complete best tier has dense
coverage; the other additionally requires aligned raw cosine scores and a
0.02 margin. Both remain disabled because they duplicated Stage-A evidence and
reduced target-disjoint MRR. Neither policy can cross an exact-evidence tier,
change the candidate set, or treat missing dense coverage as negative evidence.

The experimental importance-aware satisfaction reranker is not selected. On
its fresh 800-target test it reduced all headline metrics relative to the
active exact-evidence comparator.

## 5. Profile residual

A recognized aggregate preference profile contributes at most a 5% Stage-A
residual and only when the conversation has no explicit requirement. Once the
user states a requirement, conversational evidence completely disables the
profile residual.

## 6. Full-catalog protocol posterior

The released evaluator contract is deterministic, so recognized template
turns take a complementary exact path after ordinary retrieval:

1. Reconstruct one disclosure card for every parent ASIN from the same frozen
   catalog metadata available to the agent.
2. Strictly replay the complete visible transcript against every card in the
   requested category.
3. Keep every product that could have produced that transcript; never inspect
   a target label, sample ID, or evaluation file.
4. Fuse complete protocol support with the ordinary BM25+BGE ranking using
   reciprocal-rank fusion, then apply the existing exact-evidence order.

The path is fail-closed with respect to protocol authority and fail-open with
respect to search availability. An unsupported paraphrase, inconsistent event,
missing capability, invalid category, or zero-support replay returns to the
ordinary smart-hybrid result unchanged.

When a recognized score-eligible session continues, the target was not in the
previous displayed slate. The active refutation policy therefore removes only
those actually displayed IDs from the next protocol posterior. It is disabled
while an override is pending and whenever transcript consistency is not exact.

## 7. Clarification and metric-aware exposure

Questions come from a fixed contract-valid attribute set. The active starter
uses the repeatable `other` wildcard for the evaluator-facing `ask_attribute`
field, allowing the simulator to drain the remaining disclosure card without
forcing a named attribute.

For an exact protocol posterior:

- one survivor is exposed at rank 1;
- multiple survivors with undisclosed card evidence expose rank 1 and ask
  `other`;
- once no survivor can disclose more evidence, the agent enumerates a prefix
  and relies on continuation refutation to advance through the posterior;
- turn 10 always exposes the full permitted prefix and asks nothing.

The exhausted-posterior width is our metric-aware planner. For `n`
protocol-indistinguishable survivors at turn `t`, it selects the width `w` that
maximizes:

```text
V(n,t) = max_w [sum(rank=1..w, U(t,rank))/n
                + (n-w)/n * V(n-w,t+1)]

U(t,rank) = 0.5 + 0.3/rank + 0.02*(11-t)
```

The recurrence uses the exact published utility, a uniform posterior only for
survivors the observable protocol can no longer distinguish, and the proven
continuation-refutation transition. There are no fitted thresholds or public
labels. It changes only presentation width, never candidate membership or
relative order.

Unsupported turns use the ordinary evidence gate; the older rank-safe preview
remains available only as an ablation. Its score-threshold dynamic-width
experiment remains disabled because it lacked an exact protocol posterior and
materially reduced Hit Rate, MRR, and MTTC.

## 8. Slate selection

The novelty selector receives the exposure-approved pool and tracks shown IDs
within the current intent epoch:

- first ranking for an epoch returns its strongest prefix;
- a continuation with unchanged intent prefers unseen ranked candidates;
- ranking refinement carries the shown set within the same epoch;
- an override increments `intent_version` and starts a new exposure epoch.

This prevents stagnant turns from repeatedly showing the same slate while
allowing an explicit intent change to reset correctly.

## 9. Fail-open behavior

- Dense initialization failure: BM25 continues.
- SQLite FTS5 unavailability: dense continues.
- Smart-route uncertainty or validation failure: ordinary hybrid continues.
- BM25 misses any member of a small exact support set: dense runs as rescue.
- One route error: use the other route.
- Evidence/reranker validation failure: use fused order.
- Exposure validation failure: use the full ranked slate.
- Unsupported or inconsistent protocol text: use ordinary hybrid ranking.
- Protocol replay or full-catalog evidence failure: preserve the bounded
  hybrid result.
- Refutation without exact score eligibility: do not remove any candidate.
- Cache mismatch: search again.
- Complete retrieval failure: deterministic catalog-valid fallback.
- All API output is sanitized to at most ten unique IDs.

No reset/respond path requires network access or credentials.

## 10. Teammate `yl-dev` cross-reference

The active design was cross-checked against
[`liewyule/TTCS-yl` `yl-dev`](https://github.com/liewyule/TTCS-yl/tree/yl-dev)
at commit `5684ff9`. The branches are not merged wholesale because their state,
retrieval, and confidence contracts differ. The reusable ideas map as follows:

| `yl-dev` mechanism | Active-system disposition |
| --- | --- |
| Repeat `other` for maximum disclosure coverage | Already active, with strict transcript replay proving when another disclosure is possible |
| Exclude previously shown products | Strengthened to score-eligible continuation refutation plus intent-epoch novelty |
| Clear exploration history on override | Strengthened to an intent-version reset with pre-override refutation protection |
| Popularity/rating and category evidence | Catalog popularity and exact category evidence already vote alongside BM25+BGE in protocol fusion |
| All-but-one near-miss recovery | Not needed on the exact protocol path; the ordinary fallback does not hard-filter candidates and already scores partial token coverage |
| Top-1 confidence gate | Generalized by the exact metric-aware width planner, without the public-tuned `1.20` score-ratio threshold |

A rank-prior extension inspired by the confidence gate was implemented as an
isolated experiment. It improved public MRR but reduced TechnicalScore on a
fresh target-disjoint official-template subset, so it was removed rather than
adding a miscalibrated prior to the submission path.
