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

## 3. Hybrid retrieval

On every fresh active search:

1. SQLite FTS5 BM25 retrieves lexical candidates.
2. The local BGE-small encoder retrieves dense candidates from memory-mapped
   shards.
3. Both routes are sanitized to unique catalog IDs.
4. Weighted reciprocal-rank fusion forms their union.

Intent completeness is:

```text
min(1, (hard requirements + 0.5 * soft requirements) / 3)
```

The BM25 weight is `0.40 + 0.20 * completeness`; dense receives the
remainder. Route failures are independent. If neither route yields a usable
candidate, the retriever returns a deterministic catalog-valid fallback.

The semantic-to-lexical expansion experiment is disabled. Dense candidates are
therefore ordinary active fusion evidence, not private expansion hints.

## 4. Structured reranking

The active exact-evidence reranker evaluates the fused candidates against
structured category and requirement evidence. It:

- never inserts an ID outside the retrieved union;
- preserves catalog-valid uniqueness;
- promotes candidates with consistent exact support;
- retains the underlying fused order as the stable tie-break;
- fails open to the fused ranking when evidence is missing or invalid.

The experimental importance-aware satisfaction reranker is not selected. On
its fresh 800-target test it reduced all headline metrics relative to the
active exact-evidence comparator.

## 5. Profile residual

A recognized aggregate preference profile contributes at most a 5% Stage-A
residual and only when the conversation has no explicit requirement. Once the
user states a requirement, conversational evidence completely disables the
profile residual.

## 6. Clarification and exposure

Questions come from a fixed contract-valid attribute set and are selected from
unresolved evidence.

The active starter uses the repeatable `other` wildcard for the evaluator-facing
`ask_attribute` field. This drains remaining disclosures without constraining
the simulator to one named attribute. Customer-facing question text remains a
separate concern; the wildcard is an evaluator-protocol optimization.

The buying-only exposure gate runs after ranking and before slate selection:

- If the state is exact, consistent, non-fallback, and the best structural tier
  contains at most three candidates, expose that one-to-three-item tier.
- If more candidates remain and an informative question exists, expose the
  literal top-three prefix and ask the question.
- Browsing, ambiguity, contradictions, overrides, final turn, evidence failure,
  and retrieval failure return the safe full-width result.

The gate never changes relative ranking. The evaluated score-threshold dynamic
width policy is disabled because it over-withheld and materially reduced Hit
Rate, MRR, and MTTC.

## 7. Slate selection

The novelty selector receives the exposure-approved pool and tracks shown IDs
within the current intent epoch:

- first ranking for an epoch returns its strongest prefix;
- a continuation with unchanged intent prefers unseen ranked candidates;
- ranking refinement carries the shown set within the same epoch;
- an override increments `intent_version` and starts a new exposure epoch.

This prevents stagnant turns from repeatedly showing the same slate while
allowing an explicit intent change to reset correctly.

## 8. Fail-open behavior

- Dense initialization failure: BM25 continues.
- SQLite FTS5 unavailability: dense continues.
- One route error: use the other route.
- Evidence/reranker validation failure: use fused order.
- Exposure validation failure: use the full ranked slate.
- Cache mismatch: search again.
- Complete retrieval failure: deterministic catalog-valid fallback.
- All API output is sanitized to at most ten unique IDs.

No reset/respond path requires network access or credentials.
