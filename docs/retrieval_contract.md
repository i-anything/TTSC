# Retrieval Contract

This document freezes the representation boundary between catalog
preprocessing, conversational intent state, query encoding, and dense search.
Changing a frozen product-side field, ordering rule, model asset, tokenizer,
pooling rule, or normalization rule invalidates the existing catalog vectors
and requires a complete index rebuild.

## Contract versions

- Product text: `product-text-v2` (implemented and indexed)
- Embedding artifacts: schema `2` (implemented and indexed)
- Intent state: `intent-state-v2` (implemented)
- Intent reducer: `robust-intent-reducer-v1` (implemented; reversible comparator)
- Query text: `query-text-v1` (implemented)
- Fusion policy: `completeness-adaptive-rrf-v1` (implemented)
- Reranking policy: `deterministic-stage-a-v1` (implemented)
- Slate policy: `stagnation-aware-slate-v1` (implemented)

## Embedding space

- Model: `BAAI/bge-small-en-v1.5`
- Revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- Derived INT8 ONNX SHA-256:
  `f8b2217838ea27564f870f96e377cb6e5ca0fa37dec9599cf305d5de011d6b7f`
- Tokenizer SHA-256:
  `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66`
- Dimension: 384
- Maximum sequence length: 512 tokens including special tokens
- Pooling: CLS (`last_hidden_state[:, 0, :]`)
- Output: float32, L2-normalized
- Product prefix: empty string
- Query prefix:
  `Represent this sentence for searching relevant passages: `
- Similarity: float32 dot product, equivalent to cosine similarity for the
  normalized vectors

Products and runtime queries must use this exact model file, tokenizer,
pooling rule, and normalization rule. Matching only the model name or vector
dimension is insufficient.

## Product text: `product-text-v2`

The canonical product document is newline-separated labeled text. Optional
sections with no value are omitted. The exact section order is:

```text
Title: <title>
Category: <compact category>
Search Clues: <clue> | <clue> | <clue> | <clue>
Category Path: <category> > <category>
Brand: <brand>
Attributes: <name>: <value> | <name>: <value>
Features: <feature> | <feature>
Description: <description>
Details: <name>: <value> | <name>: <value>
Price: <normalized price>
```

`Search Clues` contains at most four unique cleaned values, selected in this
priority order: detected material, detected color, features in source order,
remaining details in source order, then price if a slot remains.

Operational fields such as ASIN duplication, availability date, model number,
package dimensions, product dimensions, and bestseller rank are excluded.
Brand/manufacturer and promoted attributes are not repeated in `Details`.
Canonical product text is transient and is not persisted beside the vectors.

The encoder uses deterministic right truncation at 512 tokens. In the frozen
catalog, 4,999 documents are longer than this limit. A local evaluator-focused
audit observed that the title/category/search-clue prefix was always within the
limit and found no generated evaluator constraint lost by truncation. This is
development evidence for the frozen public catalog, not a general guarantee
for arbitrary catalogs.

The authoritative implementation is `preprocessing/catalog.py`; the completed
index manifest records `product-text-v2` and its canonical-text SHA-256.

## Intent state: `intent-state-v2`

Each session owns an immutable active-intent state with these fields:

```text
category
requirements[] = {value, source, turn, attribute?}
excluded
no_preference
asked_attributes
last_asked_attribute
intent_version
last_turn
```

Requirement provenance is one of `initial_explicit`, `initial_tentative`,
`answer`, `override`, or `free_text`. Attribute is either inferred from a
high-confidence constraint or copied from the exact clarification field that
elicited an answer; otherwise it is absent. Unknown values are never
represented as synthetic `unknown` or `none` tokens.

Phase 1 state update precedence is:

1. Latest explicit user statement
2. Earlier explicit user statement

Anonymous profile blending is disabled until a public-set ablation shows that
it helps without overpowering explicit session intent.

An explicit override removes every initial-source requirement regardless of
its inferred attribute. This is deliberately provenance-based: evaluator
overrides can replace a closure preference with a material preference, so a
same-slot replacement rule is incorrect. Category and independently disclosed
question answers survive the override. A boundary answer such as "no
preference" clears active requirements for that named attribute, records it in
`no_preference`, and prevents the agent from repeatedly asking for it.
Unclassified but useful language remains as a free-text requirement instead of
being discarded.

Every explicit override increments `intent_version`. The version is not query
text; it is a session-local control signal that guarantees the recommendation
slate resets after an override even when the replacement happens to render the
same words.

The Phase 6 production reducer recognizes conservative natural variants of the
simulator's buying, browsing, tentative-preference, clarification-answer,
override, no-preference, and question-request families. Its `canonical` policy
retains the exact earlier grammar for controlled comparison. Bare answers are
typed only when positive attribute evidence agrees with the last requested
field. Strong replacement cues remove a same-slot requirement only when the
target is supported; ambiguous multi-slot language is retained as a search clue
instead of destructively guessing. General negation and hard numeric filtering
remain outside this contract; `excluded` is reserved for a later, separately
tested structured-filter phase.

The state is updated from the latest message and then rendered from scratch.
Conversation messages are never concatenated into the retrieval query.

The reducer adds no production randomness or inference dependency. Its frozen
five-replicate robustness experiment transformed 2,607 of 3,060 candidate
messages and produced 3,060/3,060 canonical state, query, and live-service
matches. The exact Phase 5 public evaluator payload was preserved. Full gates,
protocol hashes, and aggregate-only evidence are in `docs/phase6_results.json`.

## Query text: `query-text-v1`

The dense query is role-appropriate but uses the same labels as the product
document where they represent the same concept. Its exact optional section
order is:

```text
Category: <active category>
Search Clues: <active use case> | <feature> | <other> | <free text>
Brand: <active brand>
Attributes: Material: <value> | Color: <value> | Size: <value> | Style: <value>
Price: <active budget expression>
```

Only known, currently active positive preferences are rendered. Unknown fields
and old overridden values are omitted. The renderer does not invent `Title`,
`Description`, `Features`, or `Details` sections for a shopper request.

Excluded values are not placed in the positive dense query because embedding
models can associate strongly with a negated word. Exact structured filtering
and candidate reranking are not part of Phase 1; until that layer has an
ablation and missing-metadata policy, sizes, brands, and budgets remain positive
retrieval evidence rather than hard filters.

The BGE query prefix is applied by the encoder after rendering and is not
stored inside the intent state.

## Per-turn retrieval

On every turn:

1. Parse the latest message into an intent-state delta.
2. Apply additions, source-aware removals, and boundary answers.
3. Render `query-text-v1` from the complete active state.
4. Compute the bounded intent-completeness proxy and its route weights.
5. Compute the exact ranking-dependency digest and choose `SEARCH`, exact
   `REUSE`, or empty-result `SKIP`.
6. On `SEARCH`, encode the `query-text-v1` view as one normalized
   384-dimensional query vector.
7. Search the complete catalog index again, including after changed evidence
   and intent overrides.
8. Retrieve BM25 Top 100 and dense Top 100 candidates.
9. Fuse the two ranked lists using weighted reciprocal rank fusion with
   `k = 60`, breaking ties by catalog row order.
10. Read transient searchable text for the complete fused union from the
    existing in-memory FTS table using verified row IDs.
11. Apply the deterministic Stage-A score to obtain the complete fused-union
    order.
12. On `REUSE`, restore only a previously complete successful pre-slate ranking;
    candidate text, queries, documents, vectors, scores, and responses are never
    cached.
13. Compare the label-free ranking signature with the session slate state. On
    unchanged evidence, select unseen candidates first; otherwise reset to the
    strongest ranked window.
14. Return recommendations even when also asking a clarification question.

The one-vector rule scopes `query-text-v1`; it is not an evaluator constraint.
A later query-side contract may add a broad view or requirement probe against
the same frozen product vectors without rebuilding `product-text-v2`.

`HybridRetriever.search_with_trace()` exposes immutable, bounded BM25, dense,
and fused rank lists from this same single execution for offline ablations.
The normal `search()` API and Agent response remain unchanged. Traces contain
no target labels, scenario labels, profiles, or raw query text.

The model, memory-mapped index, and in-memory FTS table are initialized once per
agent instance, not once per session or turn. Startup verifies the model,
tokenizer, ID array, all four vector-shard checksums, and the runtime catalog's
SHA-256 against the dense manifest. A mismatch disables dense retrieval instead
of searching an incompatible space. `reset()` replaces only the named session
state. If FTS5 is unavailable, the dense route remains usable. If either route
fails, the other remains available; if neither returns a candidate, the
retriever returns a deterministic catalog-order fallback.

## Fusion policy: `completeness-adaptive-rrf-v1`

The policy uses only active requirement provenance. Strong sources
(`initial_explicit`, `answer`, and `override`) contribute `1.0`; weak sources
(`initial_tentative` and `free_text`) contribute `0.5`:

```text
C_t = clip((strong + 0.5 * weak) / 3, 0, 1)
alpha_bm25 = 0.40 + 0.20 * C_t
alpha_dense = 1 - alpha_bm25
```

For candidate `d`, the fused score is:

```text
alpha_bm25 / (60 + rank_bm25(d))
+ alpha_dense / (60 + rank_dense(d))
```

Both routes therefore retain at least 40% influence. Category-only intent
slightly favors semantic retrieval; accumulated explicit evidence gradually
favors exact lexical matching. If only one route survives, multiplying every
score by its positive route weight preserves that route's ordering.

The weights were predeclared before the Phase 3 A/B and were not grid-searched
on the public labels. Compared with equal RRF, the adopted policy changes
Hit Rate@10 from `0.770` to `0.775`, MRR from `0.406222` to `0.433784`, and
TechnicalScore from `0.632567` to `0.643135`. It adds no embeddings, dense
scans, model memory, API calls, or product artifacts. Full results and the
decision gate are recorded in `docs/phase3_results.json`.

## Reranking policy: `deterministic-stage-a-v1`

Stage A reorders only the current BM25/dense fused union, which is bounded at
200 unique products. It cannot introduce an item that neither retrieval route
returned. Candidate text is read on demand from the existing SQLite FTS table
in fused order; it is not reread from JSONL, cached across turns, or persisted
as a second catalog artifact. Phase 7 may retain only the resulting normalized
product-ID order after complete successful retrieval and Stage-A ranking.

The active category is a weak clause with weight `0.5`. Strong requirement
sources (`initial_explicit`, `answer`, and `override`) have weight `1.0`; weak
sources (`initial_tentative` and `free_text`) have weight `0.5`. After
case-folding, accent normalization, label removal, and stop-word removal, each
clause-product match is:

```text
1.0                         exact significant-token phrase
0.8                         every significant token is present
(matched / required)^2      otherwise
```

A clause that matches no candidate is excluded from the satisfaction
denominator. Typed budget clauses are also excluded: price is deliberately not
stored in the transient FTS document, so matching a number against unrelated
catalog prose would be semantically unsafe. Budget language remains available
to BM25/dense retrieval until a separately tested typed-price representation
exists.

For candidate `d`, let `RRF(d)` be the Phase 3 weighted RRF score and let
`S(d)` be weighted requirement satisfaction over observable clauses:

```text
beta_t = 0.20 + 0.25 * C_t
RRF_normalized(d) = RRF(d) / max_candidate RRF
StageA(d) = (1 - beta_t) * RRF_normalized(d) + beta_t * S(d)
```

Exact score ties retain fused order. The scorer is bounded to 200 candidates,
32,768 characters per candidate, 32 clauses, 1,024 characters per clause, and
64 significant tokens per clause. Any candidate-access or reranking exception
returns the exact Phase 3 recommendation order; missing FTS is an expected
unavailable skip rather than a reranker fault.

The predeclared sequential A/B moved Hit Rate@10 from `0.775` to `0.885`, MRR
from `0.433784` to `0.514109`, MTTC from `4.725` to `3.695`, and
TechnicalScore from `0.643135` to `0.742833`. It made no new embedding, model,
or API calls and stored no second catalog copy. All 716 attempted reranks
succeeded. The worst of the candidate and deterministic replay warm-p95 ratios
was `1.226709`, inside the frozen `1.25` gate. Full evidence is recorded in
`docs/phase4_results.json`.

## Slate policy: `stagnation-aware-slate-v1`

The active Phase 5 policy treats an unchanged continuation as implicit negative
feedback. It applies only after a complete Stage-A ranking succeeds. The
signature contains the intent version, rendered dense and lexical queries,
route weights, active requirement values and provenance, exclusions, ranking
policy, complete ranked pool, and requested result count. It deliberately
excludes turn number and clarification bookkeeping that cannot affect scoring.

```text
changed signature   -> clear shown IDs; return ranked positions 1..K
unchanged signature -> return highest-ranked unseen IDs; backfill only on exhaustion
intent override     -> increment intent version; force a reset
slate failure       -> return the exact Phase 4 Top-K; retain prior cursor
```

Each session retains one current signature and a shown-ID tuple, both bounded
by the 200-candidate fused union. The policy stores no product text or vectors,
makes no additional retrieval, embedding, model, or API calls, and is isolated
behind the reversible `repeat_top` policy used by the Phase 4 comparator.

The predeclared sequential A/B moved Hit Rate@10 from `0.885` to `0.990`, MRR
from `0.514109` to `0.522230`, MTTC from `3.695` to `3.070`, and TechnicalScore
from `0.742833` to `0.810269`. It rescued 21 sessions and lost no incumbent
hits. All 612 candidate slate selections and reranks succeeded; deterministic
replay was exact. The worst candidate/replay warm-p95 ratio was `1.006148`,
inside the frozen `1.15` gate. Full evidence is recorded in
`docs/phase5_results.json`.

## Orchestration policy: `exact-ranking-reuse-v1`

Phase 7 separates ranking computation from slate presentation. The ranking
digest contains the category, ordered requirement value/source/attribute,
exclusions, both rendered queries, exact route weights, ranking policy, and a
versioned immutable-backend contract. It deliberately excludes turn and
question bookkeeping, requirement turn, intent version, and positive Top-K:
those do not change the complete ranking. Intent version, Top-K, and the ranked
pool remain in the slate signature, so overrides and window-size changes still
reset the shown-product cursor.

```text
no requested results                  -> SKIP
exact dependencies + snapshot match  -> REUSE complete ranked IDs
anything else                         -> SEARCH + Stage-A
```

Reuse requires the exact built-in `immutable-complete-fused-ranking-v1`
capability and a payload-free snapshot token compared by identity. A similarly
named property or arbitrary custom token cannot opt in. The LRU retains at most
256 hashed session keys, one ranking per session, 200 IDs per ranking, and 64
ASCII characters per ID. It stores no raw session identifier, messages,
queries, requirements, profiles, documents, vectors, scores, questions,
responses, or slate state. Fallbacks, route faults, missing documents, reranker
faults/skips, empty rankings, and malformed cache data are never committed.

The frozen candidate matched the Phase 6 evaluator payload and all private
response/intent/slate traces exactly. It avoided 141 of 612 retrievals and
reranks, reduced wall time by `25.8%`, reduced warm p95 by `5.2%`, and retained
`2,066,907` cache bytes at the end of the run. Full evidence is recorded in
`docs/phase7_results.json`.

## Clarification policy

The active Phase 2 policy asks unresolved fields in this order:

```text
feature -> material -> color -> other -> style -> size -> use_case -> budget
```

Resolved values and explicit no-preference fields are skipped. If an intent
override arrives instead of answering the preceding question, that interrupted
field is eligible to be asked again. `other` is asked at most once after it is
answered or declined. No question is emitted on turn 10.

This order was selected by a paired public-set A/B against the frozen Phase 1
order. It preserved Hit Rate@10 at `0.770`, reduced MTTC from `5.35` to `4.715`,
and increased the recommended technical score from `0.620867` to `0.632567`.
MRR decreased slightly from `0.409556` to `0.406222`; this is therefore a
measured convergence improvement, not a ranking improvement.

## Completed index identity

- Build ID:
  `fdfcd830321690d96cd87754db62161b5264485803a08ed1b30f4a0c33c227c8`
- Logical embedding SHA-256:
  `beaabadaa1f13cf0177f7ca02b6aa9a869392c2f7ed4fa8e9b9e30c6467d0ebb`
- Manifest SHA-256:
  `c9b7291004d6ef78473b24886899ea51f427fc2e179c8216c8e8b65f6cf929b2`
- Shape: `[50000, 384]`
- Storage: four contiguous float32 `.npy` shards plus row-aligned `S10`
  product identifiers

The finalized manifest and `READY` marker are under
`assets/search-index-bge-small-en-v1.5-v2`.
