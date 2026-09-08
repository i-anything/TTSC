# Architecture

The submission configuration is selected in `starter/agent.py`. The root
`agent.py` re-exports it. All inference runs locally on CPU.

```mermaid
flowchart TD
    U[Latest user message] --> L[Bounded dialogue envelope parser]
    L --> S[Immutable intent state]
    L --> T[Complete visible transcript]
    S --> C[Exact dependency cache]
    C --> R[Smart BM25 and dense retrieval]
    R --> E[Confirmed / unknown / contradicted evidence]
    T --> P[Full catalog transcript replay]
    P --> W[Question and slate-width planner]
    E --> W
    W --> N[Intent-aware novelty]
    N --> A[Validated API response]
```

## Language and state

`language.py` normalizes only explicit, bounded speech acts. It retains product
payloads, punctuation, and order; contextual replies require an outstanding
question. Attribute declines and exhaustion of additional preferences remain
distinct. Both intent reduction and transcript parsing consume the same
interpretation, preventing disagreement between retrieval and planning.

`intent.py` maintains category, requirements with provenance and importance,
no-preference attributes, asked attributes, and the intent version. Explicit
replacement retires superseded evidence and advances the version. Unsupported
prose remains available to ordinary retrieval.

## Retrieval and evidence

`retrieval.py` combines SQLite FTS5 BM25 with BGE-small INT8 ONNX query
embeddings and four memory-mapped float32 catalog shards. The index, tokenizer,
model, and catalog are checksum-bound. Smart routing skips dense inference
only when the catalog establishes complete, narrow typed support and the
lexical route covers it. Missing or inconsistent support restores hybrid
retrieval.

`exact_evidence.py` ranks confirmed evidence above unknown evidence and
contradictions. Profile themes are bounded secondary evidence; they cannot
overrule explicit constraints. Cache reuse requires equality of every
ranking-relevant dependency, including intent and backend identity.

## Dialogue planning

`protocol_index.py` reconstructs support by replaying the complete visible
transcript against catalog-derived cards. A recognizable envelope is only an
input to this check. Unsupported turns cannot leave a partial transcript that
later resumes exact inference. Exhausted or inconsistent support returns to
ordinary retrieval.

Eligible continuation refutation removes only previously displayed products
when the transcript permits that inference. Tentative intent before a
scheduled override does not trigger premature refutation.

`exposure.py` selects questions and slate widths using remaining disclosures,
remaining turns, and the published scoring utility. `slates.py` preserves
novelty within an intent epoch. `service.py` validates the response and reports
the executed planner outcome, normalization flag, question, and presented
width through an optional diagnostic hook.

## Runtime and submission

The agent has no network client, hosted model, credentials, runtime evaluation
labels, or cross-user target memory. Reset creates session-local state.
Failures retain deterministic lexical fallback. The release check verifies
all bundled assets, requires every intended backend to initialize, exercises
actual dense inference, and checks repeated reset/respond behavior under an
offline audit guard.
