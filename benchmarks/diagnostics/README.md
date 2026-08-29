# Diagnostic benchmark reports

This subdirectory contains aggregate, append-only stage diagnostics used to decide
which system component to improve next. These reports are intentionally separate
from the official-score records in the parent directory and do not conform to
`benchmarks/schema.json`.

`scripts/run_retrieval_audit.py` records route ranks only after retrieval has
finished, aggregates them, and publishes with no-overwrite semantics. Reports must
not contain queries, messages, profiles, sample IDs, target product IDs, route
product lists, or per-session rows.

Current reports:

- `phase2-retrieval-loss-waterfall.json` captured a dependency-drift check under
  NumPy 2.5.2, ONNX Runtime 1.29.0, and Tokenizers 0.23.1. It is retained as an
  immutable diagnostic of the reproducibility failure.
- `phase2-pinned-retrieval-loss-waterfall.json` reran the same audit after
  restoring `requirements-runtime.txt`; its official metrics reproduce the
  `phase2.json` benchmark exactly.
- `phase3-retrieval-loss-waterfall.json` records the adopted
  completeness-adaptive RRF policy using the hardened audit schema, including
  rank histograms and explicitly right-censored session summaries.
- `phase4-stage-a-reranker-ab.json` records aggregate quality, route health,
  reranker health, and label-free latency evidence for the adopted Stage-A
  policy. It also preserves the two rejected implementation-latency attempts;
  neither changed the frozen scoring equation or evaluator recommendations.
- `phase5-stagnation-aware-slate-ab.json` records the aggregate Phase 4 versus
  Phase 5 comparison, paired rescue/regression counts, route/reranker/slate
  health, deterministic replay, and label-free response latency.
- `phase6-message-robustness.json` records aggregate five-replicate message
  perturbation results for the canonical and robust intent reducers. It contains
  no messages, profiles, identifiers, session rows, or per-replicate rows.
