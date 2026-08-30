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
- `embedding-package-arctic-768-v1.json` records the frozen-policy,
  aggregate-only comparison of BGE-small 384 and Arctic M v1.5 768 across two
  generator-separated suites and the released public set. It contains no
  session, scenario-cell, product, query, target, or rank-level outcomes. The
  experiment failed the Pareto promotion gates; BGE-small 384 remains active.
- `phase15-pareto-candidate-static-validation.json` records the bounded,
  no-catalog verification of the opt-in belief-state candidate, conditional
  dense route, disjoint robustness-suite contract, and evidence-recomputed
  promotion authority. It makes no score or latency-improvement claim and
  explicitly records the remaining lock and sealed-evaluation blockers.
- `phase0-active-phase13-baseline-20260830.json` is the immutable starting
  measurement for the expected-utility integration work. It freezes the active
  Phase 13 policy, official metrics, response latency, process-level memory,
  repository state, and focused contract-test result before policy activation.
- `phase1-active-agent-failure-attribution-20260830.json` is an evaluator-only
  ten-turn shadow replay of the same active policy. Runtime wrappers capture
  label-free stage orders first; labels are joined afterward and discarded once
  the ten requested failure classes, rank histograms, and exact additive score
  loss have been aggregated. It persists no dialogue rows, messages, queries,
  profiles, sample identifiers, product identifiers, or candidate lists.
- `phase2-lexicographic-exact-evidence-replayplan-ab-20260830.json` is the
  accepted Phase 2 A/B. It records the isolated lexicographic evidence layer,
  target-disjoint folds, scenario aggregates, route/cache health, latency,
  memory, and every promotion gate; earlier Phase 2 files preserve rejected
  latency and implementation variants.
- `phase3-protocol-world-model-20260830.json` records the inactive, structural
  acceptance of the typed reply-partition world model. It covers all ten legal
  actions, exact/ambiguous/free-form/recovery modes, explicit residual mass,
  normalized boundary worlds, unseen-reply recovery, synthetic contract tests,
  and independent counterexample review. The active Phase 2 runtime metrics are
  referenced unchanged because Phase 3 deliberately does not activate policy.
