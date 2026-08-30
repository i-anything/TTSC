# Phase 14 research plan

Status: frozen before fresh-suite generation, candidate implementation, or
candidate evaluation.

## Protected baseline

The protected submission is the promoted Phase 13 agent: robust intent state,
fixed conservative clarification, completeness-adaptive BM25/dense fusion,
Stage-A ranking, bounded profile residual, exact ranking reuse, and
intent-epoch continuation novelty. Its adapter SHA-256 is
`990843888a835ae4101a9c5287d8fa04947029c1a3db51cb157a306e86ba95e2`.
The complete baseline record is `docs/phase14_baseline_lock.json`.

The official public metrics are HR@10 `0.990000`, MRR `0.556748`, MTTC
`2.910000`, and TechnicalScore `0.823824`. These values are a protected
regression baseline, not an estimate of the 800 private sessions.

## One hypothesis

The current lexical query blends category and every active requirement. A
rare decisive requirement can therefore contribute too little to the main
BM25 Top-100. Phase 14 tests exactly one catalog-only mechanism: generate at
most two independent BM25 probes for strong, non-budget requirements selected
by frozen-catalog document frequency, then use only otherwise-empty capacity
inside the existing 200-candidate union to add unseen probe results.

Selection uses SQLite's supported `fts5vocab` row view. A requirement is ranked
by the smallest positive catalog document frequency among its known terms,
which is order-equivalent to its maximum smoothed IDF. OOV terms are ignored,
ties preserve active-state order, identical token signatures are deduplicated,
and no public label, target, session, scenario, result, or coefficient is used.

The complete incumbent BM25/dense union is preserved. Supplemental IDs are
interleaved by probe rank until the union reaches 200, appended after the main
BM25 route, and passed through the unchanged outer RRF and Stage-A pipeline.
There is no new model, embedding, index, dependency, persistent product state,
or external call. Exact cache reuse performs no probe work.

## Hidden-generalization control

The 996- and 1,000-case synthetic sets are content-separated but have already
been used in earlier research. Phase 14 therefore treats them as reused
generator regression suites, not fresh holdouts. Before candidate code exists,
a new 384-case mechanistic suite will be generated from catalog targets absent
from public and both prior generators. It uses three predeclared broad catalog
families, official intent-card extraction, official message templates, no
retrieval results during sampling, and publishes only aggregate counts and
hashes. This remains synthetic and is not a private-score estimator; it is an
additional target-disjoint robustness gate.

## Frozen execution and rollback

The machine-readable contract is
`docs/phase14_experiment_contract.json`. `starter/agent.py` remains unchanged
until all deterministic checks, the fresh mechanistic suite, development,
validation, and one-shot public confirmation pass.

Every suite runs sequentially on one CPU process with all numerical and ONNX
thread counts fixed to one. Variant order is candidate, exact Phase 13
baseline, candidate replay, and an independently constructed candidate.
Evaluation output is aggregate-only. No failed row, target, query, product,
rank, trajectory, or small subgroup may be printed or inspected.

Promotion requires no HR, MRR, or MTTC regression; no baseline-hit to
candidate-miss transition; strict TechnicalScore improvement; the predeclared
paired-bootstrap confidence gates; exact replay/independent construction; zero
faults; complete bounded call accounting; and no more than 1.05x warm p95 or
wall time. Any failure rejects the candidate, leaves Phase 13 active, records
one aggregate diagnostic, and consumes the hypothesis without repair or rerun.
