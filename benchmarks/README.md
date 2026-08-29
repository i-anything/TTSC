# Benchmark records

This directory contains compact, immutable evaluation summaries. Each JSON file is
one aggregate-only record conforming to [`schema.json`](schema.json).

## Privacy boundary

Records may contain only the fixed aggregate fields in the schema. Never add raw
user messages, profiles, sample identifiers, target products, ground truth,
per-session traces, or per-sample scenario labels. `scenario_metrics` contains
counts and aggregate scores only. The strict Python validator rejects every
unknown field and limits all strings to compact identifiers or a source path.

## Append-only workflow

Prepare a new JSON object with a unique lowercase `record_id`, then run:

```bash
python -m scripts.record_benchmark candidate.json
```

The recorder validates the object, serializes UTF-8 JSON with sorted keys and a
trailing newline, and publishes `<record_id>.json` atomically. Publication uses a
same-directory temporary file and a no-clobber hard link, so an existing record is
never replaced, including when two writers race. A published record must not be
edited, deleted, or reused; publish a new `record_id` for every new result.

[`schema.json`](schema.json) documents version 1. Cross-field requirements, such
as scenario sample counts summing to `sample_count`, are enforced by
[`scripts/record_benchmark.py`](../scripts/record_benchmark.py), which is the
authoritative validator.

## Recorded milestones

The three initial records copy only values present in their named source files:

- `baseline.json` copies the system, sample count, and aggregate metrics from
  `docs/baseline_results.json`. Its source field `technical_score` is represented
  by the schema's canonical `recommended_technical_score` name. The source has no
  scenario aggregates, so none were added.
- `phase1.json` and `phase2.json` copy `agent_version`,
  `evaluation.public_samples`, `metrics`, and aggregate `scenario_metrics` from
  their respective source documents.
- `phase3.json` records the adopted completeness-adaptive RRF result from
  `docs/phase3_results.json` after a sequential equal-versus-adaptive A/B run.
- `phase4.json` records the adopted deterministic Stage-A reranker from
  `docs/phase4_results.json` after a sequential shared-backend A/B, latency
  gate, deterministic replay, and independent entry-point verification.
- `phase5.json` records the adopted stagnation-aware slate policy from
  `docs/phase5_results.json` after an exact Phase 4 comparator check, paired hit
  safety audit, latency gate, deterministic replay, and independent entry-point
  verification.
- `phase6.json` records the adopted robust intent reducer from
  `docs/phase6_results.json`. The canonical official metrics remain exactly
  Phase 5; promotion is supported by five frozen message-perturbation replicates,
  complete state/query/service equivalence, deterministic replay, and an exact
  aggregate-only privacy schema.
- `phase7.json` records the adopted exact-value orchestration policy from
  `docs/phase7_results.json`. Official metrics remain exactly Phase 6; promotion
  requires complete evaluator/response/state/slate equivalence, deterministic
  replay, independent entry-point verification, bounded cache memory, zero
  faults, and a conservative candidate-first latency gate.
- Phase 8 has no official benchmark record because its locked candidate failed
  quality and paired-safety promotion gates. The aggregate rejection and exact
  rollback are recorded in `docs/phase8_results.json`; Phase 7 remains active.

Dates, commands, runtime details, hashes, comparisons, interpretations, and other
source fields are intentionally omitted. No missing value was inferred.
