# Submission validation

Validated on 8 September 2026 with Python 3.13.15 on macOS ARM64, using the
bundled CPU ONNX model, tokenizer, dense index, and frozen 50,000-product
catalog. The official evaluator and public data were unchanged.

| Evaluation | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Organizer public development set | 200 | 1.000000 | 0.996250 | 2.3500 | 0.971875 |
| Separate nonpublic target validation | 800 | 0.993750 | 0.982180 | 2.5288 | 0.960954 |
| Same 800 targets, each of three phrasing families | 800 per family | 0.993750 | 0.982180 | 2.5288 | 0.960954 |

The three phrasing families alter request, answer, override, and decline
envelopes while preserving disclosed product values and simulator behavior.
Each produced byte-identical agent responses to its canonical counterpart,
including recommendation order, question, and turn count. These are repeated
conditions on 800 distinct targets, not 3,200 independent target samples.

The 800 targets were excluded from the public set, the 400-target development
set for this change, and known earlier internal target selections. They were
sampled uniformly with neutral profiles and the official 40% buying, 40%
browsing, 15% override, and 5% boundary proportions. The runtime was frozen
before these validation runs. Historical exclusion is limited to known
selections; this synthetic distribution is not the organizer's private set.

## Runtime measurements

| Measurement | Public 200 | Nonpublic canonical 800 |
| --- | ---: | ---: |
| Initialization | 3.999 s | 4.000 s |
| Respond p95 | 54.190 ms | 59.028 ms |
| Process peak RSS | 587.0 MiB | 609.8 MiB |
| Prompt / completion tokens | 0 / 0 | 0 / 0 |
| External model charges | $0 | $0 |

RSS includes the evaluator's catalog, retained sessions, and recorded
responses; it is not an isolated per-session agent-memory measurement. Timing
comes from sequential local runs and may differ on the final host.

## Release checks

- 374 unit tests passed, including payload preservation, override/refutation
  behavior, unsupported-transcript fallback, and catalog preparation failures.
- Canonical recommendation responses remained unchanged.
- Guarded runs recorded zero exceptions, invalid responses, network/process
  attempts, or runtime evaluation-label reads.
- The offline submission check verified catalog and index checksums, required
  every intended backend, exercised dense inference, and confirmed reset replay.

Run `python -m scripts.check_submission`, `python -m unittest discover -s tests`,
and `python -m evaluator.local_evaluator` from an environment with the pinned
runtime dependencies. Raw validation sessions and development records are not
part of the submission.

These results measure the published dialogue contract and bounded paraphrase
coverage. They do not establish unrestricted language understanding or predict
the organizer's private-final score.
