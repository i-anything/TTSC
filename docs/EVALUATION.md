# Submission validation

Validated on 9 September 2026 with Python 3.13.15 on macOS ARM64, using the
bundled CPU model and frozen 50,000-product catalog. The organizer's evaluator
and public data are unchanged.

## Official public evaluator

The organizer provides **200 public sessions**. The current submission was run
on that unmodified set using the official scoring formula.

| Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 1.000000 | 1.000000 | 2.0850 | 0.978300 |

The submission finds all 200 targets at rank one. These are public development
results, not private-final scores.

## Additional synthetic validation

These sessions were generated locally from other catalog products. They are
not additional organizer-provided sessions or the organizer's private set.
The runtime was frozen before two fresh 1,200-target evaluations. Targets were
excluded from the public set and known earlier development and validation
selections. Scenario proportions match the published protocol; profiles are
neutral. One set samples uniformly over eligible catalog products. The other
samples without replacement using catalog review counts as a proxy for purchase
frequency. The sets have no targets in common; neither reproduces the organizer's
private purchase history.

| Sampling | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform catalog | 1,200 | 0.995000 | 0.984239 | 2.655833 | 0.959655 |
| Purchase proxy | 1,200 | 1.000000 | 0.996688 | 2.267500 | 0.973656 |

Synthetic distributions do not establish private-final performance. The rank
prior is a heuristic, and results depend on the target distribution and catalog
facts.

## Runtime and release checks

- 383 unit tests passed, including serialized-reply ambiguity, question
  selection through the service, override protection, and deadline coverage.
- The offline submission check verified catalog/index checksums, loaded all
  intended backends, exercised dense inference, and confirmed reset replay.
- Guarded validation recorded zero invalid responses, exceptions, runtime
  evaluation-label reads, network attempts, or subprocess attempts.
- Three bounded paraphrase families on the same 200 public targets reproduced
  the canonical responses exactly, including questions and recommendation order.
- The official public run used zero prompt/completion tokens and no model APIs.
  Local measurement: about 3.9 seconds initialization, 69 ms respond p95, and
  626 MiB process peak RSS. RSS includes evaluator data and retained sessions;
  it is not isolated agent memory. Final-host timings may differ.

Reproduce the submission checks and official score with:

```bash
python -m scripts.check_submission
python -m unittest discover -s tests
python -m evaluator.local_evaluator
```

Raw synthetic sessions and development records are not part of the submission.
