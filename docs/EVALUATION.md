# Submission validation

Validated on 9 September 2026 with Python 3.13.15 on macOS ARM64, using the
bundled CPU model and frozen 50,000-product catalog. The organizer's evaluator
and public data are unchanged.

## Official public evaluator

The organizer provides **200 public sessions**. The current submission was run
on that unmodified set using the official scoring formula.

| Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 1.000000 | 1.000000 | 1.9750 | 0.980500 |

The submission finds all 200 targets at rank one. These are public development
results, not private-final scores.

## Additional synthetic validation

These sessions were generated locally from other catalog products. They are
not additional organizer-provided sessions or the organizer's private set.
The release was checked on two frozen 1,200-target regression suites created
before this change. Targets were excluded from the public set and known earlier
development and validation selections. Scenario proportions match the published
protocol; profiles are neutral. One set samples uniformly over eligible catalog
products. The other samples without replacement using catalog review counts as
a proxy for purchase frequency. The sets have no targets in common; neither
reproduces the organizer's private purchase history.

| Sampling | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform catalog | 1,200 | 0.995000 | 0.982514 | 2.602500 | 0.960204 |
| Purchase proxy | 1,200 | 1.000000 | 0.992681 | 2.180000 | 0.974204 |

Synthetic distributions do not establish private-final performance. The rank
prior is a heuristic, and results depend on the target distribution and catalog
facts.

## Runtime and release checks

- 398 unit tests passed, including serialized-reply ambiguity, question
  selection through the service, override protection, and deadline coverage.
- The offline submission check verified catalog/index checksums, loaded all
  intended backends, exercised dense inference, and confirmed reset replay.
- Guarded validation recorded zero invalid responses, exceptions, runtime
  evaluation-label reads, network attempts, or subprocess attempts.
- Three bounded paraphrase families on the same 200 public targets reproduced
  the canonical responses exactly, including questions and recommendation order.
- The official public run used zero prompt/completion tokens and no model APIs.
  Guarded local measurement: about 6.2 seconds initialization, 109 ms respond
  p95, and 622 MiB process peak RSS. RSS includes evaluator data and retained sessions;
  it is not isolated agent memory. Final-host timings may differ.

Reproduce the submission checks and official score with:

```bash
python -m scripts.check_submission
python -m unittest discover -s tests
python -m evaluator.local_evaluator
```

Raw synthetic sessions and development records are not part of the submission.
