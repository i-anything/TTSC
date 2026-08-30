# Evaluation evidence

## Reproducible public result

The active `starter.agent.Agent` was run with the unmodified organizer
evaluator on 31 August 2026:

```bash
.venv-runtime/bin/python -m evaluator.local_evaluator
```

| Scenario | N | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 0.975 | 0.805625 | 2.075 |
| Browsing | 80 | 1.000 | 0.717386 | 2.1375 |
| Intent Override | 30 | 1.000 | 0.709061 | 3.833333 |
| Boundary | 10 | 0.900 | 0.576667 | 3.7 |
| **Overall** | **200** | **0.985** | **0.744397** | **2.445** |

```text
Efficiency      0.8555
TechnicalScore  0.886919
Prompt tokens   0
Completion      0
```

The original organizer BM25 starter reported Hit Rate@10 `0.125`, MRR
`0.068034`, MTTC `9.81`, and TechnicalScore `0.106710`.

## Target-disjoint checks

These are synthetic robustness tests, not organizer-private scores:

| Frozen suite | N | HR@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh target-disjoint suite, mixed official/shifted language | 800 | 0.965 | 0.774515 | 2.53625 | 0.884130 |
| Earlier clean-room hidden-generalization suite | 800 | 0.845 | 0.622242 | 3.6875 | 0.755423 |

Both suites excluded previously used target products and contained the official
40% Buying, 40% Browsing, 15% Intent Override, and 5% Boundary mix. The score
spread is reported deliberately: template and simulator choices materially
affect local results, so neither suite proves the hidden distribution.

## Last rejected experiments

| Candidate | Frozen comparator | Candidate result | Decision |
| --- | ---: | ---: | --- |
| Importance-aware MUST/SHOULD/PREFER reranker | 0.870921 | 0.852252 | Rejected |
| Dense-term BM25 repair while retaining hybrid fusion | 0.884130 | 0.884130 | Inactive; no measured gain |
| Score-threshold dynamic exposure | 0.884130 | 0.767233 | Rejected |
| Repair plus dynamic exposure | 0.884130 | 0.767233 | Rejected |

The active code therefore keeps exact-evidence reranking, ordinary BM25+dense
fusion, and the conservative buying-only prefix gate.

## Runtime disclosure

The public confirmation diagnostic measured:

| Quantity | Measurement |
| --- | ---: |
| Respond calls | 486 |
| p50 response latency | 23.50 ms |
| p95 response latency | 44.43 ms |
| Maximum response latency | 92.96 ms |
| Evaluation wall time | 11.45 s |
| Process peak RSS | approximately 632 MB |
| Runtime network calls | 0 |
| External model/API calls | 0 |
| Reported tokens | 0 |
| Estimated API cost | USD 0 |

Measurements are machine- and cache-dependent. The model and index occupy
about 107 MiB on disk; the 50,000-row catalog is approximately 58 MiB and is
downloaded separately from the organizer release.

## Reproducibility boundaries

- Only aggregate results are reported here.
- The official public set is development data and influenced system design.
- No claim of 100% non-overfitting or private-set equivalence is made.
- The frozen target-disjoint tests reduce product overlap but still use local
  deterministic simulators.
- Rejected candidates were not activated after observing their outcomes.
- The active entry point is `starter.agent.Agent`; the root `agent.Agent`
  is an exact re-export.
