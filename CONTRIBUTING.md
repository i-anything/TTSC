# Development workflow

The `main` branch represents the last confirmed submission candidate. New
ideas should be isolated, falsifiable, and easy to remove.

## Naming

Use a semantic hypothesis name such as:

```text
experiment/budget-aware-evidence
experiment/override-parser-coverage
experiment/lexical-recall-repair
```

Do not create sequential `phase20`, `phase21`, and similar directories,
contracts, scripts, or policy names.

## Experiment contract

Before evaluating a ranking change, record:

1. the exact hypothesis and expected failure mode;
2. the frozen active comparator;
3. the datasets and target exclusions;
4. the metrics and conjunctive promotion gates;
5. latency, memory, determinism, offline, and fail-open requirements;
6. the one-run decision rule.

Keep temporary per-run artifacts outside the repository. Add only a concise
aggregate conclusion to [docs/EVALUATION.md](docs/EVALUATION.md) when the
result changes an architectural decision.

## Promotion rules

- Do not tune from individual public misses.
- Prefer target-disjoint cases and language variants.
- Run candidates against the same immutable backend and arm order.
- Require exact replay and zero invalid responses.
- Reject any change that violates a predeclared safety gate.
- A rejected candidate must not remain on the active import path.
- Promote by changing the single configuration in `starter/agent.py`.

## Required checks

```bash
.venv-runtime/bin/python -m unittest discover -s tests
.venv-runtime/bin/python -m evaluator.local_evaluator
```

Also verify that reset/respond need no network and that model, index, and
catalog checksums still agree.
