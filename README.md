# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. Install the pinned CPU runtime dependencies
before running the hybrid agent.

```bash
python3 -m evaluator.local_evaluator
```

The editable agent now delegates to the source-aware conversational-search
package. Do not edit the evaluator or public labels when reporting a local
score.
The command writes per-session results and aggregate metrics to `results.json`.

The original weak BM25 starter scored Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

The Phase 1 source-aware hybrid agent scores Hit Rate@10 `0.770`, MRR
`0.409556`, and MTTC `5.35` on the same 200 public sessions with one ONNX CPU
thread and no external API calls. This is a public development result, not an
estimate of the private judging score. See `docs/phase1_results.json`.

The Phase 2 clarification policy keeps Hit Rate@10 at `0.770`, moves MTTC to
`4.715`, and raises the recommended technical score from `0.620867` to
`0.632567`. MRR changes slightly to `0.406222`. The paired result and policy
tradeoff are recorded in `docs/phase2_results.json`.

Phase 3 adds the first bounded decision-math policy. Requirement provenance
produces an intent-completeness proxy, which moves BM25 and dense RRF weights
only within `0.40`–`0.60`. It uses no new model calls or catalog memory. In the
predeclared sequential A/B test, Hit Rate@10 reaches `0.775`, MRR `0.433784`,
MTTC `4.725`, and TechnicalScore `0.643135`. See
`docs/phase3_results.json` and the append-only records in `benchmarks/`.

Phase 4 adds a deterministic Stage-A reranker over the existing fused union.
It reuses transient text already held by the in-memory FTS index, makes no new
embedding or model calls, and stores no second catalog copy. A bounded
requirement-satisfaction score is blended with max-normalized weighted RRF;
typed budgets are left to retrieval because the transient document omits
price. In the predeclared shared-backend A/B, Hit Rate@10 reaches `0.885`, MRR
`0.514109`, MTTC `3.695`, and TechnicalScore `0.742833`. The deterministic
replay was identical, all 716 reranks succeeded, and the worst measured warm
p95 ratio was `1.227×` against the `1.25×` gate. See
`docs/phase4_results.json` and `benchmarks/phase4.json`.

Phase 5 makes the agent feedback-aware without adding retrieval or model work.
It fingerprints the complete label-free ranking state and remembers which
products were already shown. A new requirement or intent override resets to
the strongest slate; a continuation that adds no scoring evidence advances to
unseen Stage-A candidates. The predeclared Phase 4 comparison reaches Hit
Rate@10 `0.990`, MRR `0.522230`, MTTC `3.070`, and TechnicalScore `0.810269`.
It rescued 21 prior misses without losing an incumbent hit, replayed
identically, and its worst warm-p95 ratio was `1.006×` (about `0.6%`
overhead). See
`docs/phase5_results.json` and `benchmarks/phase5.json`.

Phase 6 hardens the intent reducer against meaning-preserving natural phrasing
without adding production randomness, model calls, or catalog artifacts. A
reversible canonical policy preserves the Phase 5 comparator, while the active
policy recognizes conservative browsing, answer, override, no-preference, and
question-request variants; ambiguous useful language remains free-text evidence.
Across five frozen, seed-controlled message-perturbation replicates, all 3,060
candidate state, rendered-query, and live-service checks matched the canonical
state. Candidate HR@10 remained `0.990` in every replicate versus a perturbed
baseline mean of `0.900`; mean TechnicalScore was `0.810269` versus `0.739836`.
The unperturbed official score remains exactly Phase 5. See
`docs/phase6_results.json`, `benchmarks/phase6.json`, and the aggregate diagnostic
under `benchmarks/diagnostics/`.

Phase 7 adds exact-value dynamic orchestration without changing ranking or
clarification behavior. Before each query, the agent computes a complete
ranking-dependency digest and chooses `SEARCH`, exact `REUSE`, or empty-result
`SKIP`. Reuse is available only for the built-in immutable/full-pool backend
capability; changed evidence, overrides, partial results, faults, custom
backends, and snapshot changes fail closed to a fresh search. The candidate
matched every Phase 6 evaluator response, intent state, and slate state while
avoiding 141 of 612 retrievals and reranks. In the conservative candidate-first
timing order, wall time fell `25.8%` and warm p95 fell `5.2%`; retained cache
memory was about `1.97 MiB`. Official metrics remain exactly HR@10 `0.990`, MRR
`0.522230`, MTTC `3.070`, and TechnicalScore `0.810269`. See
`docs/phase7_results.json`, `benchmarks/phase7.json`, and the aggregate diagnostic
under `benchmarks/diagnostics/`.

Phase 8 tested candidate-grounded clarification without using public labels in
the policy design. A locked selector reordered comparable questions from a
bounded, value-free facet summary of the current Stage-A pool. The one-shot
candidate was deterministic, fault-free, and within latency/memory limits, but
it reduced MRR from `0.522230` to `0.518458`, increased MTTC from `3.070` to
`3.100`, and reduced TechnicalScore from `0.810269` to `0.808537`; an automated
paired check also found one incumbent hit-to-miss regression. It was therefore
rejected without inspecting individual cases or tuning from misses. All Phase 8
runtime code and overhead were removed. See `docs/phase8_results.json`.

Phase 9 established the protected baseline used through Phase 12. It adds a
`5%` bounded profile residual to Stage-A only when the conversation has no
explicit requirements; conversational
evidence therefore always takes precedence. Reset parses the preference tags
into a validated ten-bit generic-theme mask and immediately discards the raw
profile, retaining at most two logical bytes per session. Query construction,
retrieval, clarification, slate selection, and model/API/embedding calls are
unchanged. In the frozen, single sealed public A/B, HR@10 remains `0.990`, MRR
rises from `0.522230` to `0.529558`, MTTC improves from `3.070` to `3.065`, and
TechnicalScore rises from `0.810269` to `0.812567`. There were no incumbent
hit-to-miss regressions, all promotion gates passed, and warm-p95 latency was
`0.988×` the Phase 7 comparator. Before the contract was frozen, development
had aggregate, unlabeled exposure to public-profile values and coarse tag
frequencies; there was no row-level linkage to targets, labels, outcomes, or
ranks, and no post-run tuning or second candidate run. See
`docs/phase9_results.json`, `benchmarks/phase9.json`, and the aggregate
diagnostic under `benchmarks/diagnostics/`.

Phase 10 tested one completeness-gated, one-sided BM25 rescue over the same
fused candidate union. In its single sealed comparison, HR@10 remained `0.990`,
MRR rose from `0.529558` to `0.563391`, MTTC improved from `3.065` to `2.980`,
and TechnicalScore rose from `0.812567` to `0.824417`, with no Phase 9
hit-to-Phase 10-miss regression. It was nevertheless rejected under the
conjunctive frozen gates: the rescue recorded five aggregate validation/scoring
fallbacks, and total counted retrieval/document/Stage-A calls were `1,888`
versus `1,880` for Phase 9. Replay and independent verification were exact and
both latency gates passed. No row or scenario was inspected, and the candidate
was not repaired, tuned, or rerun. `starter.Agent` therefore remained on Phase
9 at the close of Phase 10. See `docs/phase10_results.json`.

Phase 11 tested a bounded, lossless multi-slot intent reducer without changing
retrieval, ranking, profiles, questions, slates, models, or indexes. On the
deduplicated 996-case generator-separated development suite, it improved
HR@10 from `0.991968` to `0.992972`, MRR from `0.534880` to `0.539626`, MTTC
from `3.058233` to `3.033133`, and TechnicalScore from `0.815283` to
`0.817711`, with zero baseline-hit regressions. Promotion was nevertheless
rejected because the precommitted paired-bootstrap 95% lower bound was
`-0.000320927`, below the required non-negative bound. Validation and public
confirmation were not run, no individual cases were inspected, and the starter
remained byte-for-byte Phase 9 at the close of Phase 11. See
`docs/phase11_results.json`.

Phase 12 tested a symmetric, rank-only correction for overlapping BM25 and
dense evidence. On the frozen 996-case development split it preserved HR@10
at `0.991968` and improved MTTC from `3.058233` to `2.992972`, but MRR fell
from `0.534880` to `0.526749`, TechnicalScore fell from `0.815283` to
`0.814149`, and the paired-bootstrap lower 95% bound was `-0.004931153`.
The candidate was rejected immediately; validation and public confirmation
were not run. See `docs/phase12_results.json`.

Phase 13 is the latest fully evaluated policy baseline. It treats
`intent_version` as an explicit
replacement boundary and carries the existing bounded shown-product set across
ordinary same-epoch ranking refinements. This removes repeated exposure without
changing retrieval, scores, questions, dependencies, or state shape; overrides
reset exactly. It passed every frozen gate on development, independent
validation, and public confirmation. On public confirmation, HR@10 remains
`0.990`, MRR rises from `0.529558` to `0.556748`, MTTC improves from `3.065`
to `2.910`, and TechnicalScore rises from `0.812567` to `0.823824`. The paired
bootstrap lower 95% bound is `0.002111905`, with zero baseline-hit regressions,
zero fallbacks, exact replay and independent behavior, no new calls, and a
warm-p95 ratio of `1.016139`. See `docs/phase13_results.json` and
`benchmarks/phase13.json`.

Phase 15 is an opt-in, unpromoted Pareto candidate. It combines exact
disclosure evidence, a bounded candidate belief, and a one-step utility planner
that jointly chooses the next question and recommendation width. Its dual-world
router skips dense query encoding only when the official protocol is recognized,
session evidence is consistent, and a non-empty exact structural route exists;
unsupported or paraphrased language uses the full hybrid route, while an empty
or failed BM25 route receives one dense rescue. Route identity is part of the
exact cache dependency, every failure returns a valid full-width base result,
and a sanitized diagnostic trace exposes decisions without IDs, queries,
profiles, or targets. The protected `starter.Agent` still uses Phase 13 until
all target-disjoint quality, fail-open, latency, memory, determinism, zero-token,
and zero-network gates pass. See `docs/phase15_research_plan.md`.

Before enabling dense search, runtime startup verifies the pinned model and
tokenizer, every vector shard, the ID array, and the exact catalog checksum.
Dense and BM25 routes degrade independently; a build without SQLite FTS5 can
still use dense retrieval and deterministic fallback results.

The Phase 1--13 figures above were measured with the active BGE-small
384-dimensional embedding space. They remain the protected scored baseline.
Any replacement encoder must pass the same target-disjoint quality, latency,
memory, determinism, and fail-open gates before its defaults can change.

## Offline Dense-Retrieval Foundation

The repository includes a CPU-only local embedding setup for the next agent
iteration. It does not require an API key, network service, vector database,
PyTorch, Transformers, SentenceTransformers, CUDA, or MPS.

- Encoder: `BAAI/bge-small-en-v1.5`, pinned to revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- Runtime graph: a verified INT8 ONNX graph, 384 dimensions
- Query instruction: `Represent this sentence for searching relevant passages: `
- Pooling: normalized CLS vector
- Runtime: NumPy, ONNX Runtime CPU, and the Rust `tokenizers` package
- Model bundle: `assets/bge-small-en-v1.5-int8` (about 33.4 MiB)

The derived INT8 graph SHA-256 is
`f8b2217838ea27564f870f96e377cb6e5ca0fa37dec9599cf305d5de011d6b7f`.
The bundle is below GitHub's per-file limit and is loaded directly, with no
runtime download, API call, or billed token usage. The manifest checks the
graph, tokenizer, pinned revision, and vector contract. See
`BGE_MODEL_ATTRIBUTION.md` for MIT attribution.

A frozen 2,196-case package A/B rejected the proposed Arctic 768-dimensional
migration. Arctic won one suite and lost two; its sample-weighted
TechnicalScore delta was `-0.001889`, warm p95 was about 1.51--1.55x BGE, and
a fresh-process label-free turn raised observed peak RSS by about 98.3 MiB.
No post-outcome weights or policies were tuned. Arctic support is retained only
as ignored, local research tooling; it is not an active or submitted runtime
asset. The complete aggregate-only evidence is in
`benchmarks/diagnostics/embedding-package-arctic-768-v1.json`.

### Runtime setup

Python 3.10 through 3.13 is supported. Normal agent execution needs only:

```bash
python3 -m venv .venv-runtime
.venv-runtime/bin/pip install -r requirements-runtime.txt
```

The model is already present in the repository. Normal setup must not run
`scripts.prepare_bge_model`, download model weights, or regenerate catalog
embeddings.

### Catalog embedding artifacts

The frozen 50,000-product catalog is encoded into the finalized 384-dimensional
bundle at `assets/search-index-bge-small-en-v1.5-v2`. Its
build identity is
`fdfcd830321690d96cd87754db62161b5264485803a08ed1b30f4a0c33c227c8`,
and its logical embedding SHA-256 is
`beaabadaa1f13cf0177f7ca02b6aa9a869392c2f7ed4fa8e9b9e30c6467d0ebb`.
The manifest and `READY` marker are authoritative for every file checksum.

The reproducible preprocessing command is:

```bash
python3 -m scripts.preprocess_catalog build \
  --catalog data/catalog.jsonl \
  --model-assets assets/bge-small-en-v1.5-int8 \
  --output assets/search-index-bge-small-en-v1.5-v2
```

The build streams the 50,000 JSONL rows, constructs deterministic search text
transiently, and writes exactly four row-aligned float32 embedding shards
(about 73.3 MiB of vectors). It never stores a second copy of the product text. At
runtime, `starter/dense.py` memory-maps and scores one shard at a time, so all
vectors are never duplicated in RAM. The preprocessing command defaults to one
ONNX compute thread and sequential batches of four products to keep laptop
thermal load conservative. The submitted artifact records its actual build
settings in its manifest.

Model derivation is maintainer-only. If the vendored model ever needs to be
reproduced from its immutable upstream revision, use a clean environment with
`requirements-preprocessing.txt`, then run:

```bash
python3 -m scripts.prepare_bge_model \
  --output assets/bge-small-en-v1.5-int8
```

The command refuses to overwrite an existing model directory.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
docs/phase1_results.json          Phase 1 public benchmark and baseline delta
docs/phase2_results.json          paired clarification-policy A/B decision
docs/phase3_results.json          completeness-adaptive RRF A/B decision
docs/phase4_results.json          deterministic Stage-A reranker A/B decision
docs/phase5_results.json          feedback-aware slate A/B decision
docs/phase6_experiment_contract.json predeclared message-robustness gates
docs/phase6_implementation_lock.json frozen executable perturbation protocol
docs/phase6_results.json          robust intent-reducer decision and evidence
docs/phase7_experiment_contract.json exact-value orchestration gates
docs/phase7_implementation_lock.json frozen Phase 7 source hashes
docs/phase7_results.json          exact-reuse decision and aggregate evidence
docs/phase8_experiment_contract.json predeclared question-value gates
docs/phase8_implementation_lock.json hashes of the evaluated rejected candidate
docs/phase8_results.json          aggregate rejection and rollback record
docs/phase9_experiment_contract.json predeclared profile-residual gates
docs/phase9_implementation_lock.json frozen Phase 9 source and contract hashes
docs/phase9_results.json          aggregate adoption decision and evidence
docs/phase10_experiment_contract.json predeclared BM25-rescue gates
docs/phase10_implementation_lock.json hashes of the sealed Phase 10 candidate
docs/phase10_results.json         aggregate rejection and rollback record
docs/phase11_baseline_lock.json   protected Phase 9 baseline before Phase 11
docs/phase11_dataset_audit.json   aggregate generator separation and dedup audit
docs/phase11_experiment_contract.json frozen multi-slot hypothesis and gates
docs/phase11_implementation_lock.json hashes of the sealed Phase 11 candidate
docs/phase11_results.json         aggregate development rejection and rollback
docs/phase12_baseline_lock.json   protected baseline before route correction
docs/phase12_experiment_contract.json frozen route-redundancy hypothesis
docs/phase12_implementation_lock.json sealed Phase 12 source and oracle hashes
docs/phase12_results.json         aggregate development rejection and rollback
docs/phase13_baseline_lock.json   protected baseline before slate promotion
docs/phase13_experiment_contract.json frozen intent-epoch slate hypothesis
docs/phase13_implementation_lock.json sealed pre-promotion source hashes
docs/phase13_results.json         three-suite aggregate promotion evidence
docs/retrieval_contract.md        product, intent-state, and query contracts
benchmarks/                       append-only aggregate scores and diagnostics
starter/agent.py                  thin competition API adapter
starter/dense.py                  memory-mapped exact dense scorer
conversational_search/intent.py   immutable state, renderers, inactive Phase 11 reducer
conversational_search/profiles.py bounded profile parser and ten-bit prior
conversational_search/retrieval.py BM25+dense retrieval and deterministic RRF
conversational_search/questions.py measured clarification policies
conversational_search/strategy.py bounded decision signals and fusion policies
conversational_search/ranking.py  Stage-A plus inactive Phase 10/12 policies
conversational_search/slates.py   active bounded intent-epoch novelty policy
conversational_search/orchestration.py exact SEARCH/REUSE/SKIP planner
conversational_search/service.py  Agent orchestration and strategy execution
preprocessing/catalog.py          deterministic catalog text normalization
preprocessing/encoder.py          offline CPU ONNX query/document encoder
preprocessing/embeddings.py       atomic four-shard artifact builder
scripts/prepare_arctic_model.py   rejected Arctic research reproduction
scripts/prepare_bge_model.py      active BGE artifact reproduction
scripts/preprocess_catalog.py     catalog scan/build/verification CLI
scripts/run_embedding_package_ablation.py aggregate-only encoder A/B
scripts/run_policy_ablations.py   sequential shared-backend policy comparison
scripts/run_fusion_ablations.py   sequential route-weight A/B comparison
scripts/run_reranking_ablations.py sequential fused-only versus Stage-A A/B
scripts/run_exploration_ablations.py sequential Phase 4 versus Phase 5 A/B
scripts/run_message_robustness.py seeded Phase 5 versus Phase 6 robustness test
scripts/run_orchestration_ablations.py frozen Phase 6 versus Phase 7 confirmation
scripts/run_profile_ablations.py  sealed Phase 7 versus Phase 9 profile test
scripts/run_bm25_rescue_ablations.py one-shot Phase 9 versus Phase 10 test
scripts/run_multislot_intent_ablations.py one-shot generator-separated Phase 11 test
scripts/run_route_redundancy_ablations.py sealed Phase 12 development test
scripts/run_intent_epoch_slate_ablations.py sealed Phase 13 three-suite test
scripts/verify_phase7_stage_a_oracle.py exact frozen Stage-A differential oracle
scripts/verify_phase9_ranking_oracle.py exact frozen Phase 9 ranking oracle
scripts/verify_phase10_phase9_exact_oracle.py exact rescue-fallback oracle
scripts/verify_phase11_intent_oracle.py 30,000-case intent transition oracle
scripts/verify_phase12_route_redundancy_oracle.py rank-fusion differential oracle
scripts/verify_phase13_slate_oracle.py intent-epoch slate transition oracle
scripts/run_retrieval_audit.py    label-separated retrieval loss waterfall
scripts/record_benchmark.py       strict append-only aggregate recorder
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
