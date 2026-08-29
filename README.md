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
runtime code and overhead were removed, and Phase 7 remains the active
submission. See `docs/phase8_results.json`.

Before enabling dense search, runtime startup verifies the pinned model and
tokenizer, every vector shard, the ID array, and the exact catalog checksum.
Dense and BM25 routes degrade independently; a build without SQLite FTS5 can
still use dense retrieval and deterministic fallback results.

## Offline Dense-Retrieval Foundation

The repository includes a CPU-only local embedding setup for the next agent
iteration. It does not require an API key, network service, vector database,
PyTorch, Transformers, SentenceTransformers, CUDA, or MPS.

- Encoder: `BAAI/bge-small-en-v1.5`, pinned to revision
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- Runtime graph: deterministic per-channel dynamic INT8 ONNX, 384 dimensions
- Query instruction: `Represent this sentence for searching relevant passages: `
- Pooling: normalized CLS vector
- Runtime: NumPy, ONNX Runtime CPU, and the Rust `tokenizers` package
- Model bundle: `assets/bge-small-en-v1.5-int8` (about 33 MB)

The submitted model is derived from the upstream FP32 ONNX graph whose SHA-256
is `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35`.
Its own manifest records every input file, checksum, quantization argument,
package version, CPU smoke test, and FP32 fidelity result. See
`BGE_MODEL_ATTRIBUTION.md` for the MIT attribution.

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

The frozen 50,000-product catalog has been encoded into the finalized bundle at
`assets/search-index-bge-small-en-v1.5-v2`. Its manifest records build identity
`fdfcd830321690d96cd87754db62161b5264485803a08ed1b30f4a0c33c227c8` and
logical embedding SHA-256
`beaabadaa1f13cf0177f7ca02b6aa9a869392c2f7ed4fa8e9b9e30c6467d0ebb`.

The reproducible preprocessing command is:

```bash
python3 -m scripts.preprocess_catalog build \
  --catalog data/catalog.jsonl \
  --model-assets assets/bge-small-en-v1.5-int8 \
  --output assets/search-index-bge-small-en-v1.5-v2
```

The build streams the 50,000 JSONL rows, constructs deterministic search text
transiently, and writes exactly four row-aligned float32 embedding shards. It
never stores a second copy of the product text. At runtime,
`starter/dense.py` memory-maps and scores one shard at a time, so all vectors
are never duplicated in RAM. The preprocessing command defaults to one ONNX
compute thread and sequential batches of four products to keep laptop thermal
load conservative; faster settings must be requested explicitly.

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
docs/retrieval_contract.md        product, intent-state, and query contracts
benchmarks/                       append-only aggregate scores and diagnostics
starter/agent.py                  thin competition API adapter
starter/dense.py                  memory-mapped exact dense scorer
conversational_search/intent.py   immutable session state and query renderer
conversational_search/retrieval.py BM25+dense retrieval and deterministic RRF
conversational_search/questions.py measured clarification policies
conversational_search/strategy.py bounded decision signals and fusion policies
conversational_search/ranking.py  bounded deterministic Stage-A scorer
conversational_search/slates.py   bounded stagnation-aware slate memory
conversational_search/orchestration.py exact SEARCH/REUSE/SKIP planner
conversational_search/service.py  Agent orchestration and strategy execution
preprocessing/catalog.py          deterministic catalog text normalization
preprocessing/encoder.py          offline CPU BGE query/document encoder
preprocessing/embeddings.py       atomic four-shard artifact builder
scripts/prepare_bge_model.py      reproducible pinned INT8 model derivation
scripts/preprocess_catalog.py     catalog scan/build/verification CLI
scripts/run_policy_ablations.py   sequential shared-backend policy comparison
scripts/run_fusion_ablations.py   sequential route-weight A/B comparison
scripts/run_reranking_ablations.py sequential fused-only versus Stage-A A/B
scripts/run_exploration_ablations.py sequential Phase 4 versus Phase 5 A/B
scripts/run_message_robustness.py seeded Phase 5 versus Phase 6 robustness test
scripts/run_orchestration_ablations.py frozen Phase 6 versus Phase 7 confirmation
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
