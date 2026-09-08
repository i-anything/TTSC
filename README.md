# Shopping Copilot

Our TikTok TechJam 2026 Track 4 submission: a deterministic, offline shopping
assistant for the frozen 50,000-product catalog. It combines lexical and dense
retrieval, explicit preference state, catalog evidence, and adaptive questions
and recommendation widths across a ten-turn conversation.

## Run

Use **Python 3.13**. No GPU, API key, or runtime network connection is required.
Install the three pinned CPU runtime dependencies once:

```bash
python3.13 -m venv .venv-runtime
.venv-runtime/bin/python -m pip install -r requirements-runtime.txt
.venv-runtime/bin/python -m scripts.check_submission
.venv-runtime/bin/python -m evaluator.local_evaluator
```

On Windows, use `py -3.13` to create the environment and
`.venv-runtime\Scripts\python.exe` for the subsequent commands.

The repository includes the compressed catalog, quantized ONNX model,
tokenizer, and four embedding shards. `scripts.check_submission` verifies and
expands the catalog, validates the index checksums, loads every retrieval
backend, and checks offline, deterministic reset/respond behavior. It refuses
to overwrite an existing catalog with a different checksum. No model or
catalog download is needed after cloning.

The official evaluator writes `results.json`. It loads the unchanged public
200-session development set; the runtime agent itself only reads catalog and
model/index assets.

## Method

1. A bounded language parser recognizes explicit shopping requests, answers,
   preference changes, and declines. It preserves product values and shares
   its interpretation between immutable intent state and transcript replay.
2. SQLite FTS5 BM25 and the local BGE-small INT8 ONNX encoder retrieve products.
   Exact, complete narrow support can avoid unnecessary dense inference.
3. Evidence ranking distinguishes confirmed, unknown, and contradicted
   requirements. Generic profile themes remain a bounded secondary signal.
4. For supported dialogue shapes, complete transcript replay reconstructs
   candidate support from the full catalog. Continuation eliminates only
   previously shown candidates eligible under that transcript.
5. The planner chooses a question and recommendation width using the published
   score and remaining turns. Exact dependency caching and intent-aware slate
   novelty avoid redundant work and repeated suggestions.

Unrecognized language, incomplete evidence, or unsupported transcripts use
ordinary hybrid retrieval. Normalizing a sentence never establishes catalog
support by itself. Successful exposure decisions now appear correctly in
`last_action_trace(session_id)` alongside the actual question and width.

## Agent interface

Both `agent.py` and `starter/agent.py` export the same `Agent`:

```python
from agent import Agent

agent = Agent()  # Resolves the bundled catalog independently of working directory.
agent.reset("shopper", {"preference_tags": []})
response = agent.respond(
    "shopper", "I am shopping for Shoes. It must be waterproof.", 1, 10
)
```

`respond` returns `message`, `ask_attribute`, ordered unique
`recommendations` containing `parent_asin`, and zero prompt/completion tokens.
Call `reset` before each session; turns must increase from 1 through 10.

## Validation

```bash
.venv-runtime/bin/python -m unittest discover -s tests
.venv-runtime/bin/python -m scripts.check_submission
.venv-runtime/bin/python -m evaluator.local_evaluator
```

See [evaluation results](docs/EVALUATION.md) for the final measured metrics,
latency, and scope of validation, and [architecture](docs/ARCHITECTURE.md) for
the component boundaries. The organizer's API, evaluator, and public data are
unchanged.

## Model, cost, and limitations

The encoder is `BAAI/bge-small-en-v1.5`, pinned to revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, with INT8 weights, 384-dimensional
CLS embeddings, and float32 memory-mapped product vectors. It provides local
semantic recall without a generative model. There are **zero API calls, zero
generative tokens, and $0 external model charges**; local CPU and memory are
still required. Dense initialization failure falls back to lexical retrieval;
the submission check treats that degraded setup as a failure to fix before
release.

Language coverage is bounded, not unrestricted conversational understanding.
The strongest planning path depends on the published dialogue contract and
catalog-derived disclosure cards. Catalog ambiguity, absent product facts,
unfamiliar phrasing, and a different hidden-target distribution can reduce
performance. Public and synthetic results do not establish private-final
performance.

The model's MIT notice is retained in
[BGE_MODEL_ATTRIBUTION.md](BGE_MODEL_ATTRIBUTION.md). Organizer data and
challenge documentation retain their original source and notices. To rebuild
the index, install `requirements-preprocessing.txt` and run:

```bash
.venv-runtime/bin/python -m scripts.preprocess_catalog build \
  --catalog data/catalog.jsonl \
  --model-assets assets/bge-small-en-v1.5-int8 \
  --output assets/search-index-bge-small-en-v1.5-v2
```
