# Optional grounded intent model

The runtime looks for `Qwen3-1.7B-Q4_K_M.gguf` in this directory. The model is
not committed. Prepare it with:

```bash
python -m scripts.prepare_local_intent_model
```

The downloader is pinned to the `ggml-org/Qwen3-1.7B-GGUF` conversion at
revision `daeb8e2d528a760970442092f6bf1e55c3b659eb` and verifies SHA-256
`d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5`.

The source model is `Qwen/Qwen3-1.7B`, licensed under Apache License 2.0. The
model is optional: without the GGUF file and `llama-cpp-python`, the agent keeps
the deterministic parser and its existing fail-open behavior.
