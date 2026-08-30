# Generated search assets

The active runtime assets are:

- `bge-small-en-v1.5-int8`: pinned MIT-licensed BGE-small v1.5 CPU ONNX
  bundle. ONNX Runtime consumes the checksummed graph directly; there is no
  runtime download or reassembly.
- `search-index-bge-small-en-v1.5-v2`: four row-aligned, memory-mapped float32
  shards for 50,000 products in the same 384-dimensional embedding space.

The rejected Arctic 768-dimensional experiment remains available only as
ignored local research tooling. It is neither an active default nor a submitted
runtime asset. Encoder and index packages from different embedding spaces must
never be mixed.

Reproducible commands and output locations are documented in the repository
README. Every generated bundle contains its own checksums and refuses to load
unless its manifest is finalized. The active BGE license and source notice are
in `BGE_MODEL_ATTRIBUTION.md` at the repository root and inside the model bundle.

Canonical product text is transient during preprocessing and is not stored here.
