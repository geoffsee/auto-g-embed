# auto-g-embed

Local semantic embedding pipeline with a Rust-native runtime.

## What this repo provides

- Contrastive dataset preparation (`prepare_contrastive`)
- Rust-native embedder training (`train_rust_embedder`)
- Runtime embedding APIs and examples
- Optional ONNX/SentenceTransformer path in `training/`

## Quick start

```bash
cargo test

./training/run_pipeline.sh \
  --profile kaggle_questions_million \
  --source-csv data/kaggle/one-million-reddit-questions.csv
```

Run the Rust embedding example:

```bash
cargo run --example rust_embed -- \
  artifacts/model/rust-embedder \
  "A quick test sentence for semantic embeddings."
```

## Model artifacts

Published model artifacts are available on Hugging Face:

- https://huggingface.co/geoffsee/auto-g-embed

## Project layout

- `src/`: library modules and binaries
- `examples/`: runnable embedding demos
- `tests/`: integration/performance tests
- `training/`: pipeline scripts and dataset adapters

## Development checks

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Community Benchmark

Run the reproducible benchmark CLI:

```bash
cargo run --release --bin community_benchmark -- \
  --output artifacts/benchmarks/latest.json
```

The output includes throughput, latency percentiles (`p50/p95/p99`), retrieval quality metrics, and environment metadata for publishing.
Methodology and reporting guidance: `BENCHMARKS.md`.

Latest Benchmark (M4 Max) (February 8, 2026):

```bash
cargo run --release --bin community_benchmark -- \
  --eval-count 500 --warmup-count 100 --query-count 32 \
  --output artifacts/benchmarks/smoke.json
```

- `embeds_per_second`: `219595.18`
- `p50_us`: `3.88`
- `p95_us`: `6.54`
- `p99_us`: `6.71`
- `top1_accuracy`: `0.9375`
- `separation`: `0.2886`

## Additional docs

- Training and pipeline details: `training/README.md`
- Test data notes: `test-data/README.md`
