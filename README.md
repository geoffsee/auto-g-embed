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

## Additional docs

- Training and pipeline details: `training/README.md`
- Test data notes: `test-data/README.md`
