---
license: mit
library_name: custom
pipeline_tag: feature-extraction
tags:
  - sentence-embeddings
  - text-embeddings
  - rust
---

# auto-g-embed

Local semantic embedding pipeline with a Rust-native runtime.

## What this repo provides

- Contrastive dataset preparation (`prepare_contrastive`)
- Rust-native embedder training (`train_rust_embedder`)
- Runtime embedding APIs and examples
- Optional ONNX/SentenceTransformer path in `training/`
- SentenceTransformer publish tooling for Hugging Face (`training/publish_sentence_transformer.py`)

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

To publish a SentenceTransformer-formatted repo for direct MTEB/SBERT loading:

```bash
./training/run_pipeline.sh \
  --train-backend python \
  --publish-hf \
  --hf-repo-id your-user/auto-g-embed-st
```

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

### Comparison Chart

| Model | embeds_per_second | p50_us | p95_us | p99_us | top1_accuracy | separation |
|---|---:|---:|---:|---:|---:|---:|
| auto-g-embed (local smoke run) | 219595.18 | 3.88 | 6.54 | 6.71 | 0.9375 | 0.2886 |
| Llama-3.2-NV-EmbedQA-1B-v2 | 140.7 | 7000 | 8000 | N/R | N/R | N/R |
| Llama-3.2-NeMo-Retriever-300M-Embed-V1 | 126.0 | 8000 | 8300 | N/R | N/R | N/R |
| NV-EmbedQA-E5-v5 | 196.3 | 5100 | 5400 | N/R | N/R | N/R |
| NV-EmbedQA-Mistral7B-v2 | 67.9 | 14600 | 15400 | N/R | N/R | N/R |
| SwiftEmbed (paper) | 50000 | 1120 | N/R | N/R | N/R | N/R |

Notes:

- `N/R` means not reported in the source.
- External numbers are from different hardware and workloads, so this is directional and not an apples-to-apples benchmark.
- Source links:
  - NVIDIA NIM performance tables: https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/performance.html
  - SwiftEmbed paper: https://arxiv.org/abs/2510.24793

## MTEB Results

Latest MTEB run (Hugging Face Job, `cpu-upgrade`) (February 8, 2026):

```bash
hf jobs uv run --with mteb --with sentence-transformers --with torch --with numpy \
  --flavor cpu-upgrade --timeout 6h --detach scripts/run_real_eval.py \
  --model geoffsee/auto-g-embed-st --device cpu \
  --tasks SciFact,NFCorpus,FiQA2018 --batch-size 64 --skip-perf \
  --mteb-output-dir artifacts/evals/mteb_real --output-json latest.json
```

- Job: https://huggingface.co/jobs/geoffsee/698915cab6db0e80325e19e8
- `SciFact` main score: `0.64872`
- `NFCorpus` main score: `0.31141`
- `FiQA2018` main score: `0.36869`
- `avg_main_score` (3-task mean): `0.44294`

### MTEB Comparison Chart

| Run | Runtime | SciFact | NFCorpus | FiQA2018 | avg_main_score |
|---|---|---:|---:|---:|---:|
| auto-g-embed (local eval, M4 Max) | `mps` | 0.64872 | 0.31141 | 0.36869 | 0.44294 |
| auto-g-embed-st (Hugging Face Job) | `cpu` (`cpu-upgrade`) | 0.64872 | 0.31141 | 0.36869 | 0.44294 |
| sentence-transformers/all-MiniLM-L6-v2 | `cpu/mps` | 0.64508 | 0.31594 | 0.36867 | 0.44323 |
| BAAI/bge-small-en-v1.5 | `cpu/mps` | 0.71273 | 0.34264 | 0.40343 | 0.48627 |
| intfloat/e5-small-v2 | `cpu/mps` | 0.68854 | 0.32449 | 0.37434 | 0.46246 |

Notes:

- Local values are from `artifacts/evals/real_eval/latest.json`.
- HF job used the published model repo `geoffsee/auto-g-embed-st` with the same task set and batch size.
- Comparable model values are from `mteb/results` on Hugging Face (dataset commit `5a7430bdc3c58b3c8be7e8eed4c7bb990f7d554c`), using each task file's `scores.test[0].main_score`.

## Additional docs

- Training and pipeline details: `training/README.md`
- Test data notes: `test-data/README.md`
