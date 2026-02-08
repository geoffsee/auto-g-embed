# Local Semantic Embedding Pipeline

This pipeline replaces the hash baseline with a learned local embedder:

1. Ingest + clean Kaggle pairs with `polars` (Rust CLI by default)
2. Build contrastive train/eval sets with negative sampling
3. Train a local Rust contrastive embedder (default) or Python sentence-transformers fallback
4. Export ONNX runtime assets (Python sentence-transformers path)
5. Optionally publish a SentenceTransformer package to Hugging Face Hub
6. Run native Rust inference directly from saved local weights

## One-command run

```bash
./training/run_pipeline.sh --run-rust-example
```

Useful variants:

```bash
# Force Python prep backend (Rust prep is default).
./training/run_pipeline.sh --prep-backend python

# Force Python training/export backend (Rust training is default).
./training/run_pipeline.sh --train-backend python

# Use local CSV instead of Kaggle download.
./training/run_pipeline.sh --source-csv /path/to/train.csv

# Pick a different dataset adapter profile.
./training/run_pipeline.sh --profile kaggle_questions_million

# Skip dependency install and training (export only from existing model dir).
./training/run_pipeline.sh --skip-install --skip-train

# Train Python backend and publish SentenceTransformer package to HF.
./training/run_pipeline.sh \
  --train-backend python \
  --publish-hf \
  --hf-repo-id your-user/auto-g-embed-st

# Show commands only.
./training/run_pipeline.sh --dry-run
```

You can list available adapter profiles:

```bash
cargo run --bin prepare_contrastive -- --list-profiles
```

## Dataset Adapters

Profiles are defined in `training/dataset_profiles.json`.

- `quora_pairs`: pair-labeled duplicate question dataset
- `kaggle_questions_million`: single-question dataset mined into positives via duplicate normalization

This keeps source-specific behavior behind adapter abstractions and makes dataset A/B evaluation a flag change.

## 1) Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```

## 2) Prepare contrastive data with Polars (Rust-first)

When downloading directly from Kaggle, the Rust prep CLI calls `kaggle datasets download`.

```bash
cargo run --bin prepare_contrastive -- \
  --profile quora_pairs \
  --out-dir artifacts/contrastive-data
```

If Kaggle credentials are not configured, pass a local CSV:

```bash
cargo run --bin prepare_contrastive -- \
  --profile kaggle_questions_million \
  --source-csv /path/to/train.csv \
  --out-dir artifacts/contrastive-data
```

If you need the legacy Python prep path:

```bash
python3 training/prepare_kaggle_contrastive.py --list-profiles
```

Outputs:

- `artifacts/contrastive-data/train_pairs.parquet`
- `artifacts/contrastive-data/eval_triplets.parquet`
- `artifacts/contrastive-data/metadata.json`

## 3) Train sentence-transformer model

Rust-native training (default pipeline backend):

```bash
cargo run --bin train_rust_embedder -- \
  --train-pairs artifacts/contrastive-data/train_pairs.parquet \
  --eval-triplets artifacts/contrastive-data/eval_triplets.parquet \
  --output-dir artifacts/model/rust-embedder \
  --epochs 3 \
  --embedding-dim 512
```

Outputs:

- `artifacts/model/rust-embedder/config.json`
- `artifacts/model/rust-embedder/token_embeddings.f32`
- `artifacts/model/rust-embedder/metrics.json`

Python fallback training:

```bash
python3 training/train_sentence_transformer.py \
  --train-pairs artifacts/contrastive-data/train_pairs.parquet \
  --eval-triplets artifacts/contrastive-data/eval_triplets.parquet \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 2 \
  --batch-size 64 \
  --output-dir artifacts/model
```

Outputs:

- `artifacts/model/sentence-transformer/`
- `artifacts/model/metrics.json`

## 4) Export ONNX for Rust runtime (Python fallback path only)

```bash
python3 training/export_onnx_model.py \
  --sentence-model-dir artifacts/model/sentence-transformer \
  --output-dir artifacts/model/onnx \
  --max-length 128
```

Outputs:

- `artifacts/model/onnx/model.onnx`
- `artifacts/model/onnx/tokenizer.json`
- `artifacts/model/onnx/embedder_config.json`

## 5) Publish SentenceTransformer package to Hugging Face

```bash
python3 training/publish_sentence_transformer.py \
  --sentence-model-dir artifacts/model/sentence-transformer \
  --repo-id your-user/auto-g-embed-st \
  --commit-message "Publish sentence-transformer artifact"
```

Optional flags:

- `--token <HF_TOKEN>`: explicit token (otherwise uses CLI auth / `HF_TOKEN`).
- `--private`: create repo as private (if repo does not already exist).
- `--create-pr`: upload via PR instead of direct commit.
- `--replace-root`: replace root files in the target repo.

Pipeline shortcut:

```bash
./training/run_pipeline.sh \
  --train-backend python \
  --publish-hf \
  --hf-repo-id your-user/auto-g-embed-st
```

## 6) Run native Rust embedding inference

```bash
cargo run --example rust_embed -- \
  artifacts/model/rust-embedder \
  "A quick test sentence for semantic embeddings."
```

ONNX runtime example (Python fallback path):

```bash
cargo run --example semantic_embed --features semantic -- \
  artifacts/model/onnx \
  "A quick test sentence for semantic embeddings."
```
