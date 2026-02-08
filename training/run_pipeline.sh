#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_REF=""
PROFILE="quora_pairs"
PROFILES_FILE="training/dataset_profiles.json"
PREP_BACKEND="rust"
TRAIN_BACKEND="rust"
SOURCE_CSV=""
CSV_NAME=""
OUT_CONTRASTIVE="artifacts/contrastive-data"
OUT_MODEL="artifacts/model"
BASE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
HF_REPO_ID=""
HF_COMMIT_MESSAGE="Publish sentence-transformer artifact"
HF_TOKEN="${HF_TOKEN:-}"

EPOCHS=2
BATCH_SIZE=64
LEARNING_RATE=0.06
MARGIN=0.2
VOCAB_SIZE=8192
EMBEDDING_DIM=512
MAX_SEQ_LEN=64
MAX_LENGTH=128
MAX_TRAIN_PAIRS=250000
MAX_EVAL_TRIPLETS=10000
SEED=17

SKIP_INSTALL=0
SKIP_TRAIN=0
RUN_RUST_EXAMPLE=0
PUBLISH_HF=0
HF_PRIVATE=0
HF_CREATE_PR=0
HF_REPLACE_ROOT=0
DRY_RUN=0
EXAMPLE_TEXT="A quick test sentence for semantic embeddings."

usage() {
  cat <<'EOF'
Usage:
  training/run_pipeline.sh [options]

Options:
  --prep-backend NAME        Data prep backend: rust or python (default: rust)
  --train-backend NAME       Train backend: rust or python (default: rust)
  --profile NAME             Dataset profile name for prep adapter (default: quora_pairs)
  --profiles-file PATH       Profile JSON file (default: training/dataset_profiles.json)
  --dataset-ref REF          Kaggle dataset ref override (defaults to profile setting)
  --source-csv PATH          Use local CSV instead of Kaggle download
  --csv-name NAME            Override CSV filename in downloaded dataset folder
  --out-contrastive DIR      Output dir for prepared contrastive data
  --out-model DIR            Output dir for trained + exported model
  --base-model MODEL         Base sentence-transformers model
  --publish-hf               Publish sentence-transformer artifact to Hugging Face
  --hf-repo-id ID            HF target model repo id (required with --publish-hf)
  --hf-token TOKEN           HF token (default: HF_TOKEN env var)
  --hf-private               Create repo as private if it does not exist
  --hf-create-pr             Upload as pull request instead of direct commit
  --hf-replace-root          Replace repo root contents on upload
  --hf-commit-message TEXT   Commit message for HF upload
  --epochs N                 Training epochs (default: 2)
  --batch-size N             Training batch size (default: 64)
  --learning-rate F          Rust trainer learning rate (default: 0.06)
  --margin F                 Rust trainer hinge margin (default: 0.2)
  --vocab-size N             Rust trainer hash vocab size (default: 8192)
  --embedding-dim N          Rust trainer embedding dim (default: 512)
  --max-seq-len N            Rust trainer max token length (default: 64)
  --max-train-pairs N        Max positive train pairs for prep/train
  --max-eval-triplets N      Max eval triplets for prep
  --max-length N             Exported runtime max sequence length (default: 128)
  --seed N                   Random seed (default: 17)
  --skip-install             Skip venv dependency installation
  --skip-train               Skip training step (use existing model for export)
  --run-rust-example         Run Rust semantic embedding example after export
  --example-text TEXT        Text for rust example when --run-rust-example is set
  --dry-run                  Print commands without executing
  -h, --help                 Show this help

Environment:
  PYTHON_BIN                 Python executable to use (default: python3)
  HF_TOKEN                   Hugging Face token used for --publish-hf uploads
EOF
}

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prep-backend)
      PREP_BACKEND="$2"
      shift 2
      ;;
    --train-backend)
      TRAIN_BACKEND="$2"
      shift 2
      ;;
    --dataset-ref)
      DATASET_REF="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --profiles-file)
      PROFILES_FILE="$2"
      shift 2
      ;;
    --source-csv)
      SOURCE_CSV="$2"
      shift 2
      ;;
    --csv-name)
      CSV_NAME="$2"
      shift 2
      ;;
    --out-contrastive)
      OUT_CONTRASTIVE="$2"
      shift 2
      ;;
    --out-model)
      OUT_MODEL="$2"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --publish-hf)
      PUBLISH_HF=1
      shift
      ;;
    --hf-repo-id)
      HF_REPO_ID="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN="$2"
      shift 2
      ;;
    --hf-private)
      HF_PRIVATE=1
      shift
      ;;
    --hf-create-pr)
      HF_CREATE_PR=1
      shift
      ;;
    --hf-replace-root)
      HF_REPLACE_ROOT=1
      shift
      ;;
    --hf-commit-message)
      HF_COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --margin)
      MARGIN="$2"
      shift 2
      ;;
    --vocab-size)
      VOCAB_SIZE="$2"
      shift 2
      ;;
    --embedding-dim)
      EMBEDDING_DIM="$2"
      shift 2
      ;;
    --max-seq-len)
      MAX_SEQ_LEN="$2"
      shift 2
      ;;
    --max-train-pairs)
      MAX_TRAIN_PAIRS="$2"
      shift 2
      ;;
    --max-eval-triplets)
      MAX_EVAL_TRIPLETS="$2"
      shift 2
      ;;
    --max-length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --run-rust-example)
      RUN_RUST_EXAMPLE=1
      shift
      ;;
    --example-text)
      EXAMPLE_TEXT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

NEEDS_PYTHON=0
if [[ "$PREP_BACKEND" == "python" || "$TRAIN_BACKEND" == "python" || "$PUBLISH_HF" -eq 1 ]]; then
  NEEDS_PYTHON=1
fi

if [[ "$NEEDS_PYTHON" -eq 1 ]]; then
  if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
      run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    run_cmd python -m pip install --upgrade pip
    run_cmd pip install -r training/requirements.txt
  else
    if [[ -d "$VENV_DIR" ]]; then
      # shellcheck disable=SC1091
      source "$VENV_DIR/bin/activate"
    fi
  fi
fi

mkdir -p "$OUT_CONTRASTIVE" "$OUT_MODEL"

if [[ "$PREP_BACKEND" == "rust" ]]; then
  prepare_cmd=(
    cargo run --bin prepare_contrastive --
    --profiles-file "$PROFILES_FILE"
    --profile "$PROFILE"
    --out-dir "$OUT_CONTRASTIVE"
    --max-train-pairs "$MAX_TRAIN_PAIRS"
    --max-eval-triplets "$MAX_EVAL_TRIPLETS"
    --seed "$SEED"
  )
elif [[ "$PREP_BACKEND" == "python" ]]; then
  prepare_cmd=(
    python3 training/prepare_kaggle_contrastive.py
    --profiles-file "$PROFILES_FILE"
    --profile "$PROFILE"
    --out-dir "$OUT_CONTRASTIVE"
    --max-train-pairs "$MAX_TRAIN_PAIRS"
    --max-eval-triplets "$MAX_EVAL_TRIPLETS"
    --seed "$SEED"
  )
else
  echo "Invalid --prep-backend: '$PREP_BACKEND' (expected: rust or python)" >&2
  exit 1
fi

if [[ -n "$SOURCE_CSV" ]]; then
  prepare_cmd+=(--source-csv "$SOURCE_CSV")
else
  if [[ -n "$DATASET_REF" ]]; then
    prepare_cmd+=(--dataset-ref "$DATASET_REF")
  fi
  if [[ -n "$CSV_NAME" ]]; then
    prepare_cmd+=(--csv-name "$CSV_NAME")
  fi
fi
run_cmd "${prepare_cmd[@]}"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  if [[ "$TRAIN_BACKEND" == "rust" ]]; then
    run_cmd cargo run --bin train_rust_embedder -- \
      --train-pairs "$OUT_CONTRASTIVE/train_pairs.parquet" \
      --eval-triplets "$OUT_CONTRASTIVE/eval_triplets.parquet" \
      --output-dir "$OUT_MODEL/rust-embedder" \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --margin "$MARGIN" \
      --max-train-pairs "$MAX_TRAIN_PAIRS" \
      --seed "$SEED" \
      --vocab-size "$VOCAB_SIZE" \
      --embedding-dim "$EMBEDDING_DIM" \
      --max-seq-len "$MAX_SEQ_LEN"
  elif [[ "$TRAIN_BACKEND" == "python" ]]; then
    run_cmd python3 training/train_sentence_transformer.py \
      --train-pairs "$OUT_CONTRASTIVE/train_pairs.parquet" \
      --eval-triplets "$OUT_CONTRASTIVE/eval_triplets.parquet" \
      --base-model "$BASE_MODEL" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --max-train-pairs "$MAX_TRAIN_PAIRS" \
      --seed "$SEED" \
      --output-dir "$OUT_MODEL"
  else
    echo "Invalid --train-backend: '$TRAIN_BACKEND' (expected: rust or python)" >&2
    exit 1
  fi
fi

if [[ "$TRAIN_BACKEND" == "python" ]]; then
  run_cmd python3 training/export_onnx_model.py \
    --sentence-model-dir "$OUT_MODEL/sentence-transformer" \
    --output-dir "$OUT_MODEL/onnx" \
    --max-length "$MAX_LENGTH" \
    --normalize
fi

if [[ "$PUBLISH_HF" -eq 1 ]]; then
  if [[ -z "$HF_REPO_ID" ]]; then
    echo "--hf-repo-id is required when --publish-hf is set." >&2
    exit 1
  fi
  if [[ ! -d "$OUT_MODEL/sentence-transformer" ]]; then
    echo "Missing sentence-transformer directory: $OUT_MODEL/sentence-transformer" >&2
    echo "Use --train-backend python or point --out-model to an existing sentence-transformer export." >&2
    exit 1
  fi

  publish_cmd=(
    python3 training/publish_sentence_transformer.py
    --sentence-model-dir "$OUT_MODEL/sentence-transformer"
    --repo-id "$HF_REPO_ID"
    --commit-message "$HF_COMMIT_MESSAGE"
  )
  if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN
  fi
  if [[ "$HF_PRIVATE" -eq 1 ]]; then
    publish_cmd+=(--private)
  fi
  if [[ "$HF_CREATE_PR" -eq 1 ]]; then
    publish_cmd+=(--create-pr)
  fi
  if [[ "$HF_REPLACE_ROOT" -eq 1 ]]; then
    publish_cmd+=(--replace-root)
  fi

  run_cmd "${publish_cmd[@]}"
fi

if [[ "$RUN_RUST_EXAMPLE" -eq 1 ]]; then
  if [[ "$TRAIN_BACKEND" == "rust" ]]; then
    run_cmd cargo run --example rust_embed -- \
      "$OUT_MODEL/rust-embedder" \
      "$EXAMPLE_TEXT"
  else
    run_cmd cargo run --example semantic_embed --features semantic -- \
      "$OUT_MODEL/onnx" \
      "$EXAMPLE_TEXT"
  fi
fi

echo
echo "Pipeline complete."
echo "Contrastive data: $OUT_CONTRASTIVE"
if [[ "$TRAIN_BACKEND" == "rust" ]]; then
  echo "Rust model:       $OUT_MODEL/rust-embedder"
else
  echo "Sentence model:   $OUT_MODEL/sentence-transformer"
  echo "ONNX export:      $OUT_MODEL/onnx"
fi
if [[ "$PUBLISH_HF" -eq 1 ]]; then
  echo "Published repo:   $HF_REPO_ID"
fi
