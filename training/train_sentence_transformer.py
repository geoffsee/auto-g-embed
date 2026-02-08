#!/usr/bin/env python3
"""Train a compact local sentence embedder with MultipleNegativesRankingLoss."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pairs",
        type=Path,
        default=Path("artifacts/contrastive-data/train_pairs.parquet"),
        help="Parquet path with columns: anchor, positive.",
    )
    parser.add_argument(
        "--eval-triplets",
        type=Path,
        default=Path("artifacts/contrastive-data/eval_triplets.parquet"),
        help="Parquet path with columns: anchor, positive, negative.",
    )
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers base model to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-pairs", type=int, default=250_000)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model"),
        help="Directory where sentence-transformer model + metrics are written.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_examples(train_pairs_path: Path, max_pairs: int, seed: int) -> list[InputExample]:
    frame = pl.read_parquet(train_pairs_path).select(["anchor", "positive"])
    if frame.height == 0:
        raise RuntimeError(f"Train file has no rows: {train_pairs_path}")
    if max_pairs > 0 and frame.height > max_pairs:
        frame = frame.sample(n=max_pairs, with_replacement=False, shuffle=True, seed=seed)

    examples: list[InputExample] = []
    for anchor, positive in frame.iter_rows():
        examples.append(InputExample(texts=[anchor, positive]))
    return examples


def evaluate_triplet_quality(
    model: SentenceTransformer,
    eval_triplets_path: Path,
    batch_size: int,
) -> dict[str, float]:
    frame = pl.read_parquet(eval_triplets_path).select(["anchor", "positive", "negative"])
    if frame.height == 0:
        raise RuntimeError(f"Eval file has no rows: {eval_triplets_path}")

    anchors = frame["anchor"].to_list()
    positives = frame["positive"].to_list()
    negatives = frame["negative"].to_list()

    anchor_embeddings = model.encode(
        anchors,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    positive_embeddings = model.encode(
        positives,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    negative_embeddings = model.encode(
        negatives,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    positive_scores = np.sum(anchor_embeddings * positive_embeddings, axis=1)
    negative_scores = np.sum(anchor_embeddings * negative_embeddings, axis=1)

    wins = (positive_scores > negative_scores).astype(np.float32)
    margin = positive_scores - negative_scores

    return {
        "triplet_accuracy": float(np.mean(wins)),
        "mean_positive_similarity": float(np.mean(positive_scores)),
        "mean_negative_similarity": float(np.mean(negative_scores)),
        "mean_margin": float(np.mean(margin)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.train_pairs.exists():
        raise FileNotFoundError(
            f"Missing train pairs file: {args.train_pairs}. "
            "Run training/prepare_kaggle_contrastive.py first."
        )
    if not args.eval_triplets.exists():
        raise FileNotFoundError(
            f"Missing eval triplets file: {args.eval_triplets}. "
            "Run training/prepare_kaggle_contrastive.py first."
        )

    train_examples = load_train_examples(
        args.train_pairs,
        max_pairs=args.max_train_pairs,
        seed=args.seed,
    )
    print(f"[train] loaded {len(train_examples)} train pairs")

    model = SentenceTransformer(args.base_model)
    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=False,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = max(1, len(train_loader) * max(args.epochs, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)

    sentence_model_dir = args.output_dir / "sentence-transformer"
    sentence_model_dir.parent.mkdir(parents=True, exist_ok=True)

    print(
        "[train] base_model=%s epochs=%d batch_size=%d lr=%g warmup_steps=%d"
        % (
            args.base_model,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            warmup_steps,
        )
    )
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(sentence_model_dir),
        optimizer_params={"lr": args.learning_rate},
        show_progress_bar=True,
    )

    metrics = evaluate_triplet_quality(
        model=model,
        eval_triplets_path=args.eval_triplets,
        batch_size=args.eval_batch_size,
    )
    metrics.update(
        {
            "base_model": args.base_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_pairs_used": len(train_examples),
            "seed": args.seed,
            "sentence_model_dir": str(sentence_model_dir),
        }
    )

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[train] wrote model to %s" % sentence_model_dir)
    print("[train] wrote metrics to %s" % metrics_path)
    print(
        "[train] triplet_accuracy=%.4f mean_margin=%.4f"
        % (metrics["triplet_accuracy"], metrics["mean_margin"])
    )


if __name__ == "__main__":
    main()
