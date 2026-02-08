#!/usr/bin/env python3
"""Export a fine-tuned sentence-transformer backbone to ONNX for Rust inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentence-model-dir",
        type=Path,
        default=Path("artifacts/model/sentence-transformer"),
        help="Directory produced by train_sentence_transformer.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model/onnx"),
        help="Directory to write ONNX + tokenizer files.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Recommended inference max length (saved to embedder_config.json).",
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="L2 normalize pooled sentence embeddings in Rust runtime.",
    )
    normalize_group.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalization in Rust runtime.",
    )
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def resolve_transformer_dir(sentence_model_dir: Path) -> Path:
    nested = sentence_model_dir / "0_Transformer"
    if nested.exists():
        return nested
    return sentence_model_dir


def main() -> None:
    args = parse_args()

    transformer_dir = resolve_transformer_dir(args.sentence_model_dir)
    if not transformer_dir.exists():
        raise FileNotFoundError(
            f"Could not find transformer weights at: {transformer_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading transformer from: {transformer_dir}")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        transformer_dir,
        export=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(transformer_dir, use_fast=True)
    config = AutoConfig.from_pretrained(transformer_dir)

    ort_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "pooling": "mean",
        "normalize": bool(args.normalize),
        "max_length": args.max_length,
        "hidden_size": getattr(config, "hidden_size", None),
        "transformer_dir": str(transformer_dir),
        "onnx_model_file": "model.onnx",
    }
    metadata_path = args.output_dir / "embedder_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[export] wrote ONNX model to: {args.output_dir}")
    print(f"[export] wrote runtime config to: {metadata_path}")


if __name__ == "__main__":
    main()
