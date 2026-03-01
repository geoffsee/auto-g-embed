#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers>=4.57",
#     "sentence-transformers",
#     "onnx",
#     "onnxruntime",
#     "onnxscript",
# ]
# ///
"""
Export Octen-Embedding-0.6B to ONNX with standard ops (no bitsandbytes).

Produces a CoreML-EP-compatible ONNX model by re-exporting the original
weights through torch.onnx.export with static sequence length.

Usage:
    uv run scripts/export_onnx.py                    # FP32 export
    uv run scripts/export_onnx.py --fp16             # FP16 export (~half the size)
    uv run scripts/export_onnx.py --seq-len 256      # shorter sequence
    uv run scripts/export_onnx.py --output-dir models/my-model  # custom output
    uv run scripts/export_onnx.py --help
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnx
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-id",
        default="Octen/Octen-Embedding-0.6B",
        help="HuggingFace model ID or local path (default: Octen/Octen-Embedding-0.6B)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory (default: HF_HOME or ~/.cache/huggingface)",
    )
    p.add_argument(
        "--output-dir",
        default="models/Octen-0.6B",
        help="Base model directory; files are written to <output-dir>/{fp16,fp32}/ (default: models/Octen-0.6B)",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Static sequence length to bake into the model (default: 512)",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Export in float16 (default: float32)",
    )
    return p.parse_args()


class EmbeddingWrapper(torch.nn.Module):
    """Wraps the sentence-transformers pipeline into a single module.

    Forward pass: input_ids, attention_mask -> normalized embeddings (batch, hidden_dim).
    """

    def __init__(self, model):
        super().__init__()
        self.transformer = model[0].auto_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask

        # L2 normalize
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    cache_dir = args.cache_dir
    if cache_dir is None:
        for candidate in [
            "/Volumes/ModelStore/cache/huggingface",
            os.path.expanduser("~/.cache/huggingface"),
        ]:
            if Path(candidate).exists():
                cache_dir = candidate
                break

    dtype_label = "fp16" if args.fp16 else "fp32"
    variant_dir = out_dir / dtype_label
    variant_dir.mkdir(parents=True, exist_ok=True)
    onnx_name = f"model.{dtype_label}.onnx"
    data_name = f"model.{dtype_label}.onnx.data"
    onnx_path = variant_dir / onnx_name
    data_path = variant_dir / data_name

    print(f"Model:     {args.model_id}")
    print(f"Cache:     {cache_dir}")
    print(f"Seq len:   {args.seq_len}")
    print(f"Dtype:     {dtype_label}")
    print(f"Output:    {onnx_path}")
    print()

    # ── Load model ────────────────────────────────────────────────────────

    print("Loading model...")
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(
        args.model_id,
        cache_folder=cache_dir,
        device="cpu",
    )

    wrapper = EmbeddingWrapper(st_model)
    if args.fp16:
        wrapper.half()
    else:
        wrapper.float()
    wrapper.eval()

    batch_size = 1
    seq_len = args.seq_len
    dummy_input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    print("Verifying forward pass...")
    with torch.no_grad():
        test_out = wrapper(dummy_input_ids, dummy_attention_mask)
    # Keep reference in float32 for comparison
    test_out_f32 = test_out.float()
    print(f"  Output shape: {test_out.shape}, dtype: {test_out.dtype}")
    print(f"  L2 norm: {test_out_f32.norm(dim=1).item():.6f}")
    print()

    # ── Export to ONNX ────────────────────────────────────────────────────

    print("Exporting to ONNX (opset 18)...")
    # Remove stale external data file before export
    if data_path.exists():
        data_path.unlink()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            opset_version=18,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "embeddings": {0: "batch"},
            },
        )

    # Reload and save with external data so the .onnx file stays small
    print("Saving with external data...")
    model = onnx.load(str(onnx_path))
    # Remove any leftover external data file from the save
    if data_path.exists():
        data_path.unlink()
    onnx.save(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
    )

    graph_size = onnx_path.stat().st_size / (1024 * 1024)
    data_size = data_path.stat().st_size / (1024 * 1024)
    print(f"  {onnx_name}: {graph_size:.1f} MB (graph)")
    print(f"  {data_name}: {data_size:.1f} MB (weights)")
    print()

    # ── Validate ──────────────────────────────────────────────────────────

    print("Validating...")
    model_meta = onnx.load(str(onnx_path), load_external_data=False)
    op_types = set(n.op_type for n in model_meta.graph.node)
    problematic = {"MatMulBnb4", "MatMulNBits"}
    found = op_types & problematic
    if found:
        print(f"  WARNING: non-standard ops found: {found}")
    else:
        print(f"  OK: no non-standard ops ({len(model_meta.graph.node)} nodes)")

    inputs = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in model_meta.graph.input]
    outputs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in model_meta.graph.output]
    print(f"  Inputs:  {inputs}")
    print(f"  Outputs: {outputs}")
    print()

    # ── Verify with ONNX Runtime ──────────────────────────────────────────

    print("Verifying with ONNX Runtime...")
    import onnxruntime as ort

    feed = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    }

    sess = ort.InferenceSession(str(onnx_path))
    result = sess.run(None, feed)
    embedding = result[0].astype(np.float32)
    print(f"  Output shape: {result[0].shape}, dtype: {result[0].dtype}")
    norm = np.linalg.norm(embedding, axis=1)[0]
    print(f"  L2 norm: {norm:.6f}")
    max_diff = np.max(np.abs(embedding - test_out_f32.numpy()))
    print(f"  Max diff vs PyTorch: {max_diff:.2e}")
    print()

    # ── Copy tokenizer ────────────────────────────────────────────────────

    tokenizer_dst = out_dir / "tokenizer.json"
    if not tokenizer_dst.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, cache_dir=cache_dir
        )
        tokenizer.save_pretrained(str(out_dir))
        # Keep only tokenizer.json, remove other tokenizer artifacts
        for f in out_dir.iterdir():
            if f.is_file() and f.name != "tokenizer.json":
                f.unlink(missing_ok=True)
        print(f"  Saved tokenizer.json to {out_dir}")
    else:
        print(f"  tokenizer.json already exists in {out_dir}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
