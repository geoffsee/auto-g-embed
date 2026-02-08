#!/usr/bin/env python3
"""Run a reproducible embedding evaluation with MTEB + latency/throughput profiling."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mteb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


WORD_POOL = [
    "semantic",
    "retrieval",
    "language",
    "model",
    "document",
    "query",
    "embedding",
    "vector",
    "ranking",
    "context",
    "search",
    "evidence",
    "finance",
    "science",
    "benchmark",
    "accuracy",
    "latency",
    "throughput",
    "corpus",
    "relevance",
    "passage",
    "index",
    "token",
    "encoder",
    "memory",
    "batch",
    "inference",
    "quality",
    "robust",
    "generalization",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="artifacts/model-kaggle/sentence-transformer",
        help="SentenceTransformer model path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device.",
    )
    parser.add_argument(
        "--tasks",
        default="SciFact,NFCorpus,FiQA2018",
        help="Comma-separated MTEB task names.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for MTEB and performance profiling.",
    )
    parser.add_argument(
        "--perf-seq-lens",
        default="32,128,512",
        help="Comma-separated target sequence lengths (approx token count) for perf profiling.",
    )
    parser.add_argument(
        "--perf-samples",
        type=int,
        default=2048,
        help="Number of texts per sequence length for perf profiling.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=8,
        help="Warmup batches before each timed repeat.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeats for confidence interval estimation.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=3000,
        help="Bootstrap sample count for median confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip-mteb",
        action="store_true",
        help="Skip MTEB task execution and only run perf profiling.",
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="Skip perf profiling and only run MTEB tasks.",
    )
    parser.add_argument(
        "--mteb-output-dir",
        default="artifacts/evals/mteb_real",
        help="Directory where MTEB writes task JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/evals/real_eval/latest.json",
        help="Path for consolidated evaluation JSON.",
    )
    return parser.parse_args()


def parse_csv_ints(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def parse_csv_strings(text: str) -> list[str]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("expected at least one task name")
    return values


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        value = output.strip()
        return value or None
    except Exception:
        return None


def approx_texts_for_seq_len(seq_len: int, count: int, seed: int) -> tuple[list[str], list[int]]:
    rng = random.Random(seed + seq_len * 101)
    texts: list[str] = []
    observed_token_lengths: list[int] = []

    # We approximate one word per token using common words to keep generation fast.
    for _ in range(count):
        words = [rng.choice(WORD_POOL) for _ in range(seq_len)]
        text = " ".join(words)
        texts.append(text)
        observed_token_lengths.append(len(words))

    return texts, observed_token_lengths


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def bootstrap_median_ci(
    values: list[float],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        value = float(arr[0])
        return (value, value)

    alpha = 1.0 - confidence
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(samples, arr.size))
    medians = np.median(arr[indices], axis=1)
    low = float(np.percentile(medians, 100.0 * alpha / 2.0))
    high = float(np.percentile(medians, 100.0 * (1.0 - alpha / 2.0)))
    return (low, high)


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_perf_profile(
    model: SentenceTransformer,
    *,
    device: str,
    seq_lens: list[int],
    perf_samples: int,
    warmup_batches: int,
    repeats: int,
    batch_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    profile: list[dict[str, Any]] = []

    for seq_len in seq_lens:
        texts, observed_token_lengths = approx_texts_for_seq_len(seq_len, perf_samples, seed)
        batches = chunked(texts, batch_size)
        warmup_batches = min(warmup_batches, len(batches))

        repeats_out: list[dict[str, float]] = []
        for repeat_idx in range(repeats):
            # Warmup
            for warm_batch in batches[:warmup_batches]:
                _ = model.encode(
                    warm_batch,
                    batch_size=len(warm_batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            sync_device(device)

            per_item_us: list[float] = []
            total_count = 0
            start_total = time.perf_counter()

            for batch in batches:
                sync_device(device)
                t0 = time.perf_counter()
                _ = model.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                sync_device(device)
                elapsed = time.perf_counter() - t0
                item_us = (elapsed / len(batch)) * 1_000_000.0
                per_item_us.extend([item_us] * len(batch))
                total_count += len(batch)

            sync_device(device)
            total_elapsed = time.perf_counter() - start_total
            embeds_per_second = total_count / max(total_elapsed, 1e-12)

            repeats_out.append(
                {
                    "repeat_index": float(repeat_idx + 1),
                    "embeds_per_second": float(embeds_per_second),
                    "p50_us": percentile(per_item_us, 50),
                    "p95_us": percentile(per_item_us, 95),
                    "p99_us": percentile(per_item_us, 99),
                    "mean_us": float(np.mean(np.asarray(per_item_us, dtype=np.float64))),
                    "max_us": float(np.max(np.asarray(per_item_us, dtype=np.float64))),
                }
            )

        def summarize_metric(name: str, metric_seed_offset: int) -> dict[str, Any]:
            values = [float(row[name]) for row in repeats_out]
            median = float(np.median(np.asarray(values, dtype=np.float64)))
            low, high = bootstrap_median_ci(
                values,
                confidence=0.95,
                samples=bootstrap_samples,
                seed=seed + metric_seed_offset,
            )
            return {
                "median": median,
                "ci95_low": low,
                "ci95_high": high,
                "all_repeats": values,
            }

        profile.append(
            {
                "target_seq_len": seq_len,
                "observed_token_count_mean": float(
                    np.mean(np.asarray(observed_token_lengths, dtype=np.float64))
                ),
                "observed_token_count_min": int(min(observed_token_lengths)),
                "observed_token_count_max": int(max(observed_token_lengths)),
                "sample_count": perf_samples,
                "batch_size": batch_size,
                "repeat_count": repeats,
                "warmup_batches": warmup_batches,
                "metrics": {
                    "embeds_per_second": summarize_metric("embeds_per_second", 11),
                    "p50_us": summarize_metric("p50_us", 13),
                    "p95_us": summarize_metric("p95_us", 17),
                    "p99_us": summarize_metric("p99_us", 19),
                },
                "repeat_rows": repeats_out,
            }
        )

    return {"by_sequence_length": profile}


def run_mteb_eval(
    model: SentenceTransformer,
    *,
    tasks: list[str],
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_objects = mteb.get_tasks(tasks=tasks)
    benchmark = mteb.MTEB(tasks=task_objects)
    benchmark.run(
        model,
        output_folder=str(output_dir),
        eval_splits=["test"],
        overwrite_results=True,
        raise_error=True,
        encode_kwargs={"batch_size": batch_size, "show_progress_bar": True},
    )

    parsed: list[dict[str, Any]] = []
    for task_name in tasks:
        matches = list(output_dir.rglob(f"{task_name}.json"))
        if not matches:
            parsed.append({"task_name": task_name, "error": "result_json_not_found"})
            continue

        task_path = matches[0]
        payload = json.loads(task_path.read_text())
        test_scores = payload.get("scores", {}).get("test", [])
        row = test_scores[0] if test_scores else {}
        parsed.append(
            {
                "task_name": payload.get("task_name", task_name),
                "result_file": str(task_path),
                "dataset_revision": payload.get("dataset_revision"),
                "main_score": row.get("main_score"),
                "ndcg_at_10": row.get("ndcg_at_10"),
                "map_at_10": row.get("map_at_10"),
                "recall_at_10": row.get("recall_at_10"),
                "mrr_at_10": row.get("mrr_at_10"),
                "evaluation_time_s": payload.get("evaluation_time"),
            }
        )

    return {"tasks": parsed}


def main() -> int:
    args = parse_args()
    seq_lens = parse_csv_ints(args.perf_seq_lens)
    task_names = parse_csv_strings(args.tasks)
    device = detect_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = SentenceTransformer(args.model, device=device)

    payload: dict[str, Any] = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "model": args.model,
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "device": device,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mteb": mteb.__version__,
            "git_commit": git_commit(),
            "seed": args.seed,
        },
        "config": {
            "tasks": task_names,
            "batch_size": args.batch_size,
            "perf_seq_lens": seq_lens,
            "perf_samples": args.perf_samples,
            "warmup_batches": args.warmup_batches,
            "repeats": args.repeats,
            "bootstrap_samples": args.bootstrap_samples,
            "mteb_output_dir": args.mteb_output_dir,
            "skip_mteb": args.skip_mteb,
            "skip_perf": args.skip_perf,
        },
    }

    if not args.skip_mteb:
        payload["mteb"] = run_mteb_eval(
            model,
            tasks=task_names,
            output_dir=Path(args.mteb_output_dir),
            batch_size=args.batch_size,
        )

    if not args.skip_perf:
        payload["perf"] = run_perf_profile(
            model,
            device=device,
            seq_lens=seq_lens,
            perf_samples=args.perf_samples,
            warmup_batches=args.warmup_batches,
            repeats=args.repeats,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"Wrote real eval artifact: {output_path}")
    if "mteb" in payload:
        for row in payload["mteb"]["tasks"]:
            print(
                f"MTEB {row.get('task_name')}: "
                f"main={row.get('main_score')} ndcg@10={row.get('ndcg_at_10')}"
            )
    if "perf" in payload:
        for row in payload["perf"]["by_sequence_length"]:
            e = row["metrics"]["embeds_per_second"]["median"]
            p50 = row["metrics"]["p50_us"]["median"]
            p95 = row["metrics"]["p95_us"]["median"]
            p99 = row["metrics"]["p99_us"]["median"]
            print(
                "Perf seq_len="
                f"{row['target_seq_len']}: embeds/s={e:.2f} "
                f"p50_us={p50:.2f} p95_us={p95:.2f} p99_us={p99:.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
