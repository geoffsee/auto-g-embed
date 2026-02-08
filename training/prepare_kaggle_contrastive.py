#!/usr/bin/env python3
"""Build contrastive train/eval data from profile-driven dataset adapters."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

try:
    import kagglehub
except ImportError:  # pragma: no cover
    kagglehub = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES_FILE = SCRIPT_DIR / "dataset_profiles.json"
DEFAULT_PROFILE_NAME = "quora_pairs"


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    kind: str
    dataset_ref: str | None
    csv_name: str | None
    columns: dict[str, str]
    filters: dict[str, Any]
    options: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles-file",
        type=Path,
        default=DEFAULT_PROFILES_FILE,
        help=f"JSON file with dataset adapter profiles (default: {DEFAULT_PROFILES_FILE}).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        help=f"Dataset profile name (default: {DEFAULT_PROFILE_NAME}).",
    )
    parser.add_argument(
        "--dataset-ref",
        default=None,
        help="Override Kaggle dataset ref from profile.",
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="Optional local CSV path. If set, skips Kaggle download.",
    )
    parser.add_argument(
        "--csv-name",
        default=None,
        help="Override profile CSV filename inside downloaded dataset dir.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=None,
        help="Override minimum character length filter.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Override minimum word-count filter.",
    )
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=300_000,
        help="Maximum train pairs to keep.",
    )
    parser.add_argument(
        "--max-eval-triplets",
        type=int,
        default=10_000,
        help="Maximum eval triplets to keep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/contrastive-data"),
        help="Output directory for parquet + metadata.",
    )
    return parser.parse_args()


def load_profiles(path: Path) -> dict[str, DatasetProfile]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "profiles" not in raw or not isinstance(raw["profiles"], dict):
        raise ValueError(f"Invalid profiles JSON at {path}: missing `profiles` object")

    profiles: dict[str, DatasetProfile] = {}
    for name, cfg in raw["profiles"].items():
        profiles[name] = DatasetProfile(
            name=name,
            kind=str(cfg["kind"]),
            dataset_ref=cfg.get("dataset_ref"),
            csv_name=cfg.get("csv_name"),
            columns=dict(cfg.get("columns", {})),
            filters=dict(cfg.get("filters", {})),
            options=dict(cfg.get("options", {})),
        )
    return profiles


def clean_text_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .str.replace_all(r"<[^>]+>", " ")
        .str.replace_all(r"https?://\S+", " ")
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def sample_if_needed(frame: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    if n <= 0 or frame.height <= n:
        return frame
    return frame.sample(n=n, with_replacement=False, shuffle=True, seed=seed)


def dataset_root_from_kaggle(dataset_ref: str) -> Path:
    if kagglehub is None:
        raise RuntimeError(
            "kagglehub is not installed. Install deps: pip install -r training/requirements.txt"
        )
    return Path(kagglehub.dataset_download(dataset_ref))


def find_csv(dataset_root: Path, csv_name: str | None = None) -> Path:
    if csv_name:
        matches = sorted(dataset_root.rglob(csv_name))
        if matches:
            return matches[0]
    csv_candidates = sorted(dataset_root.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")
    return csv_candidates[0]


def random_negative_column(candidates: list[str], n: int, seed: int) -> list[str]:
    if not candidates:
        raise ValueError("Negative candidate pool is empty.")
    rng = random.Random(seed)
    return [candidates[rng.randrange(len(candidates))] for _ in range(n)]


def apply_common_text_filters(
    frame: pl.DataFrame,
    text_col: str,
    min_chars: int,
    min_words: int,
    require_question_mark: bool,
    disallow_values: list[str],
) -> pl.DataFrame:
    disallow_set = {v.strip().lower() for v in disallow_values if v.strip()}
    if disallow_set:
        frame = frame.filter(~pl.col(text_col).is_in(sorted(disallow_set)))

    frame = frame.filter(
        (pl.col(text_col).str.len_chars() >= min_chars)
        & (pl.col(text_col).str.count_matches(r"\S+").fill_null(0) >= min_words)
    )

    if require_question_mark:
        frame = frame.filter(pl.col("raw_text").str.ends_with("?"))

    return frame


def build_from_pair_binary_label(
    frame: pl.DataFrame,
    profile: DatasetProfile,
    min_chars: int,
    min_words: int,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    left_col = profile.columns.get("left")
    right_col = profile.columns.get("right")
    label_col = profile.columns.get("label")
    if not left_col or not right_col or not label_col:
        raise ValueError(
            f"profile '{profile.name}' requires columns.left/columns.right/columns.label"
        )

    positive_label = int(profile.options.get("positive_label", 1))
    disallow_values = list(profile.filters.get("disallow_values", []))

    pairs = (
        frame.lazy()
        .select(
            [
                clean_text_expr(left_col).alias("left_text"),
                clean_text_expr(right_col).alias("right_text"),
                pl.col(label_col).cast(pl.Int32, strict=False).alias("label"),
            ]
        )
        .filter(
            pl.col("left_text").is_not_null()
            & pl.col("right_text").is_not_null()
            & (pl.col("left_text") != "")
            & (pl.col("right_text") != "")
        )
        .collect(engine="streaming")
    )
    rows_after_cleaning = pairs.height

    pairs = apply_common_text_filters(
        frame=pairs.rename({"left_text": "text_a", "right_text": "text_b"}),
        text_col="text_a",
        min_chars=min_chars,
        min_words=min_words,
        require_question_mark=False,
        disallow_values=disallow_values,
    )
    pairs = apply_common_text_filters(
        frame=pairs.rename({"text_a": "text_a", "text_b": "text_b"}),
        text_col="text_b",
        min_chars=min_chars,
        min_words=min_words,
        require_question_mark=False,
        disallow_values=disallow_values,
    )
    pairs = pairs.rename({"text_a": "left_text", "text_b": "right_text"})

    positives = (
        pairs.filter(pl.col("label") == positive_label)
        .select(
            [
                pl.when(pl.col("left_text") <= pl.col("right_text"))
                .then(pl.col("left_text"))
                .otherwise(pl.col("right_text"))
                .alias("anchor"),
                pl.when(pl.col("left_text") <= pl.col("right_text"))
                .then(pl.col("right_text"))
                .otherwise(pl.col("left_text"))
                .alias("positive"),
            ]
        )
        .unique(subset=["anchor", "positive"], maintain_order=True)
    )

    negatives = (
        pairs.filter(pl.col("label") != positive_label)
        .select(
            [
                pl.col("left_text").alias("anchor"),
                pl.col("right_text").alias("negative"),
            ]
        )
        .filter(pl.col("anchor") != pl.col("negative"))
        .unique(subset=["anchor", "negative"], maintain_order=True)
    )

    return positives, negatives, rows_after_cleaning


def build_from_single_text_duplicate_mining(
    frame: pl.DataFrame,
    profile: DatasetProfile,
    min_chars: int,
    min_words: int,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    text_col = profile.columns.get("text")
    if not text_col:
        raise ValueError(f"profile '{profile.name}' requires columns.text")

    require_question_mark = bool(profile.filters.get("require_question_mark", False))
    disallow_values = list(profile.filters.get("disallow_values", []))

    texts = (
        frame.lazy()
        .select(
            [
                pl.col(text_col).cast(pl.Utf8, strict=False).alias("raw_text"),
                clean_text_expr(text_col).alias("clean_text"),
            ]
        )
        .filter(pl.col("clean_text").is_not_null() & (pl.col("clean_text") != ""))
        .collect(engine="streaming")
    )

    rows_after_cleaning = texts.height
    texts = apply_common_text_filters(
        frame=texts,
        text_col="clean_text",
        min_chars=min_chars,
        min_words=min_words,
        require_question_mark=require_question_mark,
        disallow_values=disallow_values,
    )

    texts = texts.with_columns(pl.col("clean_text").alias("norm_key"))

    positives = (
        texts.sort(["norm_key", "clean_text"])
        .with_columns(
            [
                pl.col("clean_text").shift(-1).over("norm_key").alias("positive"),
                pl.len().over("norm_key").alias("group_size"),
            ]
        )
        .filter(pl.col("group_size") > 1)
        .select([pl.col("clean_text").alias("anchor"), pl.col("positive")])
        .filter(pl.col("positive").is_not_null() & (pl.col("anchor") != ""))
        .unique(subset=["anchor", "positive"], maintain_order=True)
    )

    negatives = texts.select(pl.col("clean_text").alias("negative")).unique(
        subset=["negative"], maintain_order=True
    )

    return positives, negatives, rows_after_cleaning


def make_eval_triplets(
    positives: pl.DataFrame,
    negatives: pl.DataFrame,
    max_eval_triplets: int,
    seed: int,
) -> pl.DataFrame:
    if positives.is_empty():
        raise RuntimeError("No positive pairs produced; cannot build contrastive data.")
    if negatives.is_empty():
        raise RuntimeError("No negatives produced; cannot build eval triplets.")

    eval_anchor_pos = sample_if_needed(positives, max_eval_triplets, seed + 7)
    random_negatives = random_negative_column(
        negatives["negative"].to_list(),
        eval_anchor_pos.height,
        seed + 11,
    )
    eval_triplets = pl.DataFrame(
        {
            "anchor": eval_anchor_pos["anchor"],
            "positive": eval_anchor_pos["positive"],
            "negative": random_negatives,
        }
    ).filter(
        (pl.col("anchor") != pl.col("negative"))
        & (pl.col("positive") != pl.col("negative"))
    )
    return eval_triplets


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.profiles_file)

    if args.list_profiles:
        for name in sorted(profiles):
            profile = profiles[name]
            print(f"{name}\t{profile.kind}\t{profile.dataset_ref or '-'}")
        return

    if args.profile not in profiles:
        raise KeyError(
            f"Unknown profile '{args.profile}'. "
            f"Use --list-profiles to inspect {args.profiles_file}."
        )

    profile = profiles[args.profile]
    dataset_ref = args.dataset_ref or profile.dataset_ref
    csv_name = args.csv_name or profile.csv_name

    min_chars = int(args.min_chars or profile.filters.get("min_chars", 12))
    min_words = int(args.min_words or profile.filters.get("min_words", 3))

    if args.source_csv is not None:
        csv_path = args.source_csv
    else:
        if not dataset_ref:
            raise ValueError(
                f"profile '{profile.name}' has no dataset_ref; provide --source-csv"
            )
        dataset_root = dataset_root_from_kaggle(dataset_ref)
        csv_path = find_csv(dataset_root, csv_name)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(
        f"[prepare] profile={profile.name} kind={profile.kind} source_csv={csv_path}"
    )
    print(f"[prepare] min_chars={min_chars} min_words={min_words}")

    source_frame = pl.read_csv(csv_path, infer_schema_length=20_000, ignore_errors=True)

    if profile.kind == "pair_binary_label":
        positives, negatives, rows_after_cleaning = build_from_pair_binary_label(
            frame=source_frame,
            profile=profile,
            min_chars=min_chars,
            min_words=min_words,
        )
    elif profile.kind == "single_text_duplicate_mining":
        positives, negatives, rows_after_cleaning = build_from_single_text_duplicate_mining(
            frame=source_frame,
            profile=profile,
            min_chars=min_chars,
            min_words=min_words,
        )
    else:
        raise ValueError(
            f"Unsupported profile kind '{profile.kind}' in profile '{profile.name}'"
        )

    train_pairs = sample_if_needed(positives, args.max_train_pairs, args.seed)
    negative_pool = negatives.select("negative").unique(subset=["negative"], maintain_order=True)
    eval_triplets = make_eval_triplets(
        positives=train_pairs,
        negatives=negative_pool,
        max_eval_triplets=args.max_eval_triplets,
        seed=args.seed,
    )
    eval_triplets = sample_if_needed(eval_triplets, args.max_eval_triplets, args.seed + 19)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_pairs.parquet"
    eval_path = args.out_dir / "eval_triplets.parquet"
    stats_path = args.out_dir / "metadata.json"

    train_pairs.write_parquet(train_path)
    eval_triplets.write_parquet(eval_path)

    stats = {
        "profile": profile.name,
        "profile_kind": profile.kind,
        "dataset_ref": dataset_ref if args.source_csv is None else "local_csv",
        "source_csv": str(csv_path),
        "rows_after_cleaning": rows_after_cleaning,
        "positive_pairs": positives.height,
        "negative_pool_size": negative_pool.height,
        "train_pairs": train_pairs.height,
        "eval_triplets": eval_triplets.height,
        "max_train_pairs": args.max_train_pairs,
        "max_eval_triplets": args.max_eval_triplets,
        "min_chars": min_chars,
        "min_words": min_words,
        "seed": args.seed,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[prepare] wrote {train_path}")
    print(f"[prepare] wrote {eval_path}")
    print(f"[prepare] wrote {stats_path}")
    print(
        f"[prepare] train_pairs={train_pairs.height} eval_triplets={eval_triplets.height}"
    )


if __name__ == "__main__":
    main()
