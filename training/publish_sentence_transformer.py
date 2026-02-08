#!/usr/bin/env python3
"""Publish a SentenceTransformer artifact directory to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentence-model-dir",
        type=Path,
        default=Path("artifacts/model/sentence-transformer"),
        help="Local SentenceTransformer directory to publish.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face model repo id (for example: username/model-name).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (defaults to HF_TOKEN env var or CLI auth).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private when it does not already exist.",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Upload as a pull request instead of committing directly.",
    )
    parser.add_argument(
        "--replace-root",
        action="store_true",
        help="Replace repo root contents by deleting existing files before upload.",
    )
    parser.add_argument(
        "--commit-message",
        default="Publish sentence-transformer artifact",
        help="Commit message used for upload.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("artifacts/model/metrics.json"),
        help="Optional metrics JSON to include in published files if it exists.",
    )
    parser.add_argument(
        "--real-eval-json",
        type=Path,
        default=Path("artifacts/evals/real_eval/latest.json"),
        help="Optional consolidated eval JSON to include if it exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned publish actions without uploading.",
    )
    return parser.parse_args()


def ensure_sentence_transformer_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing sentence-transformer directory: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    required = ["modules.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Directory does not look like a SentenceTransformer package. "
            f"Missing files: {', '.join(missing)} in {path}"
        )


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        value = out.strip()
        return value or None
    except Exception:
        return None


def build_minimal_readme(repo_id: str) -> str:
    model_name = repo_id.split("/")[-1]
    return f"""---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - text-embeddings
  - feature-extraction
license: mit
---

# {model_name}

Sentence embedding model published from the `auto-g-embed` training pipeline.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{repo_id}")
embeddings = model.encode(["example query", "example document"])
```
"""


def main() -> None:
    args = parse_args()
    ensure_sentence_transformer_dir(args.sentence_model_dir)

    with tempfile.TemporaryDirectory(prefix="hf_publish_stage_") as tmp_dir:
        stage_dir = Path(tmp_dir) / "model"
        shutil.copytree(args.sentence_model_dir, stage_dir)

        # Attach optional metadata artifacts if present.
        if args.metrics_json.exists():
            shutil.copy2(args.metrics_json, stage_dir / "metrics.json")
        if args.real_eval_json.exists():
            shutil.copy2(args.real_eval_json, stage_dir / "real_eval.json")

        readme_path = stage_dir / "README.md"
        if not readme_path.exists():
            readme_path.write_text(build_minimal_readme(args.repo_id), encoding="utf-8")

        publish_meta = {
            "source_sentence_model_dir": str(args.sentence_model_dir),
            "source_metrics_json": str(args.metrics_json) if args.metrics_json.exists() else None,
            "source_real_eval_json": str(args.real_eval_json) if args.real_eval_json.exists() else None,
            "git_commit": git_commit(),
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (stage_dir / "publish_metadata.json").write_text(
            json.dumps(publish_meta, indent=2),
            encoding="utf-8",
        )

        if args.dry_run:
            print("[publish] dry run")
            print(f"[publish] repo_id={args.repo_id}")
            print(f"[publish] source_dir={args.sentence_model_dir}")
            print(f"[publish] staged_dir={stage_dir}")
            print(f"[publish] replace_root={args.replace_root}")
            print(f"[publish] create_pr={args.create_pr}")
            print(f"[publish] token_provided={bool(args.token)}")
            return

        api = HfApi(token=args.token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=args.token,
        )

        delete_patterns = "*" if args.replace_root else None
        result = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=stage_dir,
            path_in_repo=".",
            commit_message=args.commit_message,
            token=args.token,
            create_pr=args.create_pr,
            delete_patterns=delete_patterns,
        )

        print(f"[publish] uploaded model to: {args.repo_id}")
        print(f"[publish] commit_url: {result.commit_url}")
        if result.pr_url:
            print(f"[publish] pr_url: {result.pr_url}")


if __name__ == "__main__":
    main()
