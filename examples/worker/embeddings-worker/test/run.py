#!/usr/bin/env python3
"""Test the deployed embeddings worker with the pre-canned payloads."""

import json
import math
import subprocess
import sys

URL = sys.argv[1] if len(sys.argv) > 1 else "https://embeddings-worker.seemueller.workers.dev"
ENDPOINT = f"{URL}/v1/embeddings"


def call(payload_path):
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", ENDPOINT, "-H", "Content-Type: application/json", "-d", f"@{payload_path}"],
        capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def histogram(values, bins=10):
    """Return a simple text histogram."""
    lo, hi = min(values), max(values)
    step = (hi - lo) / bins
    if step == 0:
        return f"  all values = {lo:.6f}"
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        counts[idx] += 1
    max_count = max(counts)
    lines = []
    for i, c in enumerate(counts):
        edge = lo + i * step
        bar = "#" * int(30 * c / max_count) if max_count else ""
        lines.append(f"      {edge:+.4f} | {bar} ({c})")
    return "\n".join(lines)


def inspect_embedding(vec, label=""):
    """Print detailed statistics about an embedding vector."""
    dim = len(vec)
    nonzero = [v for v in vec if v != 0.0]
    nz_count = len(nonzero)
    sparsity = 1.0 - nz_count / dim

    print(f"  {label}")
    print(f"    dim={dim}  non-zero={nz_count}/{dim}  sparsity={sparsity:.1%}")

    if nonzero:
        l2 = math.sqrt(sum(v * v for v in vec))
        abs_vals = [abs(v) for v in nonzero]
        std = math.sqrt(sum((v - sum(vec)/dim)**2 for v in vec) / dim)
        print(f"    L2 norm={l2:.4f}  std={std:.6f}")
        print(f"    min={min(vec):.6f}  max={max(vec):.6f}  mean={sum(vec)/dim:.6f}")

        pos = sum(1 for v in nonzero if v > 0)
        neg = nz_count - pos
        print(f"    positive={pos}  negative={neg}")

        # First 8 values as a preview
        preview = "  ".join(f"{v:+.4f}" for v in vec[:8])
        print(f"    first 8: {preview} ...")

        # Value distribution
        print(f"    distribution:")
        print(histogram(vec))
    else:
        print("    (all zeros)")
    print()


# --- single.json ---
print("=" * 60)
print("single.json")
print("=" * 60)
r = call("test/single.json")
print(f"  model={r['model']}  tokens={r['usage']['total_tokens']}")
print()
inspect_embedding(r["data"][0]["embedding"], label='"Cloudflare Workers are awesome"')

# --- batch.json ---
print("=" * 60)
print("batch.json")
print("=" * 60)
r = call("test/batch.json")
texts_batch = ["hello world", "rust is fast", "cloudflare edge computing"]
print(f"  model={r['model']}  tokens={r['usage']['total_tokens']}  embeddings={len(r['data'])}")
print()
for d, text in zip(r["data"], texts_batch):
    inspect_embedding(d["embedding"], label=f'[{d["index"]}] "{text}"')

# --- semantic_similarity.json ---
print("=" * 60)
print("semantic_similarity.json")
print("=" * 60)
r = call("test/semantic_similarity.json")
vecs = [d["embedding"] for d in r["data"]]
texts = [
    "the cat sat on the mat",
    "a feline rested on the rug",
    "stock prices rose sharply today",
    "the dog chased the ball",
]
print(f"  model={r['model']}  tokens={r['usage']['total_tokens']}  embeddings={len(vecs)}")
print()

for d, text in zip(r["data"], texts):
    inspect_embedding(d["embedding"], label=f'[{d["index"]}] "{text}"')

print("  Cosine similarity matrix:")
print(f"          {'':>4}" + "".join(f"  [{i}]   " for i in range(len(vecs))))
for i in range(len(vecs)):
    row = ""
    for j in range(len(vecs)):
        sim = cosine(vecs[i], vecs[j])
        row += f"  {sim:+.4f}"
    print(f"    [{i}]  {row}")
print()

# Pairwise L2 distances (more informative than shared dims for dense vectors)
print("  Pairwise L2 distances:")
short = ["cat/mat", "feline/rug", "stocks", "dog/ball"]
for i in range(len(vecs)):
    for j in range(i + 1, len(vecs)):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j])))
        print(f"    {short[i]:>11} <-> {short[j]:<11}  L2={dist:.4f}")
