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
        print(f"    L2 norm={l2:.4f}  min|v|={min(abs_vals):.6f}  max|v|={max(abs_vals):.6f}  mean|v|={sum(abs_vals)/len(abs_vals):.6f}")

        pos = sum(1 for v in nonzero if v > 0)
        neg = nz_count - pos
        print(f"    positive={pos}  negative={neg}")

        # Show non-zero indices and values
        active = [(i, v) for i, v in enumerate(vec) if v != 0.0]
        entries = "  ".join(f"[{i}]={v:+.4f}" for i, v in active[:12])
        suffix = f"  ... ({nz_count - 12} more)" if nz_count > 12 else ""
        print(f"    values: {entries}{suffix}")
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
header = "".join(f"{'['+str(i)+']':>8}" for i in range(len(vecs)))
print(f"          {header}")
for i in range(len(vecs)):
    row = ""
    for j in range(len(vecs)):
        sim = cosine(vecs[i], vecs[j])
        row += f"{sim:+8.4f}"
    print(f"    [{i}]   {row}")
print()

# Check if any pair of different texts share active dimensions
print("  Shared active dimensions:")
for i in range(len(vecs)):
    for j in range(i + 1, len(vecs)):
        active_i = {k for k, v in enumerate(vecs[i]) if v != 0.0}
        active_j = {k for k, v in enumerate(vecs[j]) if v != 0.0}
        shared = active_i & active_j
        if shared:
            details = ", ".join(f"[{k}]" for k in sorted(shared))
            print(f"    [{i}]<->[{j}]  {len(shared)} shared: {details}")
        else:
            print(f"    [{i}]<->[{j}]  none (orthogonal)")
