# Community Benchmark Guide

This repo includes a reproducible benchmark command for local embedding inference and retrieval quality.

## What to publish

For each reported run, publish:

- commit SHA
- machine details (CPU, OS, logical CPU count)
- Rust version
- exact benchmark command
- output JSON artifact
- median of 3 release runs (not a single run)

## Benchmark command

```bash
cargo run --release --bin community_benchmark -- \
  --output artifacts/benchmarks/latest.json
```

Optional knobs:

```bash
--eval-count 10000 --warmup-count 1000 --query-count 64 --window-size 80 --stride 40 --max-passages 180
```

## Run protocol

Use this protocol for numbers intended for README, release notes, or external comparison:

1. Use a clean working tree and record the commit hash.
2. Run on AC power with minimal background workload.
3. Build and run in release mode only.
4. Execute the benchmark 3 times with identical arguments.
5. Report the median for throughput and latency percentiles.
6. Keep quality metrics from the same run configuration.

## Metrics definition

- Throughput: total timed embeddings divided by wall-clock timed section.
- Latency: per-request embedding latency (`p50`, `p95`, `p99`, `mean`, `min`, `max`) in microseconds.
- Quality:
  - `top1_accuracy`: query retrieves its source passage at rank 1.
  - `separation`: `mean_positive_similarity - mean_negative_similarity`.

## JSON schema

`community_benchmark` writes JSON with top-level fields:

- `benchmark`: benchmark config and command metadata
- `environment`: host/runtime metadata
- `dataset`: corpus-derived stats
- `throughput`: aggregate throughput metrics
- `latency`: percentile and summary latency metrics
- `quality`: retrieval quality metrics

## Suggested report table

| commit | machine | rustc | embeds/s | p50 us | p95 us | p99 us | top1 | separation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `<sha>` | `<cpu/os>` | `<version>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |
