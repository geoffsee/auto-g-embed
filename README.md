---
license: mit
library_name: custom
pipeline_tag: feature-extraction
tags:
  - sentence-embeddings
  - text-embeddings
  - rust
---

---
language:
- en
- zh
- multilingual
license: apache-2.0
library_name: onnxruntime
tags:
- onnx
- fp16
- embeddings
- sentence-similarity
- feature-extraction
- text-embedding
- retrieval
pipeline_tag: sentence-similarity
base_model: Octen/Octen-Embedding-0.6B
---

# auto-g-embed-Octen-0.6B — ONNX

FP16 ONNX re-export of [Octen/Octen-Embedding-0.6B](https://huggingface.co/Octen/Octen-Embedding-0.6B) using only standard ONNX operators (opset 18). Compatible with any ONNX Runtime backend including CoreML, CUDA, DirectML, and CPU.

## Why this export?

The original ONNX export uses `MatMulBnb4` (bitsandbytes INT4) operators from the `com.microsoft` domain. These are not supported by most execution providers (CoreML, TensorRT, etc.) and limit the model to CPU-only inference in practice.

This re-export replaces all non-standard ops with standard ONNX `MatMul` in float16, enabling hardware-accelerated inference across all major platforms.

## Differences from the original

| | Original (INT4) | This export (FP16) |
|---|---|---|
| **Precision** | 4-bit (bitsandbytes) | float16 |
| **Weight size** | 533 MB | 1.1 GB |
| **ONNX ops** | `com.microsoft.MatMulBnb4` | Standard `MatMul` only |
| **CoreML EP** | Not supported | Supported |
| **CUDA EP** | Limited | Supported |
| **Batch perf** | Baseline | ~11-19x faster |
| **Single perf** | Baseline | ~1.2-1.4x faster |

## Files

- `model.fp16.onnx` — ONNX graph (5.3 MB)
- `model.fp16.onnx.data` — External weights (1.1 GB)
- `tokenizer.json` — HuggingFace BPE tokenizer (11 MB)

## Usage

### Python (ONNX Runtime)

```python
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np

tokenizer = Tokenizer.from_file("tokenizer.json")
session = ort.InferenceSession("model.fp16.onnx")

encoding = tokenizer.encode("hello world", add_special_tokens=True)
input_ids = np.array([encoding.ids], dtype=np.int64)
attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

embeddings = session.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
})[0]

print(f"Shape: {embeddings.shape}")  # (1, 1024)
```

### Rust (ort)

```rust
use int4_runner::EmbeddingModel;

let tok = std::fs::read("tokenizer.json").unwrap();
let model = EmbeddingModel::from_file("model.fp16.onnx", &tok).unwrap();
let embedding = model.embed("hello world").unwrap();
println!("dimensions: {}", embedding.values.len()); // 1024
```

## Model details

- **Architecture**: Qwen3 transformer, 28 layers, 1024 hidden dim, 16 attention heads
- **Max sequence length**: 32,768 tokens (model supports), 512 tokens (typical usage)
- **Output dimension**: 1024
- **Tokenizer**: Qwen BPE (vocab size 151,669)
- **Pooling**: Mean pooling + L2 normalization

## Attribution

- Original model: [Octen/Octen-Embedding-0.6B](https://huggingface.co/Octen/Octen-Embedding-0.6B) by [Octen](https://octen.ai/)
- Base model: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- License: Apache-2.0

## Changes from original

This is a precision-format conversion only. The model weights were cast from bfloat16 to float16 and re-exported using `torch.onnx.export` (opset 18) with mean pooling and L2 normalization baked into the graph. No fine-tuning or weight modification was performed.
