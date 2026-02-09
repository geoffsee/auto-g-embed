# embeddings-worker

A Cloudflare Worker that serves an OpenAI-compatible `/v1/embeddings` endpoint, powered by `auto-g-embed`'s `RustContrastiveEmbedder` compiled to WASM.

Model weights (~16 MB) are loaded from Cloudflare KV at request time.

## API

### `POST /v1/embeddings`

Accepts the same request format as the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings).

```bash
curl -X POST https://embeddings-worker.seemueller.workers.dev/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "hello world", "model": "auto-g-embed"}'
```

**Request body**

| Field             | Type                 | Required | Description                  |
|-------------------|----------------------|----------|------------------------------|
| `input`           | `string \| string[]` | yes      | Text(s) to embed             |
| `model`           | `string`             | yes      | Model name (echoed back)     |
| `encoding_format` | `string`             | no       | Currently ignored            |

**Response**

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "embedding": [0.012, -0.034, ...], "index": 0 }
  ],
  "model": "auto-g-embed",
  "usage": { "prompt_tokens": 2, "total_tokens": 2 }
}
```

CORS is enabled (`Access-Control-Allow-Origin: *`).

## Setup

### Prerequisites

- [wrangler](https://developers.cloudflare.com/workers/wrangler/) CLI
- Rust with the `wasm32-unknown-unknown` target (`rustup target add wasm32-unknown-unknown`)

### Export and upload model weights

1. Train a model (see the main `auto-g-embed` README).

2. Export to the compact bytes format:

   ```bash
   cargo run --example export_weights -- <model-dir> model-weights.bin
   ```

3. Create a KV namespace and upload:

   ```bash
   npx wrangler kv namespace create EMBEDDINGS_MODEL
   # Update the namespace id in wrangler.toml, then:
   npx wrangler kv key put --binding EMBEDDINGS_MODEL "model-weights" --path model-weights.bin --remote
   ```

### Deploy

```bash
npx wrangler deploy
```

### Local development

```bash
npx wrangler dev
```

Note: local dev requires the KV key to be uploaded locally as well (omit `--remote` above).

## Testing

Pre-canned payloads live in `test/`. Run the test script against a deployed or local instance:

```bash
python3 test/run.py                           # defaults to production URL
python3 test/run.py http://localhost:8787      # local dev
```

The script prints embedding statistics, cosine similarity matrices, and pairwise L2 distances.
