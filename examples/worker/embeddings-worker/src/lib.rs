use auto_g_embed::rust_embedder::RustContrastiveEmbedder;
use serde::{Deserialize, Serialize};
use worker::*;

const MODEL_KV_KEY: &str = "model-weights";

// --- OpenAI-compatible request/response types ---

#[derive(Deserialize)]
struct EmbeddingRequest {
    input: EmbeddingInput,
    model: String,
    #[serde(default)]
    encoding_format: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    fn into_texts(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Batch(v) => v,
        }
    }
}

#[derive(Serialize)]
struct EmbeddingResponse {
    object: &'static str,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[derive(Serialize)]
struct EmbeddingData {
    object: &'static str,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    r#type: &'static str,
}

// --- Helpers ---

fn json_response(status: http::StatusCode, body: &impl Serialize) -> Result<HttpResponse> {
    let json = serde_json::to_vec(body).map_err(|e| Error::RustError(e.to_string()))?;
    let stream = futures_util::stream::once(async move { Ok::<_, worker::Error>(json) });
    Ok(http::Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from_stream(stream)?)?)
}

fn error_response(status: http::StatusCode, message: impl Into<String>) -> Result<HttpResponse> {
    json_response(
        status,
        &ErrorResponse {
            error: ErrorBody {
                message: message.into(),
                r#type: "invalid_request_error",
            },
        },
    )
}

async fn load_embedder(env: &Env) -> Result<RustContrastiveEmbedder> {
    let kv = env.kv("EMBEDDINGS_MODEL")?;
    let bytes = kv
        .get(MODEL_KV_KEY)
        .bytes()
        .await?
        .ok_or_else(|| Error::RustError(format!("KV key '{MODEL_KV_KEY}' not found")))?;
    RustContrastiveEmbedder::from_bytes(&bytes).map_err(Error::RustError)
}

// --- Handler ---

#[event(fetch)]
async fn fetch(req: HttpRequest, env: Env, _ctx: Context) -> Result<HttpResponse> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    match (method, path.as_str()) {
        (http::Method::POST, "/v1/embeddings") => handle_embeddings(req, &env).await,
        _ => error_response(http::StatusCode::NOT_FOUND, "Not found"),
    }
}

async fn handle_embeddings(req: HttpRequest, env: &Env) -> Result<HttpResponse> {
    let mut body = req.into_body();
    let mut body_bytes = Vec::new();
    while let Some(chunk) = futures_util::StreamExt::next(&mut body).await {
        body_bytes.extend_from_slice(&chunk?);
    }

    let request: EmbeddingRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return error_response(http::StatusCode::BAD_REQUEST, e.to_string()),
    };

    let _encoding_format = request.encoding_format.as_deref().unwrap_or("float");
    let model_name = request.model;
    let texts = request.input.into_texts();

    let embedder = match load_embedder(env).await {
        Ok(e) => e,
        Err(e) => {
            return error_response(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to load model: {e}"),
            )
        }
    };

    let mut total_tokens = 0usize;
    let data: Vec<EmbeddingData> = texts
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let tokens = text.split_whitespace().count();
            total_tokens += tokens;
            EmbeddingData {
                object: "embedding",
                embedding: embedder.embed(text),
                index,
            }
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list",
        data,
        model: model_name,
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    json_response(http::StatusCode::OK, &response)
}
