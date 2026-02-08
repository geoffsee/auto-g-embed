use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::EMBEDDING_DIM;

const PAD_TOKEN_ID: usize = 0;
const UNKNOWN_TOKEN_ID: usize = 1;

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub model_dim: usize,
    pub ff_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub layer_norm_eps: f32,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_768,
            max_seq_len: 64,
            model_dim: EMBEDDING_DIM,
            ff_dim: 2_048,
            num_layers: 2,
            num_heads: 8,
            layer_norm_eps: 1e-5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScratchTransformerEmbedder {
    config: TransformerConfig,
    token_embedding: Matrix,
    layers: Vec<TransformerBlock>,
}

impl Default for ScratchTransformerEmbedder {
    fn default() -> Self {
        Self::new(TransformerConfig::default(), 17)
    }
}

impl ScratchTransformerEmbedder {
    pub fn new(config: TransformerConfig, seed: u64) -> Self {
        assert!(config.vocab_size > 2, "vocab_size must be > 2");
        assert!(config.max_seq_len > 0, "max_seq_len must be > 0");
        assert!(config.model_dim > 0, "model_dim must be > 0");
        assert!(config.ff_dim > 0, "ff_dim must be > 0");
        assert!(config.num_layers > 0, "num_layers must be > 0");
        assert!(config.num_heads > 0, "num_heads must be > 0");
        assert!(
            config.model_dim.is_multiple_of(config.num_heads),
            "model_dim must be divisible by num_heads"
        );

        let mut rng = SmallRng::new(seed);
        let token_embedding = Matrix::random(config.vocab_size, config.model_dim, &mut rng);

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerBlock::new(&config, &mut rng));
        }

        Self {
            config,
            token_embedding,
            layers,
        }
    }

    pub fn dimension(&self) -> usize {
        self.config.model_dim
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let token_ids = tokenize_to_ids(text, self.config.vocab_size, self.config.max_seq_len);
        if token_ids.is_empty() {
            return vec![0.0; self.config.model_dim];
        }

        let mut hidden = self.embed_tokens(&token_ids);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, self.config.layer_norm_eps);
        }

        let mut pooled = mean_pool(&hidden);
        l2_normalize(&mut pooled);
        pooled
    }

    fn embed_tokens(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        let mut hidden = Vec::with_capacity(token_ids.len());
        for (position, token_id) in token_ids.iter().copied().enumerate() {
            let mut token = self.token_embedding.row(token_id);
            let position_embedding = sinusoidal_position_embedding(position, self.config.model_dim);
            for i in 0..self.config.model_dim {
                token[i] += position_embedding[i];
            }
            hidden.push(token);
        }
        hidden
    }
}

#[derive(Debug, Clone)]
struct TransformerBlock {
    w_q: Matrix,
    w_k: Matrix,
    w_v: Matrix,
    w_o: Matrix,
    ff_1: Matrix,
    ff_1_bias: Vec<f32>,
    ff_2: Matrix,
    ff_2_bias: Vec<f32>,
    norm_1_gamma: Vec<f32>,
    norm_1_beta: Vec<f32>,
    norm_2_gamma: Vec<f32>,
    norm_2_beta: Vec<f32>,
    num_heads: usize,
}

impl TransformerBlock {
    fn new(config: &TransformerConfig, rng: &mut SmallRng) -> Self {
        Self {
            w_q: Matrix::random(config.model_dim, config.model_dim, rng),
            w_k: Matrix::random(config.model_dim, config.model_dim, rng),
            w_v: Matrix::random(config.model_dim, config.model_dim, rng),
            w_o: Matrix::random(config.model_dim, config.model_dim, rng),
            ff_1: Matrix::random(config.model_dim, config.ff_dim, rng),
            ff_1_bias: vec![0.0; config.ff_dim],
            ff_2: Matrix::random(config.ff_dim, config.model_dim, rng),
            ff_2_bias: vec![0.0; config.model_dim],
            norm_1_gamma: vec![1.0; config.model_dim],
            norm_1_beta: vec![0.0; config.model_dim],
            norm_2_gamma: vec![1.0; config.model_dim],
            norm_2_beta: vec![0.0; config.model_dim],
            num_heads: config.num_heads,
        }
    }

    fn forward(&self, hidden: &[Vec<f32>], eps: f32) -> Vec<Vec<f32>> {
        let normed_1: Vec<Vec<f32>> = hidden
            .iter()
            .map(|v| layer_norm(v, &self.norm_1_gamma, &self.norm_1_beta, eps))
            .collect();
        let attn_out = self.self_attention(&normed_1);
        let residual_1 = add_residual(hidden, &attn_out);

        let normed_2: Vec<Vec<f32>> = residual_1
            .iter()
            .map(|v| layer_norm(v, &self.norm_2_gamma, &self.norm_2_beta, eps))
            .collect();
        let ff_out = self.feed_forward(&normed_2);
        add_residual(&residual_1, &ff_out)
    }

    fn self_attention(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let q = self.w_q.mul_batch(hidden);
        let k = self.w_k.mul_batch(hidden);
        let v = self.w_v.mul_batch(hidden);

        let sequence_len = hidden.len();
        let model_dim = hidden[0].len();
        let head_dim = model_dim / self.num_heads;
        let scale = (head_dim as f32).sqrt();

        let mut context = vec![vec![0.0; model_dim]; sequence_len];
        for head in 0..self.num_heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;

            for i in 0..sequence_len {
                let mut scores = Vec::with_capacity(sequence_len);
                for key in k.iter().take(sequence_len) {
                    let dot = dot_slice(&q[i][head_start..head_end], &key[head_start..head_end]);
                    scores.push(dot / scale);
                }
                let probs = softmax(&scores);

                for (j, prob) in probs.iter().copied().enumerate() {
                    for d in 0..head_dim {
                        context[i][head_start + d] += prob * v[j][head_start + d];
                    }
                }
            }
        }

        self.w_o.mul_batch(&context)
    }

    fn feed_forward(&self, hidden: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let ff_inner = self.ff_1.mul_batch(hidden);
        let ff_inner = add_bias_and_gelu(&ff_inner, &self.ff_1_bias);
        let ff_out = self.ff_2.mul_batch(&ff_inner);
        add_bias(&ff_out, &self.ff_2_bias)
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    fn random(rows: usize, cols: usize, rng: &mut SmallRng) -> Self {
        let scale = (2.0_f32 / (rows + cols) as f32).sqrt();
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            data.push(rng.next_signed() * scale);
        }
        Self { rows, cols, data }
    }

    fn row(&self, row: usize) -> Vec<f32> {
        assert!(row < self.rows, "row index out of bounds");
        let start = row * self.cols;
        let end = start + self.cols;
        self.data[start..end].to_vec()
    }

    fn mul_vec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.rows, "input size mismatch");
        let mut output = vec![0.0; self.cols];
        for (r, x) in input.iter().copied().enumerate() {
            let row_offset = r * self.cols;
            for (c, out) in output.iter_mut().enumerate() {
                *out += x * self.data[row_offset + c];
            }
        }
        output
    }

    fn mul_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter().map(|input| self.mul_vec(input)).collect()
    }
}

#[derive(Debug, Clone)]
struct SmallRng {
    state: u64,
}

impl SmallRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

fn tokenize_to_ids(text: &str, vocab_size: usize, max_seq_len: usize) -> Vec<usize> {
    let mut ids = Vec::with_capacity(max_seq_len);
    for token in text.split_whitespace() {
        if ids.len() >= max_seq_len {
            break;
        }
        let cleaned = token
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();

        if cleaned.is_empty() {
            ids.push(UNKNOWN_TOKEN_ID);
            continue;
        }

        let mut hasher = DefaultHasher::new();
        cleaned.hash(&mut hasher);
        let hash = hasher.finish() as usize;
        let bucket = hash % (vocab_size - 2);
        ids.push(bucket + 2);
    }
    ids.retain(|id| *id != PAD_TOKEN_ID);
    ids
}

fn add_residual(left: &[Vec<f32>], right: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert_eq!(left.len(), right.len(), "residual batch mismatch");
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| {
            assert_eq!(l.len(), r.len(), "residual vector mismatch");
            l.iter().zip(r.iter()).map(|(a, b)| a + b).collect()
        })
        .collect()
}

fn add_bias(values: &[Vec<f32>], bias: &[f32]) -> Vec<Vec<f32>> {
    values
        .iter()
        .map(|row| {
            assert_eq!(row.len(), bias.len(), "bias size mismatch");
            row.iter().zip(bias.iter()).map(|(v, b)| v + b).collect()
        })
        .collect()
}

fn add_bias_and_gelu(values: &[Vec<f32>], bias: &[f32]) -> Vec<Vec<f32>> {
    values
        .iter()
        .map(|row| {
            assert_eq!(row.len(), bias.len(), "bias size mismatch");
            row.iter()
                .zip(bias.iter())
                .map(|(v, b)| gelu(v + b))
                .collect()
        })
        .collect()
}

fn mean_pool(hidden: &[Vec<f32>]) -> Vec<f32> {
    let sequence_len = hidden.len();
    let hidden_dim = hidden[0].len();
    let mut pooled = vec![0.0; hidden_dim];
    for token in hidden {
        for (i, value) in token.iter().copied().enumerate() {
            pooled[i] += value;
        }
    }
    for value in &mut pooled {
        *value /= sequence_len as f32;
    }
    pooled
}

fn layer_norm(values: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(values.len(), gamma.len(), "gamma size mismatch");
    assert_eq!(values.len(), beta.len(), "beta size mismatch");

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|v| {
            let centered = v - mean;
            centered * centered
        })
        .sum::<f32>()
        / values.len() as f32;
    let denom = (variance + eps).sqrt();

    values
        .iter()
        .zip(gamma.iter().zip(beta.iter()))
        .map(|(v, (g, b))| ((v - mean) / denom) * g + b)
        .collect()
}

fn sinusoidal_position_embedding(position: usize, dim: usize) -> Vec<f32> {
    let mut pos = vec![0.0; dim];
    for (i, value) in pos.iter_mut().enumerate().take(dim) {
        let angle_rate = 1.0_f32 / 10_000_f32.powf((2 * (i / 2)) as f32 / dim as f32);
        let angle = position as f32 * angle_rate;
        *value = if i % 2 == 0 { angle.sin() } else { angle.cos() };
    }
    pos
}

fn dot_slice(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max_value = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
    let mut exp_values = Vec::with_capacity(values.len());
    let mut sum = 0.0_f32;
    for v in values {
        let exp = (v - max_value).exp();
        exp_values.push(exp);
        sum += exp;
    }
    if sum <= 0.0 {
        return vec![1.0 / values.len() as f32; values.len()];
    }
    exp_values.into_iter().map(|v| v / sum).collect()
}

fn gelu(value: f32) -> f32 {
    // tanh approximation from the original GELU paper.
    let k = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * value * (1.0 + (k * (value + 0.044_715 * value.powi(3))).tanh())
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_outputs_configured_dimensions() {
        let model = ScratchTransformerEmbedder::default();
        let embedding = model.embed("transformer from scratch in rust");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn deterministic_for_same_seed() {
        let config = TransformerConfig::default();
        let first = ScratchTransformerEmbedder::new(config.clone(), 123);
        let second = ScratchTransformerEmbedder::new(config, 123);

        let emb_a = first.embed("repeatable semantics");
        let emb_b = second.embed("repeatable semantics");
        assert_eq!(emb_a, emb_b);
    }

    #[test]
    fn normalized_non_empty_embedding() {
        let model = ScratchTransformerEmbedder::default();
        let embedding = model.embed("normalization check");
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn empty_text_returns_zero_vector() {
        let model = ScratchTransformerEmbedder::default();
        let embedding = model.embed("");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert!(embedding.iter().all(|v| *v == 0.0));
    }
}
