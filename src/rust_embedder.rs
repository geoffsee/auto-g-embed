use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::EMBEDDING_DIM;

const PAD_TOKEN_ID: usize = 0;
const UNKNOWN_TOKEN_ID: usize = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustEmbedderConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_seq_len: usize,
    pub normalize: bool,
}

impl Default for RustEmbedderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8_192,
            embedding_dim: EMBEDDING_DIM,
            max_seq_len: 64,
            normalize: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RustEvalMetrics {
    pub triplet_accuracy: f32,
    pub mean_positive_similarity: f32,
    pub mean_negative_similarity: f32,
    pub mean_margin: f32,
}

#[derive(Debug, Clone)]
pub struct RustContrastiveEmbedder {
    config: RustEmbedderConfig,
    token_embeddings: Vec<f32>,
}

impl Default for RustContrastiveEmbedder {
    fn default() -> Self {
        Self::new(RustEmbedderConfig::default(), 17)
    }
}

impl RustContrastiveEmbedder {
    pub fn new(config: RustEmbedderConfig, seed: u64) -> Self {
        assert!(config.vocab_size > 2, "vocab_size must be > 2");
        assert!(config.embedding_dim > 0, "embedding_dim must be > 0");
        assert!(config.max_seq_len > 0, "max_seq_len must be > 0");

        let mut rng = SmallRng::new(seed);
        let scale = 0.05_f32;
        let total = config.vocab_size * config.embedding_dim;
        let mut token_embeddings = Vec::with_capacity(total);
        for _ in 0..total {
            token_embeddings.push(rng.next_signed() * scale);
        }
        for dim in 0..config.embedding_dim {
            token_embeddings[PAD_TOKEN_ID * config.embedding_dim + dim] = 0.0;
        }

        Self {
            config,
            token_embeddings,
        }
    }

    pub fn config(&self) -> &RustEmbedderConfig {
        &self.config
    }

    pub fn dimension(&self) -> usize {
        self.config.embedding_dim
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let token_ids = self.tokenize_to_ids(text);
        if token_ids.is_empty() {
            return vec![0.0; self.config.embedding_dim];
        }

        let mut embedding = self.mean_embed_token_ids(&token_ids);
        if self.config.normalize {
            l2_normalize(&mut embedding);
        }
        embedding
    }

    pub fn train_step(
        &mut self,
        anchor: &str,
        positive: &str,
        negative: &str,
        learning_rate: f32,
        margin: f32,
    ) -> f32 {
        let anchor_tokens = self.tokenize_to_ids(anchor);
        let positive_tokens = self.tokenize_to_ids(positive);
        let negative_tokens = self.tokenize_to_ids(negative);

        if anchor_tokens.is_empty() || positive_tokens.is_empty() || negative_tokens.is_empty() {
            return 0.0;
        }

        let mut anchor_emb = self.mean_embed_token_ids(&anchor_tokens);
        let mut positive_emb = self.mean_embed_token_ids(&positive_tokens);
        let mut negative_emb = self.mean_embed_token_ids(&negative_tokens);

        if self.config.normalize {
            l2_normalize(&mut anchor_emb);
            l2_normalize(&mut positive_emb);
            l2_normalize(&mut negative_emb);
        }

        let sim_positive = dot(&anchor_emb, &positive_emb);
        let sim_negative = dot(&anchor_emb, &negative_emb);
        let loss = margin - sim_positive + sim_negative;
        if loss <= 0.0 {
            return 0.0;
        }

        let mut grad_anchor = vec![0.0_f32; self.config.embedding_dim];
        let mut grad_positive = vec![0.0_f32; self.config.embedding_dim];
        let mut grad_negative = vec![0.0_f32; self.config.embedding_dim];

        for dim in 0..self.config.embedding_dim {
            grad_anchor[dim] = -positive_emb[dim] + negative_emb[dim];
            grad_positive[dim] = -anchor_emb[dim];
            grad_negative[dim] = anchor_emb[dim];
        }

        self.apply_token_gradients(&anchor_tokens, &grad_anchor, learning_rate);
        self.apply_token_gradients(&positive_tokens, &grad_positive, learning_rate);
        self.apply_token_gradients(&negative_tokens, &grad_negative, learning_rate);

        loss
    }

    pub fn evaluate_triplets(&self, triplets: &[(String, String, String)]) -> RustEvalMetrics {
        if triplets.is_empty() {
            return RustEvalMetrics {
                triplet_accuracy: 0.0,
                mean_positive_similarity: 0.0,
                mean_negative_similarity: 0.0,
                mean_margin: 0.0,
            };
        }

        let mut wins = 0usize;
        let mut positive_total = 0.0_f32;
        let mut negative_total = 0.0_f32;

        for (anchor, positive, negative) in triplets {
            let anchor_emb = self.embed(anchor);
            let positive_emb = self.embed(positive);
            let negative_emb = self.embed(negative);

            let positive_sim = dot(&anchor_emb, &positive_emb);
            let negative_sim = dot(&anchor_emb, &negative_emb);
            if positive_sim > negative_sim {
                wins += 1;
            }

            positive_total += positive_sim;
            negative_total += negative_sim;
        }

        let count = triplets.len() as f32;
        let mean_positive_similarity = positive_total / count;
        let mean_negative_similarity = negative_total / count;

        RustEvalMetrics {
            triplet_accuracy: wins as f32 / count,
            mean_positive_similarity,
            mean_negative_similarity,
            mean_margin: mean_positive_similarity - mean_negative_similarity,
        }
    }

    pub fn save_dir(&self, out_dir: impl AsRef<Path>) -> Result<(), String> {
        let out_dir = out_dir.as_ref();
        fs::create_dir_all(out_dir)
            .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

        let config_path = out_dir.join("config.json");
        let config_text = serde_json::to_string_pretty(&self.config)
            .map_err(|e| format!("failed to serialize config: {e}"))?;
        fs::write(&config_path, config_text)
            .map_err(|e| format!("failed to write {}: {e}", config_path.display()))?;

        let weights_path = out_dir.join("token_embeddings.f32");
        let file = File::create(&weights_path)
            .map_err(|e| format!("failed to write {}: {e}", weights_path.display()))?;
        let mut writer = BufWriter::new(file);
        for value in &self.token_embeddings {
            writer
                .write_all(&value.to_le_bytes())
                .map_err(|e| format!("failed to write weights: {e}"))?;
        }
        writer
            .flush()
            .map_err(|e| format!("failed to flush weights: {e}"))?;

        Ok(())
    }

    pub fn load_dir(model_dir: impl AsRef<Path>) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("token_embeddings.f32");

        let config_text = fs::read_to_string(&config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
        let config: RustEmbedderConfig = serde_json::from_str(&config_text)
            .map_err(|e| format!("invalid {}: {e}", config_path.display()))?;
        if config.vocab_size <= 2 {
            return Err("invalid config: vocab_size must be > 2".to_string());
        }
        if config.embedding_dim == 0 {
            return Err("invalid config: embedding_dim must be > 0".to_string());
        }
        if config.max_seq_len == 0 {
            return Err("invalid config: max_seq_len must be > 0".to_string());
        }

        let expected_len = config.vocab_size * config.embedding_dim;
        let expected_bytes = expected_len * std::mem::size_of::<f32>();

        let file = File::open(&weights_path)
            .map_err(|e| format!("failed to read {}: {e}", weights_path.display()))?;
        let mut reader = BufReader::new(file);
        let mut raw = Vec::new();
        reader
            .read_to_end(&mut raw)
            .map_err(|e| format!("failed to read {}: {e}", weights_path.display()))?;

        if raw.len() != expected_bytes {
            return Err(format!(
                "invalid weights size in {}: expected {} bytes, found {}",
                weights_path.display(),
                expected_bytes,
                raw.len()
            ));
        }

        let mut token_embeddings = Vec::with_capacity(expected_len);
        for chunk in raw.chunks_exact(4) {
            let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            token_embeddings.push(f32::from_le_bytes(bytes));
        }

        Ok(Self {
            config,
            token_embeddings,
        })
    }

    pub fn shuffle_pairs(pairs: &mut [(String, String)], rng: &mut SmallRng) {
        if pairs.len() < 2 {
            return;
        }
        for i in (1..pairs.len()).rev() {
            let j = rng.gen_range(0, i + 1);
            pairs.swap(i, j);
        }
    }

    pub fn sample_negative<'a>(
        &self,
        negative_pool: &'a [String],
        anchor: &str,
        positive: &str,
        rng: &mut SmallRng,
    ) -> Option<&'a str> {
        if negative_pool.is_empty() {
            return None;
        }
        for _ in 0..32 {
            let idx = rng.gen_range(0, negative_pool.len());
            let candidate = negative_pool[idx].as_str();
            if candidate != anchor && candidate != positive {
                return Some(candidate);
            }
        }
        None
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.config.max_seq_len);
        for token in text.split_whitespace() {
            if ids.len() >= self.config.max_seq_len {
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
            let bucket = (hasher.finish() as usize) % (self.config.vocab_size - 2);
            ids.push(bucket + 2);
        }
        ids.retain(|id| *id != PAD_TOKEN_ID);
        ids
    }

    fn mean_embed_token_ids(&self, token_ids: &[usize]) -> Vec<f32> {
        let mut embedding = vec![0.0_f32; self.config.embedding_dim];
        let inv_len = 1.0_f32 / token_ids.len() as f32;

        for token_id in token_ids {
            let offset = token_id * self.config.embedding_dim;
            for (dim, value) in embedding.iter_mut().enumerate() {
                *value += self.token_embeddings[offset + dim];
            }
        }

        for value in &mut embedding {
            *value *= inv_len;
        }
        embedding
    }

    fn apply_token_gradients(&mut self, token_ids: &[usize], grad: &[f32], learning_rate: f32) {
        let scale = learning_rate / token_ids.len() as f32;
        for token_id in token_ids {
            let offset = token_id * self.config.embedding_dim;
            for (dim, grad_value) in grad.iter().copied().enumerate() {
                self.token_embeddings[offset + dim] -= scale * grad_value;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    pub fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    pub fn gen_range(&mut self, start: usize, end: usize) -> usize {
        if end <= start {
            return start;
        }
        let span = end - start;
        (self.next_u64() as usize % span) + start
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
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
        let model = RustContrastiveEmbedder::default();
        let embedding = model.embed("rust native semantic embedding");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn training_step_produces_positive_loss_on_hard_triplet() {
        let config = RustEmbedderConfig {
            vocab_size: 128,
            embedding_dim: 32,
            max_seq_len: 16,
            normalize: true,
        };
        let mut model = RustContrastiveEmbedder::new(config, 13);

        let loss = model.train_step(
            "what is rust ownership",
            "best chocolate cake recipe",
            "what is rust ownership",
            0.08,
            0.2,
        );
        assert!(loss > 0.0);
    }

    #[test]
    fn save_load_round_trip_preserves_embeddings() {
        let config = RustEmbedderConfig {
            vocab_size: 128,
            embedding_dim: 16,
            max_seq_len: 16,
            normalize: true,
        };
        let model = RustContrastiveEmbedder::new(config, 19);
        let text = "round trip test";
        let before = model.embed(text);

        let tmp = std::env::temp_dir().join(format!(
            "auto_g_embed_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time should be after epoch")
                .as_nanos()
        ));
        model.save_dir(&tmp).expect("save should succeed");

        let loaded = RustContrastiveEmbedder::load_dir(&tmp).expect("load should succeed");
        let after = loaded.embed(text);
        assert_eq!(before, after);

        let _ = fs::remove_dir_all(tmp);
    }
}
