use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use polars::prelude::{DataFrame, NamedFrom, PolarsResult, Series};

pub mod rust_embedder;
#[cfg(feature = "semantic")]
pub mod semantic;
pub mod transformer;

pub const EMBEDDING_DIM: usize = 512;

#[derive(Debug, Clone)]
pub struct TinyTextEmbedder {
    dim: usize,
}

impl Default for TinyTextEmbedder {
    fn default() -> Self {
        Self::new(EMBEDDING_DIM)
    }
}

impl TinyTextEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "embedding dimension must be greater than zero");
        Self { dim }
    }

    pub fn dimension(&self) -> usize {
        self.dim
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dim];
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return vector;
        }

        let token_scale = 1.0_f32 / (tokens.len() as f32).sqrt();
        for token in tokens {
            let token = token.to_ascii_lowercase();
            let hash = hash64(&token);
            let primary = (hash as usize) % self.dim;
            let secondary = (hash.rotate_left(17) as usize) % self.dim;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };

            vector[primary] += sign * token_scale;
            vector[secondary] += -sign * token_scale * 0.5;
        }

        l2_normalize(&mut vector);
        vector
    }

    pub fn embed_texts_frame(&self, texts: &[&str]) -> PolarsResult<DataFrame> {
        let embeddings: Vec<Vec<f32>> = texts.iter().map(|text| self.embed(text)).collect();

        let mut columns = Vec::with_capacity(self.dim + 1);
        columns.push(Series::new("text".into(), texts.to_vec()).into());

        for dim in 0..self.dim {
            let values: Vec<f32> = embeddings.iter().map(|row| row[dim]).collect();
            columns.push(Series::new(format!("emb_{dim}").into(), values).into());
        }

        DataFrame::new(columns)
    }
}

fn hash64(value: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedder_outputs_512_dimensions() {
        let model = TinyTextEmbedder::default();
        let embedding = model.embed("tiny embedding model");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        assert_eq!(model.dimension(), EMBEDDING_DIM);
    }

    #[test]
    fn embedding_is_deterministic() {
        let model = TinyTextEmbedder::default();
        let first = model.embed("Deterministic output");
        let second = model.embed("deterministic output");
        assert_eq!(first, second);
    }

    #[test]
    fn empty_text_returns_zero_vector() {
        let model = TinyTextEmbedder::default();
        let embedding = model.embed("");
        assert!(embedding.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn non_empty_text_is_l2_normalized() {
        let model = TinyTextEmbedder::default();
        let embedding = model.embed("normalization check");
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn embeddings_dataframe_has_expected_shape() {
        let model = TinyTextEmbedder::default();
        let texts = ["first sample", "second sample"];
        let frame = model
            .embed_texts_frame(&texts)
            .expect("dataframe should build");
        assert_eq!(frame.height(), 2);
        assert_eq!(frame.width(), EMBEDDING_DIM + 1);
    }
}
