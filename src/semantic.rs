use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ndarray::{Array2, ArrayViewD};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

const DEFAULT_MAX_LENGTH: usize = 128;

#[derive(Debug, Deserialize)]
struct ExportedEmbedderConfig {
    max_length: Option<usize>,
    pooling: Option<String>,
    normalize: Option<bool>,
}

/// ONNX-backed semantic text embedder exported by `training/export_onnx_model.py`.
///
/// This runtime assumes a transformer feature-extraction graph and performs
/// mean pooling + optional L2 normalization in Rust.
pub struct SemanticEmbedder {
    tokenizer: Tokenizer,
    session: Session,
    max_length: usize,
    normalize: bool,
}

impl SemanticEmbedder {
    pub fn from_onnx_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let model_path = resolve_existing_path(dir, &["model.onnx", "onnx/model.onnx"])
            .with_context(|| format!("No ONNX model found under {}", dir.display()))?;
        let tokenizer_path = resolve_existing_path(dir, &["tokenizer.json", "onnx/tokenizer.json"])
            .with_context(|| format!("No tokenizer.json found under {}", dir.display()))?;
        let config_path =
            resolve_existing_path(dir, &["embedder_config.json", "onnx/embedder_config.json"]);

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow::anyhow!(
                "failed to load tokenizer {}: {err}",
                tokenizer_path.display()
            )
        })?;

        let mut pooling = String::from("mean");
        let mut normalize = true;
        let mut max_length = DEFAULT_MAX_LENGTH;

        if let Some(path) = config_path {
            let cfg_text = fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))?;
            let cfg: ExportedEmbedderConfig = serde_json::from_str(&cfg_text)
                .with_context(|| format!("invalid JSON in {}", path.display()))?;
            if let Some(value) = cfg.pooling {
                pooling = value;
            }
            if let Some(value) = cfg.normalize {
                normalize = value;
            }
            if let Some(value) = cfg.max_length {
                max_length = value;
            }
        }

        if !pooling.eq_ignore_ascii_case("mean") {
            bail!("unsupported pooling strategy '{pooling}', expected 'mean'");
        }
        if max_length == 0 {
            bail!("max_length must be > 0");
        }

        let pad_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);
        let pad_token = tokenizer
            .id_to_token(pad_id)
            .unwrap_or_else(|| "[PAD]".to_owned());

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|err| anyhow::anyhow!("invalid truncation settings: {err}"))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(max_length),
            pad_id,
            pad_token,
            ..Default::default()
        }));

        let session = Session::builder()
            .context("failed to build ONNX session")?
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load ONNX model {}", model_path.display()))?;

        Ok(Self {
            tokenizer,
            session,
            max_length,
            normalize,
        })
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| anyhow::anyhow!("tokenization failed: {err}"))?;

        let token_count = encoding.len();
        if token_count == 0 {
            return Ok(Vec::new());
        }

        let input_ids_values: Vec<i64> = encoding.get_ids().iter().map(|&v| i64::from(v)).collect();
        let attention_mask_values: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&v| i64::from(v))
            .collect();

        let type_ids_values: Vec<i64> = if encoding.get_type_ids().is_empty() {
            vec![0; token_count]
        } else {
            encoding
                .get_type_ids()
                .iter()
                .map(|&v| i64::from(v))
                .collect()
        };

        let input_ids = Array2::from_shape_vec((1, token_count), input_ids_values)
            .context("failed to build input_ids tensor")?;
        let attention_mask =
            Array2::from_shape_vec((1, token_count), attention_mask_values.clone())
                .context("failed to build attention_mask tensor")?;
        let token_type_ids = Array2::from_shape_vec((1, token_count), type_ids_values)
            .context("failed to build token_type_ids tensor")?;

        let input_name_lowers: Vec<String> = self
            .session
            .inputs()
            .iter()
            .map(|o| o.name().to_ascii_lowercase())
            .collect();
        let has_named_input_ids = input_name_lowers.iter().any(|n| n.contains("input_ids"));
        let has_named_attention = input_name_lowers
            .iter()
            .any(|n| n.contains("attention_mask"));
        let has_named_type_ids = input_name_lowers
            .iter()
            .any(|n| n.contains("token_type_ids"));

        let mut ort_inputs = Vec::with_capacity(self.session.inputs().len());
        for (index, outlet) in self.session.inputs().iter().enumerate() {
            let lower_name = outlet.name().to_ascii_lowercase();

            let tensor = if lower_name.contains("input_ids") || (!has_named_input_ids && index == 0) {
                Tensor::from_array(input_ids.clone())
            } else if lower_name.contains("attention_mask")
                || (!has_named_attention && index == 1)
            {
                Tensor::from_array(attention_mask.clone())
            } else if lower_name.contains("token_type_ids")
                || (!has_named_type_ids && index == 2)
            {
                Tensor::from_array(token_type_ids.clone())
            } else {
                bail!(
                    "unsupported ONNX input '{}'; expected input_ids/attention_mask/token_type_ids",
                    outlet.name()
                );
            }
            .with_context(|| format!("failed to build input tensor '{}'", outlet.name()))?;

            ort_inputs.push((outlet.name().to_owned(), tensor));
        }

        let outputs = self
            .session
            .run(ort_inputs)
            .context("ONNX inference failed")?;

        let output = if outputs.contains_key("last_hidden_state") {
            &outputs["last_hidden_state"]
        } else {
            &outputs[0]
        };

        let output_array = output
            .try_extract_array::<f32>()
            .context("failed to extract ONNX output tensor as f32 array")?;

        let mut embedding = match output_array.ndim() {
            3 => mean_pool_3d(&output_array, &attention_mask_values)?,
            2 => first_row_2d(&output_array)?,
            ndim => bail!("unexpected ONNX output rank {ndim}; expected 2D or 3D tensor"),
        };

        if self.normalize {
            l2_normalize(&mut embedding);
        }

        Ok(embedding)
    }

    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }
}

fn resolve_existing_path(root: &Path, candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(|relative| root.join(relative))
        .find(|candidate| candidate.exists())
}

fn first_row_2d(array: &ArrayViewD<'_, f32>) -> Result<Vec<f32>> {
    let shape = array.shape();
    if shape.len() != 2 || shape[0] == 0 {
        bail!("invalid 2D output shape: {shape:?}");
    }

    let hidden_size = shape[1];
    let mut embedding = Vec::with_capacity(hidden_size);
    for h in 0..hidden_size {
        embedding.push(array[[0, h]]);
    }
    Ok(embedding)
}

fn mean_pool_3d(array: &ArrayViewD<'_, f32>, attention_mask: &[i64]) -> Result<Vec<f32>> {
    let shape = array.shape();
    if shape.len() != 3 || shape[0] == 0 {
        bail!("invalid 3D output shape: {shape:?}");
    }

    let sequence_len = shape[1];
    let hidden_size = shape[2];
    let mut pooled = vec![0.0_f32; hidden_size];
    let mut token_counter = 0.0_f32;

    for token_idx in 0..sequence_len {
        if attention_mask.get(token_idx).copied().unwrap_or(0) <= 0 {
            continue;
        }

        token_counter += 1.0;
        for hidden_idx in 0..hidden_size {
            pooled[hidden_idx] += array[[0, token_idx, hidden_idx]];
        }
    }

    if token_counter > 0.0 {
        for value in &mut pooled {
            *value /= token_counter;
        }
    }

    Ok(pooled)
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
    use ndarray::Array3;

    #[test]
    fn mean_pool_uses_attention_mask() {
        let array = Array3::from_shape_vec(
            (1, 3, 2),
            vec![
                1.0, 2.0, //
                999.0, 999.0, //
                5.0, 6.0,
            ],
        )
        .expect("array should build")
        .into_dyn();

        let pooled = mean_pool_3d(&array.view(), &[1, 0, 1]).expect("pooling should work");
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 3.0).abs() < 1e-6);
        assert!((pooled[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_scales_to_unit_norm() {
        let mut values = vec![3.0_f32, 4.0_f32];
        l2_normalize(&mut values);
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
