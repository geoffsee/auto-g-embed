use std::collections::HashSet;
use std::env;
use std::fs;
use std::fs::File;
use std::path::PathBuf;

use auto_g_embed::rust_embedder::{
    RustContrastiveEmbedder, RustEmbedderConfig, RustEvalMetrics, SmallRng,
};
use polars::io::parquet::read::ParquetReader;
use polars::prelude::SerReader;
use serde::Serialize;

const DEFAULT_TRAIN_PAIRS: &str = "artifacts/contrastive-data/train_pairs.parquet";
const DEFAULT_EVAL_TRIPLETS: &str = "artifacts/contrastive-data/eval_triplets.parquet";
const DEFAULT_OUTPUT_DIR: &str = "artifacts/model/rust-embedder";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(env::args().skip(1).collect())?;

    let mut train_pairs = load_train_pairs(&args.train_pairs)?;
    if train_pairs.is_empty() {
        return Err(format!(
            "No train pairs found in {}",
            args.train_pairs.display()
        ));
    }
    sample_in_place(&mut train_pairs, args.max_train_pairs, args.seed);

    let eval_triplets = load_eval_triplets(&args.eval_triplets)?;
    if eval_triplets.is_empty() {
        return Err(format!(
            "No eval triplets found in {}",
            args.eval_triplets.display()
        ));
    }

    let mut negative_pool = HashSet::<String>::new();
    for (anchor, positive) in &train_pairs {
        negative_pool.insert(anchor.clone());
        negative_pool.insert(positive.clone());
    }
    let negative_pool = negative_pool.into_iter().collect::<Vec<_>>();
    if negative_pool.len() < 2 {
        return Err("negative pool is too small to sample negatives".to_string());
    }

    let config = RustEmbedderConfig {
        vocab_size: args.vocab_size,
        embedding_dim: args.embedding_dim,
        max_seq_len: args.max_seq_len,
        normalize: true,
    };
    let mut model = RustContrastiveEmbedder::new(config, args.seed.wrapping_add(1));

    println!(
        "[train-rust] pairs={} eval_triplets={} vocab={} dim={} seq_len={}",
        train_pairs.len(),
        eval_triplets.len(),
        args.vocab_size,
        args.embedding_dim,
        args.max_seq_len
    );
    println!(
        "[train-rust] epochs={} lr={} margin={}",
        args.epochs, args.learning_rate, args.margin
    );

    let mut rng = SmallRng::new(args.seed.wrapping_add(7));
    for epoch in 1..=args.epochs {
        RustContrastiveEmbedder::shuffle_pairs(train_pairs.as_mut_slice(), &mut rng);

        let mut epoch_loss = 0.0_f32;
        let mut active_steps = 0usize;
        let mut steps = 0usize;

        for (anchor, positive) in &train_pairs {
            let Some(negative) = model.sample_negative(&negative_pool, anchor, positive, &mut rng)
            else {
                continue;
            };

            let loss =
                model.train_step(anchor, positive, negative, args.learning_rate, args.margin);
            if loss > 0.0 {
                epoch_loss += loss;
                active_steps += 1;
            }
            steps += 1;
        }

        let avg_loss = if active_steps > 0 {
            epoch_loss / active_steps as f32
        } else {
            0.0
        };
        println!(
            "[train-rust] epoch={epoch} steps={steps} active_steps={active_steps} avg_active_loss={avg_loss:.6}"
        );
    }

    let metrics = model.evaluate_triplets(&eval_triplets);
    fs::create_dir_all(&args.output_dir).map_err(|e| {
        format!(
            "failed to create output directory {}: {e}",
            args.output_dir.display()
        )
    })?;
    model.save_dir(&args.output_dir)?;

    let report = TrainingReport {
        backend: "rust_local".to_string(),
        train_pairs_used: train_pairs.len(),
        eval_triplets: eval_triplets.len(),
        epochs: args.epochs,
        learning_rate: args.learning_rate,
        margin: args.margin,
        vocab_size: args.vocab_size,
        embedding_dim: args.embedding_dim,
        max_seq_len: args.max_seq_len,
        seed: args.seed,
        metrics,
    };
    let report_path = args.output_dir.join("metrics.json");
    let report_text = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("failed to serialize metrics: {e}"))?;
    fs::write(&report_path, report_text)
        .map_err(|e| format!("failed to write {}: {e}", report_path.display()))?;

    println!(
        "[train-rust] wrote model={} metrics={}",
        args.output_dir.display(),
        report_path.display()
    );
    println!(
        "[train-rust] triplet_accuracy={:.4} mean_margin={:.4}",
        report.metrics.triplet_accuracy, report.metrics.mean_margin
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct CliArgs {
    train_pairs: PathBuf,
    eval_triplets: PathBuf,
    output_dir: PathBuf,
    epochs: usize,
    learning_rate: f32,
    margin: f32,
    max_train_pairs: usize,
    seed: u64,
    vocab_size: usize,
    embedding_dim: usize,
    max_seq_len: usize,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            train_pairs: PathBuf::from(DEFAULT_TRAIN_PAIRS),
            eval_triplets: PathBuf::from(DEFAULT_EVAL_TRIPLETS),
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            epochs: 3,
            learning_rate: 0.06,
            margin: 0.2,
            max_train_pairs: 250_000,
            seed: 17,
            vocab_size: 8_192,
            embedding_dim: 512,
            max_seq_len: 64,
        }
    }
}

impl CliArgs {
    fn parse(tokens: Vec<String>) -> Result<Self, String> {
        let mut args = Self::default();
        let mut index = 0usize;
        while index < tokens.len() {
            match tokens[index].as_str() {
                "--train-pairs" => {
                    index += 1;
                    args.train_pairs =
                        PathBuf::from(require_value(&tokens, index, "--train-pairs")?);
                }
                "--eval-triplets" => {
                    index += 1;
                    args.eval_triplets =
                        PathBuf::from(require_value(&tokens, index, "--eval-triplets")?);
                }
                "--output-dir" => {
                    index += 1;
                    args.output_dir = PathBuf::from(require_value(&tokens, index, "--output-dir")?);
                }
                "--epochs" => {
                    index += 1;
                    args.epochs = require_value(&tokens, index, "--epochs")?
                        .parse()
                        .map_err(|e| format!("invalid --epochs: {e}"))?;
                }
                "--learning-rate" => {
                    index += 1;
                    args.learning_rate = require_value(&tokens, index, "--learning-rate")?
                        .parse()
                        .map_err(|e| format!("invalid --learning-rate: {e}"))?;
                }
                "--margin" => {
                    index += 1;
                    args.margin = require_value(&tokens, index, "--margin")?
                        .parse()
                        .map_err(|e| format!("invalid --margin: {e}"))?;
                }
                "--max-train-pairs" => {
                    index += 1;
                    args.max_train_pairs = require_value(&tokens, index, "--max-train-pairs")?
                        .parse()
                        .map_err(|e| format!("invalid --max-train-pairs: {e}"))?;
                }
                "--seed" => {
                    index += 1;
                    args.seed = require_value(&tokens, index, "--seed")?
                        .parse()
                        .map_err(|e| format!("invalid --seed: {e}"))?;
                }
                "--vocab-size" => {
                    index += 1;
                    args.vocab_size = require_value(&tokens, index, "--vocab-size")?
                        .parse()
                        .map_err(|e| format!("invalid --vocab-size: {e}"))?;
                }
                "--embedding-dim" => {
                    index += 1;
                    args.embedding_dim = require_value(&tokens, index, "--embedding-dim")?
                        .parse()
                        .map_err(|e| format!("invalid --embedding-dim: {e}"))?;
                }
                "--max-seq-len" => {
                    index += 1;
                    args.max_seq_len = require_value(&tokens, index, "--max-seq-len")?
                        .parse()
                        .map_err(|e| format!("invalid --max-seq-len: {e}"))?;
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("Unknown argument: {other}")),
            }
            index += 1;
        }

        if args.embedding_dim == 0 {
            return Err("embedding_dim must be > 0".to_string());
        }
        if args.vocab_size <= 2 {
            return Err("vocab_size must be > 2".to_string());
        }
        if args.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".to_string());
        }

        Ok(args)
    }
}

fn require_value<'a>(tokens: &'a [String], index: usize, flag: &str) -> Result<&'a str, String> {
    tokens
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_help() {
    println!(
        "Usage:
  cargo run --bin train_rust_embedder -- [options]

Options:
  --train-pairs PATH         Train parquet with anchor/positive columns
  --eval-triplets PATH       Eval parquet with anchor/positive/negative columns
  --output-dir DIR           Output directory for model + metrics
  --epochs N                 Epoch count (default: 3)
  --learning-rate F          SGD learning rate (default: 0.06)
  --margin F                 Hinge margin (default: 0.2)
  --max-train-pairs N        Max train pairs to use (default: 250000)
  --seed N                   Random seed (default: 17)
  --vocab-size N             Hash vocab buckets (default: 8192)
  --embedding-dim N          Embedding dimensions (default: 512)
  --max-seq-len N            Max tokenized sequence length (default: 64)
  -h, --help                 Show this help
"
    );
}

#[derive(Debug, Clone, Serialize)]
struct TrainingReport {
    backend: String,
    train_pairs_used: usize,
    eval_triplets: usize,
    epochs: usize,
    learning_rate: f32,
    margin: f32,
    vocab_size: usize,
    embedding_dim: usize,
    max_seq_len: usize,
    seed: u64,
    metrics: RustEvalMetrics,
}

fn load_train_pairs(path: &PathBuf) -> Result<Vec<(String, String)>, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let frame = ParquetReader::new(file)
        .finish()
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let anchors = frame
        .column("anchor")
        .map_err(|e| format!("missing anchor column: {e}"))?
        .str()
        .map_err(|e| format!("anchor is not string: {e}"))?;
    let positives = frame
        .column("positive")
        .map_err(|e| format!("missing positive column: {e}"))?
        .str()
        .map_err(|e| format!("positive is not string: {e}"))?;

    let mut out = Vec::with_capacity(frame.height());
    for idx in 0..frame.height() {
        let (Some(anchor), Some(positive)) = (anchors.get(idx), positives.get(idx)) else {
            continue;
        };
        if anchor.is_empty() || positive.is_empty() {
            continue;
        }
        out.push((anchor.to_string(), positive.to_string()));
    }
    Ok(out)
}

fn load_eval_triplets(path: &PathBuf) -> Result<Vec<(String, String, String)>, String> {
    let file = File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let frame = ParquetReader::new(file)
        .finish()
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let anchors = frame
        .column("anchor")
        .map_err(|e| format!("missing anchor column: {e}"))?
        .str()
        .map_err(|e| format!("anchor is not string: {e}"))?;
    let positives = frame
        .column("positive")
        .map_err(|e| format!("missing positive column: {e}"))?
        .str()
        .map_err(|e| format!("positive is not string: {e}"))?;
    let negatives = frame
        .column("negative")
        .map_err(|e| format!("missing negative column: {e}"))?
        .str()
        .map_err(|e| format!("negative is not string: {e}"))?;

    let mut out = Vec::with_capacity(frame.height());
    for idx in 0..frame.height() {
        let (Some(anchor), Some(positive), Some(negative)) =
            (anchors.get(idx), positives.get(idx), negatives.get(idx))
        else {
            continue;
        };
        if anchor.is_empty() || positive.is_empty() || negative.is_empty() {
            continue;
        }
        out.push((
            anchor.to_string(),
            positive.to_string(),
            negative.to_string(),
        ));
    }
    Ok(out)
}

fn sample_in_place<T>(items: &mut Vec<T>, max_items: usize, seed: u64) {
    if max_items == 0 || items.len() <= max_items {
        return;
    }
    let mut rng = SmallRng::new(seed);
    if items.len() > 1 {
        for i in (1..items.len()).rev() {
            let j = rng.gen_range(0, i + 1);
            items.swap(i, j);
        }
    }
    items.truncate(max_items);
}
