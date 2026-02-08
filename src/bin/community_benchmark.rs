use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use auto_g_embed::TinyTextEmbedder;
use serde::Serialize;

const BOOK_PATH: &str = "test-data/pride-and-prejudice.txt";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(env::args().skip(1).collect())?;
    let passages = load_passages(args.window_size, args.stride, args.max_passages)?;
    let model = TinyTextEmbedder::default();

    let eval_texts = build_eval_texts(&passages, args.eval_count);
    for text in build_eval_texts(&passages, args.warmup_count) {
        black_box(model.embed(&text));
    }

    let wall_start = Instant::now();
    let mut latencies_micros = Vec::with_capacity(eval_texts.len());
    for text in &eval_texts {
        let start = Instant::now();
        black_box(model.embed(text));
        latencies_micros.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    let elapsed_secs = wall_start.elapsed().as_secs_f64();
    let embeds_per_second = (args.eval_count as f64) / elapsed_secs.max(f64::EPSILON);

    let latency = latency_stats(latencies_micros);
    let quality = quality_metrics(&model, &passages, args.query_count);
    let dataset = dataset_info(&passages);
    let report = BenchmarkReport {
        benchmark: BenchmarkMeta {
            name: "auto-g-embed-community-benchmark".to_string(),
            unix_timestamp_sec: unix_timestamp_now(),
            eval_count: args.eval_count,
            warmup_count: args.warmup_count,
            query_count: args.query_count.min(passages.len()),
            window_size: args.window_size,
            stride: args.stride,
            max_passages: args.max_passages,
            corpus_file: BOOK_PATH.to_string(),
            command: env::args().collect::<Vec<_>>().join(" "),
        },
        environment: collect_environment(),
        dataset,
        throughput: Throughput {
            elapsed_ms: elapsed_secs * 1000.0,
            embeds_per_second,
        },
        latency,
        quality,
    };

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("failed to serialize benchmark report: {e}"))?;
    fs::write(&args.output, json)
        .map_err(|e| format!("failed to write {}: {e}", args.output.display()))?;

    println!("wrote benchmark report: {}", args.output.display());
    println!(
        "throughput embeds_per_sec={:.2} p50_us={:.2} p95_us={:.2} p99_us={:.2}",
        report.throughput.embeds_per_second,
        report.latency.p50_us,
        report.latency.p95_us,
        report.latency.p99_us
    );
    println!(
        "quality top1_accuracy={:.3} separation={:.3}",
        report.quality.top1_accuracy, report.quality.separation
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct CliArgs {
    output: PathBuf,
    eval_count: usize,
    warmup_count: usize,
    query_count: usize,
    window_size: usize,
    stride: usize,
    max_passages: usize,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            output: PathBuf::from("artifacts/benchmarks/latest.json"),
            eval_count: 10_000,
            warmup_count: 1_000,
            query_count: 64,
            window_size: 80,
            stride: 40,
            max_passages: 180,
        }
    }
}

impl CliArgs {
    fn parse(tokens: Vec<String>) -> Result<Self, String> {
        let mut args = Self::default();
        let mut index = 0usize;
        while index < tokens.len() {
            match tokens[index].as_str() {
                "--output" => {
                    index += 1;
                    args.output = PathBuf::from(require_value(&tokens, index, "--output")?);
                }
                "--eval-count" => {
                    index += 1;
                    args.eval_count = parse_usize(&tokens, index, "--eval-count")?;
                }
                "--warmup-count" => {
                    index += 1;
                    args.warmup_count = parse_usize(&tokens, index, "--warmup-count")?;
                }
                "--query-count" => {
                    index += 1;
                    args.query_count = parse_usize(&tokens, index, "--query-count")?;
                }
                "--window-size" => {
                    index += 1;
                    args.window_size = parse_usize(&tokens, index, "--window-size")?;
                }
                "--stride" => {
                    index += 1;
                    args.stride = parse_usize(&tokens, index, "--stride")?;
                }
                "--max-passages" => {
                    index += 1;
                    args.max_passages = parse_usize(&tokens, index, "--max-passages")?;
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
            index += 1;
        }

        if args.eval_count == 0 {
            return Err("--eval-count must be > 0".to_string());
        }
        if args.warmup_count == 0 {
            return Err("--warmup-count must be > 0".to_string());
        }
        if args.query_count == 0 {
            return Err("--query-count must be > 0".to_string());
        }
        if args.window_size == 0 {
            return Err("--window-size must be > 0".to_string());
        }
        if args.stride == 0 {
            return Err("--stride must be > 0".to_string());
        }
        if args.max_passages < 60 {
            return Err("--max-passages must be >= 60".to_string());
        }

        Ok(args)
    }
}

fn print_help() {
    println!("Usage: cargo run --release --bin community_benchmark -- [options]");
    println!();
    println!("Options:");
    println!(
        "  --output <path>         Output JSON path (default: artifacts/benchmarks/latest.json)"
    );
    println!("  --eval-count <n>        Number of timed embeddings (default: 10000)");
    println!("  --warmup-count <n>      Warmup embeddings (default: 1000)");
    println!("  --query-count <n>       Query count for retrieval quality (default: 64)");
    println!("  --window-size <n>       Passage window size in words (default: 80)");
    println!("  --stride <n>            Passage stride in words (default: 40)");
    println!("  --max-passages <n>      Maximum number of passages to evaluate (default: 180)");
}

fn require_value<'a>(tokens: &'a [String], index: usize, flag: &str) -> Result<&'a str, String> {
    tokens
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_usize(tokens: &[String], index: usize, flag: &str) -> Result<usize, String> {
    require_value(tokens, index, flag)?
        .parse::<usize>()
        .map_err(|e| format!("invalid {flag}: {e}"))
}

fn load_passages(
    window_size: usize,
    stride: usize,
    max_windows: usize,
) -> Result<Vec<String>, String> {
    let raw =
        fs::read_to_string(BOOK_PATH).map_err(|e| format!("failed to read {BOOK_PATH}: {e}"))?;
    let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
    let content = strip_gutenberg_wrapper(&normalized);
    let passages = build_sliding_word_windows(content, window_size, stride, max_windows);
    if passages.len() < 60 {
        return Err(format!(
            "need at least 60 passages from {BOOK_PATH}, found {}",
            passages.len()
        ));
    }
    Ok(passages)
}

fn strip_gutenberg_wrapper(raw: &str) -> &str {
    let start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK";
    let end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK";

    let start = raw
        .find(start_marker)
        .and_then(|idx| raw[idx..].find('\n').map(|line_end| idx + line_end + 1))
        .unwrap_or(0);
    let end = raw.find(end_marker).unwrap_or(raw.len());
    &raw[start..end]
}

fn build_sliding_word_windows(
    text: &str,
    window_size: usize,
    stride: usize,
    max_windows: usize,
) -> Vec<String> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < window_size {
        return Vec::new();
    }

    let mut windows = Vec::with_capacity(max_windows);
    let mut start = 0usize;
    while start + window_size <= tokens.len() && windows.len() < max_windows {
        windows.push(tokens[start..start + window_size].join(" "));
        start += stride;
    }
    windows
}

fn build_eval_texts(passages: &[String], count: usize) -> Vec<String> {
    let mut eval_texts = Vec::with_capacity(count);
    for i in 0..count {
        eval_texts.push(passages[i % passages.len()].clone());
    }
    eval_texts
}

fn latency_stats(mut latencies_micros: Vec<f64>) -> LatencyStats {
    latencies_micros.sort_by(f64::total_cmp);
    let count = latencies_micros.len();
    let mean = latencies_micros.iter().sum::<f64>() / count as f64;
    LatencyStats {
        mean_us: mean,
        p50_us: percentile(&latencies_micros, 0.50),
        p95_us: percentile(&latencies_micros, 0.95),
        p99_us: percentile(&latencies_micros, 0.99),
        min_us: *latencies_micros.first().unwrap_or(&0.0),
        max_us: *latencies_micros.last().unwrap_or(&0.0),
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn quality_metrics(
    model: &TinyTextEmbedder,
    passages: &[String],
    query_count: usize,
) -> QualityStats {
    let corpus_embeddings: Vec<Vec<f32>> = passages.iter().map(|p| model.embed(p)).collect();
    let eval_count = query_count.min(passages.len());

    let mut top1_hits = 0usize;
    let mut positives = Vec::with_capacity(eval_count);
    let mut negatives = Vec::with_capacity(eval_count);

    for i in 0..eval_count {
        let query = build_query_from_passage(&passages[i]);
        let query_embedding = model.embed(&query);

        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (j, embedding) in corpus_embeddings.iter().enumerate() {
            let score = cosine_similarity(&query_embedding, embedding);
            if score > best_score {
                best_score = score;
                best_idx = j;
            }
        }

        if best_idx == i {
            top1_hits += 1;
        }

        positives.push(cosine_similarity(&query_embedding, &corpus_embeddings[i]));
        let negative_idx = (i + 23) % corpus_embeddings.len();
        negatives.push(cosine_similarity(
            &query_embedding,
            &corpus_embeddings[negative_idx],
        ));
    }

    let top1_accuracy = top1_hits as f64 / eval_count as f64;
    let mean_positive = mean_f32(&positives) as f64;
    let mean_negative = mean_f32(&negatives) as f64;
    QualityStats {
        eval_count,
        top1_accuracy,
        mean_positive_similarity: mean_positive,
        mean_negative_similarity: mean_negative,
        separation: mean_positive - mean_negative,
    }
}

fn build_query_from_passage(passage: &str) -> String {
    let tokens: Vec<String> = passage
        .split_whitespace()
        .map(clean_ascii_token)
        .filter(|token| token.len() >= 3)
        .collect();

    if tokens.is_empty() {
        return "empty".to_string();
    }

    tokens
        .iter()
        .step_by(3)
        .take(32)
        .map(|token| token.to_ascii_uppercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn clean_ascii_token(token: &str) -> String {
    token
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn mean_f32(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn dataset_info(passages: &[String]) -> DatasetInfo {
    let passage_count = passages.len();
    let total_chars: usize = passages.iter().map(String::len).sum();
    let total_words: usize = passages.iter().map(|p| p.split_whitespace().count()).sum();

    DatasetInfo {
        passage_count,
        mean_chars_per_passage: total_chars as f64 / passage_count as f64,
        mean_words_per_passage: total_words as f64 / passage_count as f64,
    }
}

fn collect_environment() -> Environment {
    Environment {
        os: env::consts::OS.to_string(),
        arch: env::consts::ARCH.to_string(),
        cpu: detect_cpu_model(),
        logical_cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        rustc: read_command_output("rustc", &["--version"]),
        git_commit: read_command_output("git", &["rev-parse", "HEAD"]),
    }
}

fn detect_cpu_model() -> Option<String> {
    if cfg!(target_os = "macos") {
        return read_command_output("sysctl", &["-n", "machdep.cpu.brand_string"]);
    }
    if cfg!(target_os = "linux")
        && let Ok(contents) = fs::read_to_string("/proc/cpuinfo")
    {
        for line in contents.lines() {
            if let Some(rest) = line.strip_prefix("model name\t: ") {
                return Some(rest.trim().to_string());
            }
        }
    }
    None
}

fn read_command_output(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn unix_timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    benchmark: BenchmarkMeta,
    environment: Environment,
    dataset: DatasetInfo,
    throughput: Throughput,
    latency: LatencyStats,
    quality: QualityStats,
}

#[derive(Debug, Serialize)]
struct BenchmarkMeta {
    name: String,
    unix_timestamp_sec: u64,
    eval_count: usize,
    warmup_count: usize,
    query_count: usize,
    window_size: usize,
    stride: usize,
    max_passages: usize,
    corpus_file: String,
    command: String,
}

#[derive(Debug, Serialize)]
struct Environment {
    os: String,
    arch: String,
    cpu: Option<String>,
    logical_cpus: usize,
    rustc: Option<String>,
    git_commit: Option<String>,
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    passage_count: usize,
    mean_chars_per_passage: f64,
    mean_words_per_passage: f64,
}

#[derive(Debug, Serialize)]
struct Throughput {
    elapsed_ms: f64,
    embeds_per_second: f64,
}

#[derive(Debug, Serialize)]
struct LatencyStats {
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Debug, Serialize)]
struct QualityStats {
    eval_count: usize,
    top1_accuracy: f64,
    mean_positive_similarity: f64,
    mean_negative_similarity: f64,
    separation: f64,
}
