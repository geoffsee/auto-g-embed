use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::ffi::OsStr;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use csv::StringRecord;
use polars::io::parquet::write::ParquetWriter;
use polars::prelude::{DataFrame, NamedFrom, PolarsResult, Series};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const DEFAULT_PROFILES_FILE: &str = "training/dataset_profiles.json";
const DEFAULT_PROFILE_NAME: &str = "quora_pairs";
const DEFAULT_OUT_DIR: &str = "artifacts/contrastive-data";

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(env::args().skip(1).collect())?;
    let profiles = load_profiles(&args.profiles_file)?;

    if args.list_profiles {
        for (name, profile) in &profiles {
            println!(
                "{name}\t{}\t{}",
                profile.kind,
                profile.dataset_ref.as_deref().unwrap_or("-")
            );
        }
        return Ok(());
    }

    let profile = profiles
        .get(&args.profile)
        .ok_or_else(|| format!("Unknown profile '{}'", args.profile))?;

    let dataset_ref = args
        .dataset_ref
        .clone()
        .or_else(|| profile.dataset_ref.clone());
    let csv_name = args.csv_name.clone().or_else(|| profile.csv_name.clone());

    let min_chars = args
        .min_chars
        .unwrap_or(profile.filters.min_chars.unwrap_or(12));
    let min_words = args
        .min_words
        .unwrap_or(profile.filters.min_words.unwrap_or(3));
    let require_question_mark = profile.filters.require_question_mark.unwrap_or(false);
    let disallow_values = profile
        .filters
        .disallow_values
        .iter()
        .map(|v| v.trim().to_ascii_lowercase())
        .filter(|v| !v.is_empty())
        .collect::<HashSet<_>>();

    let csv_path = if let Some(path) = args.source_csv.clone() {
        path
    } else {
        let dataset_ref = dataset_ref
            .clone()
            .ok_or_else(|| format!("profile '{}' has no dataset_ref", profile.name))?;
        let root = dataset_root_from_kaggle(&dataset_ref)?;
        find_csv(&root, csv_name.as_deref())?
    };

    if !csv_path.exists() {
        return Err(format!("CSV file not found: {}", csv_path.display()));
    }

    println!(
        "[prepare] profile={} kind={} source_csv={}",
        profile.name,
        profile.kind,
        csv_path.display()
    );
    println!("[prepare] min_chars={min_chars} min_words={min_words}");

    let dataset = ingest_dataset(
        &csv_path,
        profile,
        min_chars,
        min_words,
        require_question_mark,
        &disallow_values,
    )?;

    let mut positives = dataset.positives;
    sample_in_place(&mut positives, args.max_train_pairs, args.seed);

    let eval_triplets = make_eval_triplets(
        &positives,
        &dataset.negative_pool,
        args.max_eval_triplets,
        args.seed,
    )?;

    fs::create_dir_all(&args.out_dir).map_err(|e| {
        format!(
            "failed to create output directory {}: {e}",
            args.out_dir.display()
        )
    })?;
    let train_path = args.out_dir.join("train_pairs.parquet");
    let eval_path = args.out_dir.join("eval_triplets.parquet");
    let stats_path = args.out_dir.join("metadata.json");

    write_train_pairs(&train_path, &positives).map_err(|e| e.to_string())?;
    write_eval_triplets(&eval_path, &eval_triplets).map_err(|e| e.to_string())?;

    let metadata = Metadata {
        profile: profile.name.clone(),
        profile_kind: profile.kind.clone(),
        dataset_ref: if args.source_csv.is_some() {
            "local_csv".to_string()
        } else {
            dataset_ref.unwrap_or_else(|| "local_csv".to_string())
        },
        source_csv: csv_path.display().to_string(),
        rows_after_cleaning: dataset.rows_after_cleaning,
        positive_pairs: dataset.positive_pairs_before_sampling,
        negative_pool_size: dataset.negative_pool.len(),
        train_pairs: positives.len(),
        eval_triplets: eval_triplets.len(),
        max_train_pairs: args.max_train_pairs,
        max_eval_triplets: args.max_eval_triplets,
        min_chars,
        min_words,
        seed: args.seed,
    };

    let metadata_text =
        serde_json::to_string_pretty(&metadata).map_err(|e| format!("json encode failed: {e}"))?;
    fs::write(&stats_path, metadata_text).map_err(|e| {
        format!(
            "failed to write metadata file {}: {e}",
            stats_path.display()
        )
    })?;

    println!("[prepare] wrote {}", train_path.display());
    println!("[prepare] wrote {}", eval_path.display());
    println!("[prepare] wrote {}", stats_path.display());
    println!(
        "[prepare] train_pairs={} eval_triplets={}",
        positives.len(),
        eval_triplets.len()
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct CliArgs {
    profiles_file: PathBuf,
    list_profiles: bool,
    profile: String,
    dataset_ref: Option<String>,
    source_csv: Option<PathBuf>,
    csv_name: Option<String>,
    min_chars: Option<usize>,
    min_words: Option<usize>,
    max_train_pairs: usize,
    max_eval_triplets: usize,
    seed: u64,
    out_dir: PathBuf,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            profiles_file: PathBuf::from(DEFAULT_PROFILES_FILE),
            list_profiles: false,
            profile: DEFAULT_PROFILE_NAME.to_string(),
            dataset_ref: None,
            source_csv: None,
            csv_name: None,
            min_chars: None,
            min_words: None,
            max_train_pairs: 300_000,
            max_eval_triplets: 10_000,
            seed: 17,
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
        }
    }
}

impl CliArgs {
    fn parse(tokens: Vec<String>) -> Result<Self, String> {
        let mut args = Self::default();
        let mut index = 0usize;

        while index < tokens.len() {
            let flag = tokens[index].as_str();
            match flag {
                "--profiles-file" => {
                    index += 1;
                    args.profiles_file =
                        PathBuf::from(require_value(&tokens, index, "--profiles-file")?);
                }
                "--list-profiles" => {
                    args.list_profiles = true;
                }
                "--profile" => {
                    index += 1;
                    args.profile = require_value(&tokens, index, "--profile")?.to_string();
                }
                "--dataset-ref" => {
                    index += 1;
                    args.dataset_ref =
                        Some(require_value(&tokens, index, "--dataset-ref")?.to_string());
                }
                "--source-csv" => {
                    index += 1;
                    args.source_csv = Some(PathBuf::from(require_value(
                        &tokens,
                        index,
                        "--source-csv",
                    )?));
                }
                "--csv-name" => {
                    index += 1;
                    args.csv_name = Some(require_value(&tokens, index, "--csv-name")?.to_string());
                }
                "--min-chars" => {
                    index += 1;
                    args.min_chars = Some(
                        require_value(&tokens, index, "--min-chars")?
                            .parse()
                            .map_err(|e| format!("invalid --min-chars: {e}"))?,
                    );
                }
                "--min-words" => {
                    index += 1;
                    args.min_words = Some(
                        require_value(&tokens, index, "--min-words")?
                            .parse()
                            .map_err(|e| format!("invalid --min-words: {e}"))?,
                    );
                }
                "--max-train-pairs" => {
                    index += 1;
                    args.max_train_pairs = require_value(&tokens, index, "--max-train-pairs")?
                        .parse()
                        .map_err(|e| format!("invalid --max-train-pairs: {e}"))?;
                }
                "--max-eval-triplets" => {
                    index += 1;
                    args.max_eval_triplets = require_value(&tokens, index, "--max-eval-triplets")?
                        .parse()
                        .map_err(|e| format!("invalid --max-eval-triplets: {e}"))?;
                }
                "--seed" => {
                    index += 1;
                    args.seed = require_value(&tokens, index, "--seed")?
                        .parse()
                        .map_err(|e| format!("invalid --seed: {e}"))?;
                }
                "--out-dir" => {
                    index += 1;
                    args.out_dir = PathBuf::from(require_value(&tokens, index, "--out-dir")?);
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    return Err(format!("Unknown argument: {flag}"));
                }
            }
            index += 1;
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
  cargo run --bin prepare_contrastive -- [options]

Options:
  --profiles-file PATH       Profile JSON file (default: {DEFAULT_PROFILES_FILE})
  --list-profiles            List available profiles and exit
  --profile NAME             Profile name (default: {DEFAULT_PROFILE_NAME})
  --dataset-ref REF          Kaggle dataset ref override
  --source-csv PATH          Local CSV path override
  --csv-name NAME            CSV filename inside downloaded dataset
  --min-chars N              Override minimum chars filter
  --min-words N              Override minimum words filter
  --max-train-pairs N        Maximum train pairs (default: 300000)
  --max-eval-triplets N      Maximum eval triplets (default: 10000)
  --seed N                   Random seed (default: 17)
  --out-dir DIR              Output directory (default: {DEFAULT_OUT_DIR})
  -h, --help                 Show this help
"
    );
}

#[derive(Debug, Deserialize)]
struct ProfileRoot {
    profiles: BTreeMap<String, DatasetProfile>,
}

#[derive(Debug, Deserialize, Clone)]
struct DatasetProfile {
    #[serde(default)]
    name: String,
    kind: String,
    dataset_ref: Option<String>,
    csv_name: Option<String>,
    #[serde(default)]
    columns: HashMap<String, String>,
    #[serde(default)]
    filters: DatasetFilters,
    #[serde(default)]
    options: HashMap<String, Value>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct DatasetFilters {
    min_chars: Option<usize>,
    min_words: Option<usize>,
    require_question_mark: Option<bool>,
    #[serde(default)]
    disallow_values: Vec<String>,
}

#[derive(Debug)]
struct IngestResult {
    positives: Vec<(String, String)>,
    negative_pool: Vec<String>,
    rows_after_cleaning: usize,
    positive_pairs_before_sampling: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Metadata {
    profile: String,
    profile_kind: String,
    dataset_ref: String,
    source_csv: String,
    rows_after_cleaning: usize,
    positive_pairs: usize,
    negative_pool_size: usize,
    train_pairs: usize,
    eval_triplets: usize,
    max_train_pairs: usize,
    max_eval_triplets: usize,
    min_chars: usize,
    min_words: usize,
    seed: u64,
}

fn load_profiles(path: &Path) -> Result<BTreeMap<String, DatasetProfile>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("failed to read profiles file {}: {e}", path.display()))?;
    let mut parsed: ProfileRoot =
        serde_json::from_str(&text).map_err(|e| format!("invalid profiles JSON: {e}"))?;
    for (name, profile) in &mut parsed.profiles {
        if profile.name.is_empty() {
            profile.name = name.clone();
        }
    }
    Ok(parsed.profiles)
}

fn ingest_dataset(
    csv_path: &Path,
    profile: &DatasetProfile,
    min_chars: usize,
    min_words: usize,
    require_question_mark: bool,
    disallow_values: &HashSet<String>,
) -> Result<IngestResult, String> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(csv_path)
        .map_err(|e| format!("failed to open CSV {}: {e}", csv_path.display()))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("failed to read CSV headers: {e}"))?
        .clone();

    let result = match profile.kind.as_str() {
        "pair_binary_label" => ingest_pair_binary_label(
            &mut reader,
            &headers,
            profile,
            min_chars,
            min_words,
            disallow_values,
        )?,
        "single_text_duplicate_mining" => ingest_single_text_duplicate_mining(
            &mut reader,
            &headers,
            profile,
            min_chars,
            min_words,
            require_question_mark,
            disallow_values,
        )?,
        other => {
            return Err(format!(
                "Unsupported profile kind '{other}' in profile '{}'",
                profile.name
            ));
        }
    };

    Ok(result)
}

fn ingest_pair_binary_label(
    reader: &mut csv::Reader<File>,
    headers: &StringRecord,
    profile: &DatasetProfile,
    min_chars: usize,
    min_words: usize,
    disallow_values: &HashSet<String>,
) -> Result<IngestResult, String> {
    let left_col = profile
        .columns
        .get("left")
        .ok_or_else(|| format!("profile '{}' missing columns.left", profile.name))?;
    let right_col = profile
        .columns
        .get("right")
        .ok_or_else(|| format!("profile '{}' missing columns.right", profile.name))?;
    let label_col = profile
        .columns
        .get("label")
        .ok_or_else(|| format!("profile '{}' missing columns.label", profile.name))?;

    let left_idx = find_col_idx(headers, left_col)?;
    let right_idx = find_col_idx(headers, right_col)?;
    let label_idx = find_col_idx(headers, label_col)?;
    let positive_label = profile
        .options
        .get("positive_label")
        .and_then(Value::as_i64)
        .unwrap_or(1);

    let mut positive_set = HashSet::<(String, String)>::new();
    let mut negative_pool = HashSet::<String>::new();
    let mut rows_after_cleaning = 0usize;

    for row in reader.records() {
        let row = row.map_err(|e| format!("CSV row read failed: {e}"))?;
        let left_raw = row.get(left_idx).unwrap_or("");
        let right_raw = row.get(right_idx).unwrap_or("");

        let left_clean = clean_text(left_raw);
        let right_clean = clean_text(right_raw);
        if left_clean.is_empty() || right_clean.is_empty() {
            continue;
        }
        rows_after_cleaning += 1;

        if !passes_text_filters(
            left_raw,
            &left_clean,
            min_chars,
            min_words,
            false,
            disallow_values,
        ) {
            continue;
        }
        if !passes_text_filters(
            right_raw,
            &right_clean,
            min_chars,
            min_words,
            false,
            disallow_values,
        ) {
            continue;
        }

        let parsed_label = row
            .get(label_idx)
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(0);

        if parsed_label == positive_label {
            let (anchor, positive) = canonical_pair(left_clean, right_clean);
            if anchor != positive {
                positive_set.insert((anchor, positive));
            }
        } else if left_clean != right_clean {
            negative_pool.insert(right_clean);
        }
    }

    let positives = positive_set.into_iter().collect::<Vec<_>>();
    let negative_pool = negative_pool.into_iter().collect::<Vec<_>>();

    Ok(IngestResult {
        positive_pairs_before_sampling: positives.len(),
        positives,
        negative_pool,
        rows_after_cleaning,
    })
}

fn ingest_single_text_duplicate_mining(
    reader: &mut csv::Reader<File>,
    headers: &StringRecord,
    profile: &DatasetProfile,
    min_chars: usize,
    min_words: usize,
    require_question_mark: bool,
    disallow_values: &HashSet<String>,
) -> Result<IngestResult, String> {
    let text_col = profile
        .columns
        .get("text")
        .ok_or_else(|| format!("profile '{}' missing columns.text", profile.name))?;
    let text_idx = find_col_idx(headers, text_col)?;

    let mut rows_after_cleaning = 0usize;
    let mut accepted = Vec::<String>::new();

    for row in reader.records() {
        let row = row.map_err(|e| format!("CSV row read failed: {e}"))?;
        let raw = row.get(text_idx).unwrap_or("");
        let clean = clean_text(raw);
        if clean.is_empty() {
            continue;
        }
        rows_after_cleaning += 1;

        if !passes_text_filters(
            raw,
            &clean,
            min_chars,
            min_words,
            require_question_mark,
            disallow_values,
        ) {
            continue;
        }
        accepted.push(clean);
    }

    let mut counts = HashMap::<String, usize>::new();
    for text in &accepted {
        *counts.entry(text.clone()).or_insert(0) += 1;
    }

    let mut positives = Vec::<(String, String)>::new();
    for (text, count) in &counts {
        if *count > 1 {
            for _ in 0..(*count - 1) {
                positives.push((text.clone(), text.clone()));
            }
        }
    }

    let negative_pool = counts.keys().cloned().collect::<Vec<_>>();

    Ok(IngestResult {
        positive_pairs_before_sampling: positives.len(),
        positives,
        negative_pool,
        rows_after_cleaning,
    })
}

fn find_col_idx(headers: &StringRecord, name: &str) -> Result<usize, String> {
    headers
        .iter()
        .position(|h| h == name)
        .ok_or_else(|| format!("column '{name}' not found in CSV headers"))
}

fn canonical_pair(left: String, right: String) -> (String, String) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn passes_text_filters(
    raw_text: &str,
    clean_text: &str,
    min_chars: usize,
    min_words: usize,
    require_question_mark: bool,
    disallow_values: &HashSet<String>,
) -> bool {
    if disallow_values.contains(clean_text) {
        return false;
    }
    if clean_text.chars().count() < min_chars {
        return false;
    }
    if clean_text.split_whitespace().count() < min_words {
        return false;
    }
    if require_question_mark && !raw_text.trim_end().ends_with('?') {
        return false;
    }
    true
}

fn clean_text(input: &str) -> String {
    let mut flattened = String::with_capacity(input.len());
    for token in input.split_whitespace() {
        let lowered = token.to_ascii_lowercase();
        if lowered.starts_with("http://") || lowered.starts_with("https://") {
            continue;
        }
        for ch in lowered.chars() {
            if ch.is_ascii_alphanumeric() {
                flattened.push(ch);
            } else {
                flattened.push(' ');
            }
        }
        flattened.push(' ');
    }
    flattened.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn make_eval_triplets(
    positives: &[(String, String)],
    negatives: &[String],
    max_eval_triplets: usize,
    seed: u64,
) -> Result<Vec<(String, String, String)>, String> {
    if positives.is_empty() {
        return Err("No positive pairs produced; cannot build contrastive data".to_string());
    }
    if negatives.is_empty() {
        return Err("No negatives produced; cannot build eval triplets".to_string());
    }

    let mut anchor_pos = positives.to_vec();
    sample_in_place(&mut anchor_pos, max_eval_triplets, seed.wrapping_add(7));

    let mut rng = SmallRng::new(seed.wrapping_add(11));
    let mut triplets = Vec::with_capacity(anchor_pos.len());
    for (anchor, positive) in anchor_pos {
        if negatives.is_empty() {
            break;
        }
        let mut chosen = None;
        for _ in 0..32 {
            let idx = rng.gen_range(0, negatives.len());
            let candidate = &negatives[idx];
            if candidate != &anchor && candidate != &positive {
                chosen = Some(candidate.clone());
                break;
            }
        }
        if let Some(negative) = chosen {
            triplets.push((anchor, positive, negative));
        }
    }

    if triplets.is_empty() {
        return Err("Failed to sample non-colliding negatives for eval triplets".to_string());
    }

    Ok(triplets)
}

fn sample_in_place<T>(items: &mut Vec<T>, max_items: usize, seed: u64) {
    if max_items == 0 || items.len() <= max_items {
        return;
    }
    let mut rng = SmallRng::new(seed);
    fisher_yates_shuffle(items.as_mut_slice(), &mut rng);
    items.truncate(max_items);
}

fn fisher_yates_shuffle<T>(slice: &mut [T], rng: &mut SmallRng) {
    if slice.len() < 2 {
        return;
    }
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(0, i + 1);
        slice.swap(i, j);
    }
}

fn write_train_pairs(path: &Path, pairs: &[(String, String)]) -> PolarsResult<()> {
    let anchors = pairs.iter().map(|(a, _)| a.clone()).collect::<Vec<_>>();
    let positives = pairs.iter().map(|(_, p)| p.clone()).collect::<Vec<_>>();
    let mut frame = DataFrame::new(vec![
        Series::new("anchor".into(), anchors).into(),
        Series::new("positive".into(), positives).into(),
    ])?;

    let mut file = File::create(path)?;
    ParquetWriter::new(&mut file).finish(&mut frame)?;
    Ok(())
}

fn write_eval_triplets(path: &Path, triplets: &[(String, String, String)]) -> PolarsResult<()> {
    let anchors = triplets
        .iter()
        .map(|(a, _, _)| a.clone())
        .collect::<Vec<_>>();
    let positives = triplets
        .iter()
        .map(|(_, p, _)| p.clone())
        .collect::<Vec<_>>();
    let negatives = triplets
        .iter()
        .map(|(_, _, n)| n.clone())
        .collect::<Vec<_>>();

    let mut frame = DataFrame::new(vec![
        Series::new("anchor".into(), anchors).into(),
        Series::new("positive".into(), positives).into(),
        Series::new("negative".into(), negatives).into(),
    ])?;

    let mut file = File::create(path)?;
    ParquetWriter::new(&mut file).finish(&mut frame)?;
    Ok(())
}

fn dataset_root_from_kaggle(dataset_ref: &str) -> Result<PathBuf, String> {
    let dataset_slug = dataset_ref.replace('/', "_");
    let output_dir = PathBuf::from("data/kaggle").join(dataset_slug);
    fs::create_dir_all(&output_dir).map_err(|e| {
        format!(
            "failed to create Kaggle output directory {}: {e}",
            output_dir.display()
        )
    })?;

    let status = Command::new("kaggle")
        .args(["datasets", "download", "-d", dataset_ref, "--unzip"])
        .arg("-p")
        .arg(&output_dir)
        .status()
        .map_err(|e| format!("failed to invoke kaggle CLI: {e}"))?;

    if !status.success() {
        return Err(format!(
            "kaggle CLI download failed for dataset '{dataset_ref}'"
        ));
    }
    Ok(output_dir)
}

fn find_csv(root: &Path, csv_name: Option<&str>) -> Result<PathBuf, String> {
    if let Some(name) = csv_name {
        let maybe = walk_files(root)
            .into_iter()
            .find(|path| path.file_name() == Some(OsStr::new(name)));
        if let Some(path) = maybe {
            return Ok(path);
        }
    }

    walk_files(root)
        .into_iter()
        .find(|path| path.extension() == Some(OsStr::new("csv")))
        .ok_or_else(|| format!("No CSV file found under {}", root.display()))
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();

    while let Some(path) = stack.pop() {
        let entries = match fs::read_dir(&path) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                stack.push(child);
            } else if child.is_file() {
                files.push(child);
            }
        }
    }

    files.sort();
    files
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
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn gen_range(&mut self, start: usize, end: usize) -> usize {
        if end <= start {
            return start;
        }
        let span = end - start;
        (self.next_u64() as usize % span) + start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_strips_urls_and_punctuation() {
        let cleaned = clean_text("Hello, WORLD! https://example.com/test?x=1");
        assert_eq!(cleaned, "hello world");
    }

    #[test]
    fn clean_text_normalizes_spaces() {
        let cleaned = clean_text(" A   b\tc\n\rd ");
        assert_eq!(cleaned, "a b c d");
    }

    #[test]
    fn text_filters_enforce_question_mark() {
        let empty = HashSet::new();
        assert!(passes_text_filters(
            "What now?",
            "what now",
            3,
            2,
            true,
            &empty
        ));
        assert!(!passes_text_filters(
            "What now", "what now", 3, 2, true, &empty
        ));
    }

    #[test]
    fn sampler_truncates_deterministically() {
        let mut first = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut second = vec![1, 2, 3, 4, 5, 6, 7, 8];
        sample_in_place(&mut first, 3, 11);
        sample_in_place(&mut second, 3, 11);

        assert_eq!(first.len(), 3);
        assert_eq!(second.len(), 3);
        assert_eq!(first, second);
    }
}
