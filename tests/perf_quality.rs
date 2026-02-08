use std::fs;
use std::hint::black_box;
use std::time::Instant;

use auto_g_embed::TinyTextEmbedder;

const BOOK_PATH: &str = "test-data/pride-and-prejudice.txt";

#[test]
fn embedding_performance_on_gutenberg_corpus() {
    let passages = load_passages();
    let model = TinyTextEmbedder::default();

    let eval_count = 10_000usize;
    let mut eval_texts = Vec::with_capacity(eval_count);
    for i in 0..eval_count {
        eval_texts.push(passages[i % passages.len()].as_str());
    }

    let start = Instant::now();
    for text in &eval_texts {
        black_box(model.embed(text));
    }
    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let embeds_per_sec = (eval_count as f64) / elapsed_secs.max(f64::EPSILON);

    eprintln!(
        "perf metrics: eval_count={eval_count}, elapsed_ms={:.2}, embeds_per_sec={:.2}",
        elapsed_secs * 1000.0,
        embeds_per_sec
    );

    assert!(
        embeds_per_sec > 300.0,
        "Embedding throughput unexpectedly low: {embeds_per_sec:.2} embeds/sec"
    );
}

#[test]
fn embedding_quality_on_gutenberg_corpus() {
    let passages = load_passages();
    let model = TinyTextEmbedder::default();

    let corpus_embeddings: Vec<Vec<f32>> = passages.iter().map(|p| model.embed(p)).collect();
    let eval_count = passages.len().min(48);

    let mut top1_hits = 0usize;
    let mut positive_scores = Vec::with_capacity(eval_count);
    let mut negative_scores = Vec::with_capacity(eval_count);

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

        positive_scores.push(cosine_similarity(&query_embedding, &corpus_embeddings[i]));
        let negative_idx = (i + 23) % corpus_embeddings.len();
        negative_scores.push(cosine_similarity(
            &query_embedding,
            &corpus_embeddings[negative_idx],
        ));
    }

    let top1_accuracy = top1_hits as f32 / eval_count as f32;
    let mean_positive = mean(&positive_scores);
    let mean_negative = mean(&negative_scores);
    let separation = mean_positive - mean_negative;

    eprintln!(
        "quality metrics: eval_count={eval_count}, top1_accuracy={top1_accuracy:.3}, mean_positive={mean_positive:.3}, mean_negative={mean_negative:.3}, separation={separation:.3}"
    );

    assert!(
        top1_accuracy >= 0.60,
        "Top-1 retrieval accuracy too low: {top1_accuracy:.3}"
    );
    assert!(
        separation >= 0.12,
        "Positive/negative similarity separation too small: {separation:.3}"
    );
}

fn load_passages() -> Vec<String> {
    let raw = fs::read_to_string(BOOK_PATH)
        .unwrap_or_else(|err| panic!("Failed to read {BOOK_PATH}: {err}"));
    let normalized = normalize_line_endings(&raw);
    let book = strip_gutenberg_wrapper(&normalized);
    let passages = build_sliding_word_windows(book, 80, 40, 180);

    assert!(
        passages.len() >= 60,
        "Need at least 60 passages from {BOOK_PATH}, found {}",
        passages.len()
    );

    passages
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

fn normalize_line_endings(input: &str) -> String {
    input.replace("\r\n", "\n").replace('\r', "\n")
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
        .map(|t| t.to_ascii_uppercase())
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

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}
