//! Export a trained RustContrastiveEmbedder to the compact bytes format.
//! Usage: cargo run --example export_weights -- <model-dir> <output-file>

use auto_g_embed::rust_embedder::RustContrastiveEmbedder;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <model-dir> <output-file>", args[0]);
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let output_file = &args[2];

    let model = RustContrastiveEmbedder::load_dir(model_dir).expect("failed to load model");
    let bytes = model.to_bytes().expect("failed to serialize");

    println!("Model config: {:?}", model.config());
    println!(
        "Weight bytes: {} ({:.2} MB)",
        bytes.len(),
        bytes.len() as f64 / 1_048_576.0
    );

    fs::write(output_file, &bytes).expect("failed to write output");
    println!("Wrote {output_file}");
}
