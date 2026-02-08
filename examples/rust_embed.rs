use std::env;

use auto_g_embed::rust_embedder::RustContrastiveEmbedder;

fn main() {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example rust_embed -- <model_dir> <text>");
        std::process::exit(1);
    }

    let text = args.split_off(1).join(" ");
    let model_dir = &args[0];

    let model = RustContrastiveEmbedder::load_dir(model_dir)
        .unwrap_or_else(|err| panic!("failed to load model from {model_dir}: {err}"));
    let embedding = model.embed(&text);

    let preview = embedding
        .iter()
        .take(8)
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(", ");

    println!("embedding_dims={}", embedding.len());
    println!("embedding_preview=[{}]", preview);
}
