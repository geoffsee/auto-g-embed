#[cfg(feature = "semantic")]
fn main() -> anyhow::Result<()> {
    use auto_g_embed::semantic::SemanticEmbedder;
    use std::env;

    let mut args = env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "artifacts/model/onnx".to_owned());
    let text = args
        .next()
        .unwrap_or_else(|| "The quick brown fox jumps over the lazy dog.".to_owned());

    let mut embedder = SemanticEmbedder::from_onnx_dir(model_dir)?;
    let embedding = embedder.embed(&text)?;

    println!("embedding_dims={}", embedding.len());
    println!(
        "embedding_preview={:?}",
        &embedding[..embedding.len().min(8)]
    );
    Ok(())
}

#[cfg(not(feature = "semantic"))]
fn main() {
    eprintln!("Re-run with `--features semantic` to enable ONNX semantic inference.");
}
