//! Checks that the models the crate embeds from
//! `../src/e2k/models` exist, so a crate copied out of the e2k repository fails
//! with a hint instead of a bare `include_bytes!` error.

use std::path::Path;

const MODELS: [&str; 4] = [
    "model-c2k.safetensors",
    "model-p2k.safetensors",
    "accent.safetensors",
    "ngram.json.zip",
];

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../src/e2k/models");
    println!("cargo:rerun-if-changed={}", dir.display());
    let missing: Vec<_> = MODELS.iter().filter(|f| !dir.join(f).exists()).collect();
    if !missing.is_empty() {
        panic!(
            "e2k-rs needs these files in {}: {missing:?}\n\
             e2k-rs embeds the models from the e2k repository, so depend on it with \
             `git = \"https://github.com/Patchethium/e2k\"` or a path inside a full checkout.",
            dir.display()
        );
    }
}
