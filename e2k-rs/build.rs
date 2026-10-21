//! With the `embed` feature, checks that `models/` holds every model file so a
//! missing one fails with a hint instead of a bare `include_bytes!` error.

use std::path::Path;

const MODELS: [&str; 4] = [
    "model-c2k.npz",
    "model-p2k.npz",
    "accent.npz",
    "ngram.json.zip",
];

fn main() {
    println!("cargo:rerun-if-changed=models");
    if std::env::var_os("CARGO_FEATURE_EMBED").is_none() {
        return;
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let missing: Vec<_> = MODELS.iter().filter(|f| !dir.join(f).exists()).collect();
    if !missing.is_empty() {
        panic!(
            "the `embed` feature needs these files in {}: {missing:?}\n\
             Use the e2k-rs release archive from GitHub Releases, which includes them, \
             or download them from the Releases page into that folder.",
            dir.display()
        );
    }
}
