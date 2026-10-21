//! Runs against the model files in `../src/e2k/models` (see the root README
//! for where to download them). Tests are skipped when the files are missing.

use std::path::PathBuf;

use e2k::{AccentPredictor, NGram, Strategy, C2K, P2K};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn model(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../src/e2k/models")
        .join(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: {} not found", path.display());
        None
    }
}

#[test]
fn c2k() {
    let Some(path) = model("model-c2k.npz") else {
        return;
    };
    let c2k = C2K::from_path(path).unwrap();
    assert_eq!(c2k.infer("geogaddi"), "ジオガディ");
    assert_eq!(c2k.infer("GeoGaddi"), "ジオガディ");
    assert!(c2k.in_table().iter().any(|s| s == "a"));

    let mut rng = StdRng::seed_from_u64(0);
    for strategy in [Strategy::top_k(), Strategy::top_p()] {
        assert!(!c2k
            .infer_with_rng("geogaddi", strategy, &mut rng)
            .is_empty());
    }
    assert!(c2k.with_max_len(2).infer("geogaddi").chars().count() <= 2);
}

#[test]
fn p2k() {
    let Some(path) = model("model-p2k.npz") else {
        return;
    };
    let p2k = P2K::from_path(path).unwrap();
    assert_eq!(
        p2k.infer(&["K", "AH0", "M", "P", "Y", "UW1", "T", "ER0"]),
        "コンピューター"
    );
}

#[test]
fn accent() {
    let Some(path) = model("accent.npz") else {
        return;
    };
    let ap = AccentPredictor::from_path(path).unwrap();
    assert_eq!(ap.predict("ステートレス"), 2);
    assert_eq!(ap.predict("ヘパリン"), 1);
    assert_eq!(ap.predict(""), 0);
}

#[test]
fn ngram() {
    let Some(path) = model("ngram.json.zip") else {
        return;
    };
    let ngram = NGram::from_path(path).unwrap();
    assert!(ngram.is_valid("doggy"));
    assert!(!ngram.is_valid("ussr"));
    assert_eq!(ngram.as_is("ussr"), "ユーエスエスアール");
}
