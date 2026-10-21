# e2k (Rust)

Rust port of the [`e2k`](../README.md) English to Katakana library. It loads the same model files as the Python package and runs inference with [`ndarray`](https://crates.io/crates/ndarray), no Python or BLAS required.

## Installation

The crate isn't on crates.io, as the models exceed its size limit. Each [GitHub release](https://github.com/Patchethium/e2k/releases) has an `e2k-rs-<version>.zip` with the crate source and the models. Unpack it into your project, e.g. as `vendor/e2k-rs`, and add:

```toml
[dependencies]
e2k = { path = "vendor/e2k-rs", features = ["embed"] }
```

The `embed` feature builds the models into your binary (about 11 MB) and enables the `embedded()` constructors. Without it, load the model files at runtime with `from_path`, or `from_bytes` (`from_zip_bytes` for `NGram`).

The model files are the same as the Python package's, also available on the Releases page:

- [`model-c2k.npz`](https://github.com/Patchethium/e2k/releases/download/0.4.0/model-c2k.npz), [`model-p2k.npz`](https://github.com/Patchethium/e2k/releases/download/0.4.0/model-p2k.npz)
- [`accent.npz`](https://github.com/Patchethium/e2k/releases/download/0.5.0/accent.npz)
- [`ngram.json.zip`](https://github.com/Patchethium/e2k/releases/download/0.6.1/ngram.json.zip)

## Usage

```rust
use e2k::{AccentPredictor, NGram, Strategy, C2K, P2K};

// With the `embed` feature
let c2k = C2K::embedded()?;
let p2k = P2K::embedded()?;
let ap = AccentPredictor::embedded()?;
let ngram = NGram::embedded()?;

// Or from files
let c2k = C2K::from_path("model-c2k.npz")?;

// 1. Translation
println!("{}", c2k.infer("geogaddi")); // ジオガディ
println!("{}", p2k.infer(&["K", "AH0", "M", "P", "Y", "UW1", "T", "ER0"])); // コンピューター

// Sampling, see `Strategy`
c2k.infer_with("geogaddi", Strategy::TopK { k: 5 });
c2k.infer_with("geogaddi", Strategy::TopP { p: 0.9, t: 2.0 });

// 2. Pitch accent
let katakana = c2k.infer("geogaddi");
println!("{katakana}: {}", ap.predict(&katakana));

// 3. Transliterate or spell out
let word = "ussr";
let katakana = if ngram.is_valid(word) { c2k.infer(word) } else { ngram.as_is(word) };
println!("{katakana}"); // ユーエスエスアール

// 4. Symbol tables
c2k.in_table();
p2k.out_table();
ap.in_table();
```

| Python                       | Rust                                  |
| ---------------------------- | ------------------------------------- |
| `c2k(word)`                  | `c2k.infer(word)`                     |
| `c2k(word, "top_k", k=5)`    | `c2k.infer_with(word, Strategy::TopK { k: 5 })` |
| `p2k(phonemes)`              | `p2k.infer(&phonemes)`                |
| `ap(katakana)`               | `ap.predict(katakana)`                |
| `ngram(word)`                | `ngram.is_valid(word)`                |
| `ngram.as_is(word)`          | `ngram.as_is(word)`                   |

N-gram warnings about unseen grams go through the [`log`](https://crates.io/crates/log) crate.

## Testing

```bash
cargo test
cargo run --release --example demo -- ../src/e2k/models geogaddi ussr
```

Tests read the models from `../src/e2k/models` and are skipped when they are missing.

## Releasing

```bash
scripts/release.sh v0.7.0
```

This writes `target/release-archive/e2k-rs-<version>.zip` from the crate as committed at the given git ref, plus the models from `../src/e2k/models` (override with `E2K_MODEL_DIR`). Attach the zip to the GitHub release.
