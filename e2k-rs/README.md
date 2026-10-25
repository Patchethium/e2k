# e2k (Rust)

Rust port of the [`e2k`](../README.md) English to Katakana library. It loads the same model files as the Python package and runs inference with [`ndarray`](https://crates.io/crates/ndarray), no Python or BLAS required.

## Installation

The crate isn't on crates.io, as the models exceed its size limit. Depend on it from git, pinned to a release tag:

```toml
[dependencies]
e2k = { git = "https://github.com/Patchethium/e2k", tag = "v0.8.1" }
```

The models are tracked in git under [`src/e2k/models`](../src/e2k/models), shared with the Python package. The `embedded()` constructors build them into your binary (about 11 MB). To load other model files at runtime instead, use `from_path`, or `from_bytes` (`from_zip_bytes` for `NGram`); the linker then leaves the built-in models out.

Cargo clones the whole repository history for git dependencies. To fetch only the tagged commit, enable shallow clones (nightly) in `.cargo/config.toml`:

```toml
[unstable.git]
shallow_index = true
shallow_deps = true
```

## Usage

```rust
use e2k::{AccentPredictor, NGram, Strategy, C2K, P2K};

// Models built into the binary
let c2k = C2K::embedded()?;
let p2k = P2K::embedded()?;
let ap = AccentPredictor::embedded()?;
let ngram = NGram::embedded()?;

// Or from files
let c2k = C2K::from_path("model-c2k.safetensors")?;

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

## Testing

```bash
cargo test
cargo run --release --example demo -- ../src/e2k/models geogaddi ussr
```

Tests read the models from `../src/e2k/models`.

