//! English to Katakana conversion with lightweight RNN models.
//!
//! This is the Rust port of the `e2k` Python package. It loads the same
//! model files (`model-c2k.safetensors`, `model-p2k.safetensors`,
//! `accent.safetensors` and `ngram.json.zip`) and runs inference with
//! [`ndarray`].
//!
//! ```no_run
//! use e2k::{AccentPredictor, NGram, Strategy, C2K};
//!
//! let c2k = C2K::from_path("model-c2k.safetensors")?;
//! let ap = AccentPredictor::from_path("accent.safetensors")?;
//! let ngram = NGram::from_path("ngram.json.zip")?;
//!
//! let word = "geogaddi";
//! let katakana = if ngram.is_valid(word) {
//!     c2k.infer(word)
//! } else {
//!     ngram.as_is(word)
//! };
//! println!("{katakana}, accent: {}", ap.predict(&katakana));
//! println!("{}", c2k.infer_with(word, Strategy::TopK { k: 2 }));
//! # Ok::<(), e2k::Error>(())
//! ```

mod accent;
mod embedded;
mod error;
mod ngram;
mod nn;
mod s2s;
mod safetensors;

pub use accent::AccentPredictor;
pub use error::Error;
pub use ngram::NGram;
pub use s2s::{Strategy, C2K, P2K};

pub type Result<T> = std::result::Result<T, Error>;
