//! Models built into the binary, taken from the Python package's
//! `src/e2k/models` so both share one copy in git. The linker drops them from
//! binaries that never call `embedded()`.

use crate::{AccentPredictor, NGram, Result, C2K, P2K};

macro_rules! model {
    ($file:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../src/e2k/models/",
            $file
        ))
    };
}

impl C2K {
    /// Loads the C2K model built into the binary.
    pub fn embedded() -> Result<Self> {
        Self::from_bytes(model!("model-c2k.npz"))
    }
}

impl P2K {
    /// Loads the P2K model built into the binary.
    pub fn embedded() -> Result<Self> {
        Self::from_bytes(model!("model-p2k.npz"))
    }
}

impl AccentPredictor {
    /// Loads the accent model built into the binary.
    pub fn embedded() -> Result<Self> {
        Self::from_bytes(model!("accent.npz"))
    }
}

impl NGram {
    /// Loads the n-gram model built into the binary.
    pub fn embedded() -> Result<Self> {
        Self::from_zip_bytes(model!("ngram.json.zip"))
    }
}
