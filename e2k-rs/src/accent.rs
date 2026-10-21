use std::collections::HashMap;
use std::path::Path;

use ndarray::{concatenate, Axis};

use crate::nn::{Embedding, Gru, Linear};
use crate::npz::Npz;
use crate::Result;

/// Predicts the pitch accent core of a Katakana word, trained on Unidic.
pub struct AccentPredictor {
    symbols: Vec<String>,
    in_index: HashMap<char, usize>,
    emb: Embedding,
    rnn: Gru,
    rnn_rev: Gru,
    head: Linear,
}

impl AccentPredictor {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let npz = Npz::from_bytes(bytes)?;
        let symbols: Vec<String> = npz
            .meta("in_table")?
            .split('\0')
            .map(String::from)
            .collect();
        let in_index = symbols
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.chars().next().map(|c| (c, i)))
            .collect();
        Ok(Self {
            symbols,
            in_index,
            emb: Embedding::load(&npz, "emb.weight")?,
            rnn: Gru::load(&npz, "rnn", "")?,
            rnn_rev: Gru::load(&npz, "rnn", "_reverse")?,
            head: Linear::load(&npz, "head")?,
        })
    }

    /// Valid Katakana symbols; anything else is ignored.
    pub fn in_table(&self) -> &[String] {
        &self.symbols
    }

    /// Returns the accent core position (0 = heiban) of `katakana`.
    pub fn predict(&self, katakana: &str) -> i32 {
        let idx: Vec<usize> = katakana
            .chars()
            .filter_map(|c| self.in_index.get(&c).copied())
            .collect();
        if idx.is_empty() {
            return 0;
        }
        let x = self.emb.forward(&idx);
        let (_, hf) = self.rnn.forward(x.view(), false);
        let (_, hr) = self.rnn_rev.forward(x.view(), true);
        let h = concatenate(Axis(0), &[hf.view(), hr.view()]).expect("1-d");
        // The head predicts the accent position relative to the word length.
        let rel = self.head.forward(h.view())[0];
        (rel * idx.len() as f32 + 0.5) as i32
    }
}
