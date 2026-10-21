use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read};
use std::path::Path;

use serde_json::Value;

use crate::error::format_err;
use crate::Result;

/// Character n-gram model of one order.
struct NGramModel {
    n: usize,
    /// `{prefix: {appendix: frequency}}`
    freq: HashMap<String, HashMap<char, f64>>,
}

impl NGramModel {
    /// Mean log likelihood of the n-grams in `word`, `-inf` if any is unseen.
    fn score(&self, word: &str) -> f64 {
        let mut chars: Vec<char> = format!("^{}$", word.to_lowercase()).chars().collect();
        if chars.len() < self.n {
            let pad = vec!['^'; self.n - chars.len()];
            chars.splice(0..0, pad);
        }
        let mut total = 0.0;
        let mut count = 0;
        for gram in chars.windows(self.n) {
            let (prefix, appendix) = gram.split_at(self.n - 1);
            let prefix: String = prefix.iter().collect();
            let Some(&freq) = self.freq.get(&prefix).and_then(|m| m.get(&appendix[0])) else {
                log::warn!(
                    "gram `{prefix}-{}` in word `{word}` is not found in {}-gram model",
                    appendix[0],
                    self.n
                );
                return f64::NEG_INFINITY;
            };
            total += (freq + 1e-10).ln();
            count += 1;
        }
        if count == 0 {
            log::warn!(
                "no grams extracted from {}-gram model for word `{word}`",
                self.n
            );
            return f64::NEG_INFINITY;
        }
        total / count as f64
    }
}

/// Decides whether a word is pronounceable (transliterate with [`crate::C2K`])
/// or should be spelled out letter by letter ([`NGram::as_is`]).
pub struct NGram {
    /// In the order of `weights`.
    models: Vec<NGramModel>,
    weights: Option<Vec<f64>>,
    threshold: f64,
    valid_chars: HashSet<char>,
}

impl NGram {
    /// Loads the LZMA-compressed `ngram.json.zip`.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_zip_bytes(&std::fs::read(path)?)
    }

    pub fn from_zip_bytes(bytes: &[u8]) -> Result<Self> {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
        let mut json = String::new();
        archive.by_name("ngram.json")?.read_to_string(&mut json)?;
        Self::from_json(&json)
    }

    /// Loads the uncompressed `ngram.json`.
    pub fn from_json(json: &str) -> Result<Self> {
        let data: Value = serde_json::from_str(json)?;
        let threshold = data["threshold"]
            .as_f64()
            .ok_or_else(|| format_err("`threshold` is not a number"))?;
        let weights = match &data["weights"] {
            Value::Null => None,
            w => Some(serde_json::from_value::<Vec<f64>>(w.clone())?),
        };
        let mut valid_chars: HashSet<char> = data["valid_chars"]
            .as_array()
            .ok_or_else(|| format_err("`valid_chars` is not a list"))?
            .iter()
            .filter_map(|c| c.as_str()?.chars().next())
            .collect();
        valid_chars.insert(' ');

        let models = data["models"]
            .as_object()
            .ok_or_else(|| format_err("`models` is not an object"))?
            .iter()
            .map(|(n, freq)| {
                let n = n
                    .parse()
                    .map_err(|_| format_err(format!("bad n-gram order `{n}`")))?;
                let freq =
                    serde_json::from_value::<HashMap<String, HashMap<String, f64>>>(freq.clone())?
                        .into_iter()
                        .map(|(prefix, apps)| {
                            let apps = apps
                                .into_iter()
                                .filter_map(|(a, f)| Some((a.chars().next()?, f)))
                                .collect();
                            (prefix, apps)
                        })
                        .collect();
                Ok(NGramModel { n, freq })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            models,
            weights,
            threshold,
            valid_chars,
        })
    }

    /// Weighted (or plain mean, without weights) log likelihood across orders.
    pub fn score(&self, word: &str) -> f64 {
        let scores = self.models.iter().map(|m| m.score(word));
        match &self.weights {
            Some(w) => w.iter().zip(scores).map(|(w, s)| w * s).sum(),
            None => scores.sum::<f64>() / self.models.len() as f64,
        }
    }

    /// Whether `word` looks like it should be transliterated rather than spelled out.
    pub fn is_valid(&self, word: &str) -> bool {
        let cleaned: String = word
            .to_lowercase()
            .chars()
            .filter(|c| self.valid_chars.contains(c))
            .collect();
        self.score(&cleaned) > self.threshold
    }

    /// Spells `word` out letter by letter, e.g. `ussr` -> `ユーエスエスアール`.
    /// Characters other than lowercase ASCII letters are dropped.
    pub fn as_is(&self, word: &str) -> String {
        word.chars().filter_map(spell).collect()
    }
}

fn spell(c: char) -> Option<&'static str> {
    Some(match c {
        'a' => "エー",
        'b' => "ビー",
        'c' => "シー",
        'd' => "ディー",
        'e' => "イー",
        'f' => "エフ",
        'g' => "ジー",
        'h' => "エイチ",
        'i' => "アイ",
        'j' => "ジェー",
        'k' => "ケー",
        'l' => "エル",
        'm' => "エム",
        'n' => "エヌ",
        'o' => "オー",
        'p' => "ピー",
        'q' => "キュー",
        'r' => "アール",
        's' => "エス",
        't' => "ティー",
        'u' => "ユー",
        'v' => "ヴィー",
        'w' => "ダブリュー",
        'x' => "エックス",
        'y' => "ワイ",
        'z' => "ゼット",
        _ => return None,
    })
}
