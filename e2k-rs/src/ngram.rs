use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read};
use std::path::Path;

use serde_json::Value;

use crate::error::format_err;
use crate::Result;

/// One order of an interpolated Kneser-Ney character model.
struct Level {
    discount: f64,
    /// `{gram: count}`
    counts: HashMap<String, f64>,
    /// `{history: (total count, distinct next characters)}`
    hists: HashMap<String, (f64, f64)>,
}

impl Level {
    fn from_json(data: &Value) -> Result<Self> {
        let discount = data["discount"]
            .as_f64()
            .ok_or_else(|| format_err("`discount` is not a number"))?;
        let counts: HashMap<String, f64> = data["counts"]
            .as_object()
            .ok_or_else(|| format_err("`counts` is not an object"))?
            .iter()
            .map(|(gram, c)| {
                let c = c
                    .as_f64()
                    .ok_or_else(|| format_err(format!("count of `{gram}` is not a number")))?;
                Ok((gram.clone(), c))
            })
            .collect::<Result<_>>()?;
        let mut hists: HashMap<String, (f64, f64)> = HashMap::new();
        for (gram, &c) in &counts {
            let mut hist = gram.clone();
            hist.pop();
            let entry = hists.entry(hist).or_insert((0.0, 0.0));
            entry.0 += c;
            entry.1 += 1.0;
        }
        Ok(Self {
            discount,
            counts,
            hists,
        })
    }
}

/// Decides whether a word is pronounceable (transliterate with [`crate::C2K`])
/// or should be spelled out letter by letter ([`NGram::as_is`]).
///
/// A linear SVM over the Kneser-Ney scores of orders 1 to 5, trained by
/// `e2k_train/hybrid.py`.
pub struct NGram {
    orders: Vec<usize>,
    /// The levels below each model's top order, shared by all models.
    lower: Vec<Level>,
    /// The top level of each model, in the order of `orders`.
    top: Vec<Level>,
    vocab: f64,
    weights: Vec<f64>,
    bias: f64,
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
        let number = |key: &str| {
            data[key]
                .as_f64()
                .ok_or_else(|| format_err(format!("`{key}` is not a number")))
        };
        let levels = |key: &str| -> Result<Vec<Level>> {
            data[key]
                .as_array()
                .ok_or_else(|| format_err(format!("`{key}` is not a list")))?
                .iter()
                .map(Level::from_json)
                .collect()
        };
        let orders = serde_json::from_value::<Vec<usize>>(data["orders"].clone())?;
        let weights = serde_json::from_value::<Vec<f64>>(data["weights"].clone())?;
        let lower = levels("lower")?;
        let top = levels("top")?;
        if top.len() != orders.len() || weights.len() != orders.len() {
            return Err(format_err("`orders`, `top` and `weights` differ in length"));
        }
        if orders.iter().any(|&n| n == 0 || n > lower.len() + 1) {
            return Err(format_err("an order has no levels to score with"));
        }
        // the unigrams are the characters seen in training, and the end of word
        let valid_chars = top[0]
            .counts
            .keys()
            .filter_map(|g| g.chars().next())
            .filter(|&c| c != '$')
            .collect();

        Ok(Self {
            orders,
            lower,
            top,
            vocab: number("vocab")?,
            weights,
            bias: number("bias")?,
            threshold: number("threshold")?,
            valid_chars,
        })
    }

    /// P(char | hist) under the `i`-th model, interpolated down to a uniform distribution.
    fn prob(&self, i: usize, c: char, hist: &[char]) -> f64 {
        let n = self.orders[i];
        let mut p = 1.0 / self.vocab;
        for k in 1..=hist.len() + 1 {
            let level = if k == n {
                &self.top[i]
            } else {
                &self.lower[k - 1]
            };
            let h: String = hist[hist.len() + 1 - k..].iter().collect();
            let Some(&(total, types)) = level.hists.get(&h) else {
                continue;
            };
            let mut gram = h;
            gram.push(c);
            let count = level.counts.get(&gram).copied().unwrap_or(0.0);
            p = (count - level.discount).max(0.0) / total + level.discount * types / total * p;
        }
        p
    }

    /// Mean log probability per character under the `i`-th model, including the end of word.
    fn model_score(&self, i: usize, word: &str) -> f64 {
        let n = self.orders[i];
        let s: Vec<char> = format!("^{word}$").chars().collect();
        let total: f64 = (1..s.len())
            .map(|j| self.prob(i, s[j], &s[j.saturating_sub(n - 1)..j]).ln())
            .sum();
        total / (s.len() - 1) as f64
    }

    fn word_score(&self, word: &str) -> f64 {
        let weighted: f64 = (0..self.orders.len())
            .map(|i| self.weights[i] * self.model_score(i, word))
            .sum();
        weighted + self.bias
    }

    /// How pronounceable `word` is, valid above [`NGram::threshold`].
    /// A phrase scores as its least pronounceable word, `-inf` without any letters.
    pub fn score(&self, word: &str) -> f64 {
        let scores: Vec<f64> = word
            .to_lowercase()
            .split_whitespace()
            .map(|part| {
                part.chars()
                    .filter(|c| self.valid_chars.contains(c))
                    .collect::<String>()
            })
            .filter(|part| !part.is_empty())
            .map(|part| self.word_score(&part))
            .collect();
        if scores.is_empty() {
            return f64::NEG_INFINITY;
        }
        scores.into_iter().fold(f64::INFINITY, f64::min)
    }

    /// The score above which a word is valid.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Whether `word` looks like it should be transliterated rather than spelled out.
    pub fn is_valid(&self, word: &str) -> bool {
        self.score(word) > self.threshold
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
