use std::collections::HashMap;
use std::path::Path;

use ndarray::{concatenate, Array1, Axis};
use rand::seq::IndexedRandom;
use rand::Rng;

use crate::error::format_err;
use crate::nn::{softmax_inplace, Embedding, Gru, Linear, Mha};
use crate::safetensors::SafeTensors;
use crate::Result;

/// How the next token is picked at each decoding step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Strategy {
    /// Always take the most likely token. Deterministic.
    #[default]
    Greedy,
    /// Pick uniformly among the `k` most likely tokens.
    TopK { k: usize },
    /// Pick uniformly among the smallest set of most likely tokens whose
    /// probability (at temperature `t`) adds up to `p`.
    TopP { p: f32, t: f32 },
}

impl Strategy {
    /// Python's defaults: `k=2`.
    pub fn top_k() -> Self {
        Strategy::TopK { k: 2 }
    }

    /// Python's defaults: `p=0.9, t=1.0`.
    pub fn top_p() -> Self {
        Strategy::TopP { p: 0.9, t: 1.0 }
    }

    fn pick<R: Rng + ?Sized>(&self, logits: Array1<f32>, rng: &mut R) -> usize {
        let sorted_desc = |x: &Array1<f32>| {
            let mut idx: Vec<usize> = (0..x.len()).collect();
            idx.sort_by(|&a, &b| x[b].total_cmp(&x[a]));
            idx
        };
        match *self {
            Strategy::Greedy => argmax(&logits),
            Strategy::TopK { k } => {
                let idx = sorted_desc(&logits);
                *idx[..k.clamp(1, idx.len())]
                    .choose(rng)
                    .expect("candidates are not empty")
            }
            Strategy::TopP { p, t } => {
                let mut probs = logits / t;
                softmax_inplace(&mut probs);
                let idx = sorted_desc(&probs);
                let mut cumsum = 0.0;
                let mut n = 0;
                while cumsum < p && n < idx.len() {
                    cumsum += probs[idx[n]];
                    n += 1;
                }
                *idx[..n.max(1)]
                    .choose(rng)
                    .expect("candidates are not empty")
            }
        }
    }
}

fn argmax(x: &Array1<f32>) -> usize {
    x.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// GRU encoder-decoder with attention, shared by [`C2K`] and [`P2K`].
struct S2S {
    sos_idx: usize,
    eos_idx: usize,
    in_table: Vec<String>,
    in_index: HashMap<String, usize>,
    out_table: Vec<String>,
    e_emb: Embedding,
    k_emb: Embedding,
    encoder: Gru,
    encoder_reverse: Gru,
    encoder_fc: Linear,
    pre_decoder: Gru,
    post_decoder: Gru,
    attn: Mha,
    fc: Linear,
    max_len: usize,
}

impl S2S {
    fn new(st: SafeTensors) -> Result<Self> {
        let parse_idx = |key: &str| -> Result<usize> {
            st.meta(key)?
                .parse()
                .map_err(|_| format_err(format!("metadata `{key}` is not an integer")))
        };
        let split = |s: &str| s.split('\0').map(String::from).collect::<Vec<_>>();
        let in_table = split(st.meta("in_table")?);
        let in_index = in_table
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i))
            .collect();
        Ok(Self {
            sos_idx: parse_idx("sos_idx")?,
            eos_idx: parse_idx("eos_idx")?,
            in_table,
            in_index,
            out_table: split(st.meta("out_table")?),
            e_emb: Embedding::load(&st, "e_emb.weight")?,
            k_emb: Embedding::load(&st, "k_emb.weight")?,
            encoder: Gru::load(&st, "encoder", "")?,
            encoder_reverse: Gru::load(&st, "encoder", "_reverse")?,
            encoder_fc: Linear::load(&st, "encoder_fc.0")?,
            pre_decoder: Gru::load(&st, "pre_decoder", "")?,
            post_decoder: Gru::load(&st, "post_decoder", "")?,
            attn: Mha::load(&st, "attn", 4)?,
            fc: Linear::load(&st, "fc")?,
            max_len: 16,
        })
    }

    /// Maps symbols through the input table, silently dropping unknown ones.
    fn encode<'a>(&self, symbols: impl IntoIterator<Item = &'a str>) -> Vec<usize> {
        let mut src = vec![self.sos_idx];
        src.extend(symbols.into_iter().filter_map(|s| self.in_index.get(s)));
        src.push(self.eos_idx);
        src
    }

    fn forward<R: Rng + ?Sized>(&self, src: &[usize], strategy: Strategy, rng: &mut R) -> String {
        let e_emb = self.e_emb.forward(src);
        let (fwd, _) = self.encoder.forward(e_emb.view(), false);
        let (rev, _) = self.encoder_reverse.forward(e_emb.view(), true);
        let enc_out = concatenate(Axis(1), &[fwd.view(), rev.view()]).expect("same length");
        let enc_out = self.encoder_fc.forward_seq(enc_out.view()).mapv(f32::tanh);
        let memory = self.attn.memory(enc_out.view());

        let mut out = String::new();
        let mut last = self.sos_idx;
        let mut h1 = self.pre_decoder.zeros();
        let mut h2 = self.post_decoder.zeros();
        for _ in 0..self.max_len {
            h1 = self.pre_decoder.cell(self.k_emb.row(last), &h1);
            let attn_out = self.attn.forward(h1.view(), &memory);
            let x = concatenate(Axis(0), &[h1.view(), attn_out.view()]).expect("1-d");
            h2 = self.post_decoder.cell(x.view(), &h2);
            last = strategy.pick(self.fc.forward(h2.view()), rng);
            if last == self.eos_idx {
                break;
            }
            out.push_str(&self.out_table[last]);
        }
        out
    }
}

macro_rules! model {
    ($(#[$doc:meta])* $name:ident, $input:ty) => {
        $(#[$doc])*
        pub struct $name(S2S);

        impl $name {
            pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
                Self::from_bytes(&std::fs::read(path)?)
            }

            pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
                S2S::new(SafeTensors::from_bytes(bytes)?).map(Self)
            }

            /// Maximum number of output symbols, 16 by default.
            pub fn with_max_len(mut self, max_len: usize) -> Self {
                self.0.max_len = max_len;
                self
            }

            /// Valid input symbols; anything else is ignored.
            pub fn in_table(&self) -> &[String] {
                &self.0.in_table
            }

            /// Katakana output symbols.
            pub fn out_table(&self) -> &[String] {
                &self.0.out_table
            }

            /// Greedy decoding.
            pub fn infer(&self, src: $input) -> String {
                self.infer_with(src, Strategy::Greedy)
            }

            /// Decodes with `strategy`, sampling from the thread-local RNG.
            pub fn infer_with(&self, src: $input, strategy: Strategy) -> String {
                self.infer_with_rng(src, strategy, &mut rand::rng())
            }

            /// Decodes with `strategy`, sampling from `rng`.
            pub fn infer_with_rng<R: Rng + ?Sized>(
                &self,
                src: $input,
                strategy: Strategy,
                rng: &mut R,
            ) -> String {
                let src = self.encode(src);
                self.0.forward(&src, strategy, rng)
            }
        }
    };
}

model!(
    /// Converts English spelling (lowercase ASCII letters, space and `'`) to Katakana.
    C2K,
    &str
);

model!(
    /// Converts CMUdict phonemes (e.g. `["JH", "IY2", "OW0"]`) to Katakana.
    P2K,
    &[&str]
);

impl C2K {
    fn encode(&self, word: &str) -> Vec<usize> {
        let chars: Vec<String> = word.to_lowercase().chars().map(String::from).collect();
        self.0.encode(chars.iter().map(String::as_str))
    }
}

impl P2K {
    fn encode(&self, phonemes: &[&str]) -> Vec<usize> {
        self.0.encode(phonemes.iter().copied())
    }
}
