//! The handful of PyTorch layers the models are built from, inference only.

use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::format_err;
use crate::safetensors::SafeTensors;
use crate::Result;

pub(crate) struct Linear {
    /// `[out, in]`, as stored by PyTorch.
    weight: Array2<f32>,
    bias: Array1<f32>,
}

impl Linear {
    pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        Self { weight, bias }
    }

    pub fn load(st: &SafeTensors, prefix: &str) -> Result<Self> {
        Ok(Self::new(
            st.array2(&format!("{prefix}.weight"))?,
            st.array1(&format!("{prefix}.bias"))?,
        ))
    }

    /// `x: [in]` -> `[out]`
    pub fn forward(&self, x: ArrayView1<f32>) -> Array1<f32> {
        self.weight.dot(&x) + &self.bias
    }

    /// `x: [T, in]` -> `[T, out]`
    pub fn forward_seq(&self, x: ArrayView2<f32>) -> Array2<f32> {
        x.dot(&self.weight.t()) + &self.bias
    }
}

pub(crate) struct Embedding {
    weight: Array2<f32>,
}

impl Embedding {
    pub fn load(st: &SafeTensors, name: &str) -> Result<Self> {
        Ok(Self {
            weight: st.array2(name)?,
        })
    }

    pub fn row(&self, idx: usize) -> ArrayView1<'_, f32> {
        self.weight.row(idx)
    }

    /// `idx: [T]` -> `[T, D]`
    pub fn forward(&self, idx: &[usize]) -> Array2<f32> {
        self.weight.select(Axis(0), idx)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// A single-layer, single-direction GRU (gate order `r, z, n`, as in PyTorch).
pub(crate) struct Gru {
    ih: Linear,
    hh: Linear,
    hidden: usize,
}

impl Gru {
    /// Loads e.g. `encoder.weight_ih_l0{suffix}` where `suffix` is `""` or `"_reverse"`.
    pub fn load(st: &SafeTensors, prefix: &str, suffix: &str) -> Result<Self> {
        let get2 = |n: &str| st.array2(&format!("{prefix}.{n}_l0{suffix}"));
        let get1 = |n: &str| st.array1(&format!("{prefix}.{n}_l0{suffix}"));
        let weight_hh = get2("weight_hh")?;
        let hidden = weight_hh.ncols();
        if weight_hh.nrows() != 3 * hidden {
            return Err(format_err(format!("{prefix}: bad GRU weight shape")));
        }
        Ok(Self {
            ih: Linear::new(get2("weight_ih")?, get1("bias_ih")?),
            hh: Linear::new(weight_hh, get1("bias_hh")?),
            hidden,
        })
    }

    pub fn zeros(&self) -> Array1<f32> {
        Array1::zeros(self.hidden)
    }

    /// One step given the precomputed input projection `gi = W_ih x + b_ih`.
    fn step(&self, gi: ArrayView1<f32>, h: &Array1<f32>) -> Array1<f32> {
        let gh = self.hh.forward(h.view());
        let d = self.hidden;
        Array1::from_shape_fn(d, |i| {
            let r = sigmoid(gi[i] + gh[i]);
            let z = sigmoid(gi[d + i] + gh[d + i]);
            let n = (gi[2 * d + i] + r * gh[2 * d + i]).tanh();
            (1.0 - z) * n + z * h[i]
        })
    }

    /// Single step on one input vector.
    pub fn cell(&self, x: ArrayView1<f32>, h: &Array1<f32>) -> Array1<f32> {
        self.step(self.ih.forward(x).view(), h)
    }

    /// Runs over `x: [T, D]`, returning all hidden states `[T, H]` (in input
    /// order) and the final one. With `reverse`, the sequence is consumed from
    /// the end, like the backward half of a bidirectional PyTorch GRU.
    pub fn forward(&self, x: ArrayView2<f32>, reverse: bool) -> (Array2<f32>, Array1<f32>) {
        let gi = self.ih.forward_seq(x);
        let t = x.nrows();
        let mut out = Array2::zeros((t, self.hidden));
        let mut h = self.zeros();
        let order: Box<dyn Iterator<Item = usize>> = if reverse {
            Box::new((0..t).rev())
        } else {
            Box::new(0..t)
        };
        for i in order {
            h = self.step(gi.row(i), &h);
            out.row_mut(i).assign(&h);
        }
        (out, h)
    }
}

/// `nn.MultiheadAttention` for a single query against a fixed memory.
pub(crate) struct Mha {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    scale: f32,
}

/// Keys and values projected once per input sequence.
pub(crate) struct Memory {
    k: Array2<f32>,
    v: Array2<f32>,
}

impl Mha {
    pub fn load(st: &SafeTensors, prefix: &str, num_heads: usize) -> Result<Self> {
        let w = st.array2(&format!("{prefix}.in_proj_weight"))?;
        let b = st.array1(&format!("{prefix}.in_proj_bias"))?;
        let dim = w.ncols();
        if w.nrows() != 3 * dim || b.len() != 3 * dim || dim % num_heads != 0 {
            return Err(format_err(format!("{prefix}: bad attention weight shape")));
        }
        let proj = |i: usize| {
            Linear::new(
                w.slice(s![i * dim..(i + 1) * dim, ..]).to_owned(),
                b.slice(s![i * dim..(i + 1) * dim]).to_owned(),
            )
        };
        Ok(Self {
            q_proj: proj(0),
            k_proj: proj(1),
            v_proj: proj(2),
            o_proj: Linear::load(st, &format!("{prefix}.out_proj"))?,
            num_heads,
            scale: ((dim / num_heads) as f32).sqrt(),
        })
    }

    pub fn memory(&self, kv: ArrayView2<f32>) -> Memory {
        Memory {
            k: self.k_proj.forward_seq(kv),
            v: self.v_proj.forward_seq(kv),
        }
    }

    /// `q: [D]` -> `[D]`
    pub fn forward(&self, q: ArrayView1<f32>, mem: &Memory) -> Array1<f32> {
        let q = self.q_proj.forward(q);
        let d_head = q.len() / self.num_heads;
        let heads: Vec<Array1<f32>> = (0..self.num_heads)
            .map(|h| {
                let cols = s![.., h * d_head..(h + 1) * d_head];
                let k = mem.k.slice(cols);
                let v = mem.v.slice(cols);
                let mut attn = k.dot(&q.slice(s![h * d_head..(h + 1) * d_head])) / self.scale;
                softmax_inplace(&mut attn);
                attn.dot(&v)
            })
            .collect();
        let views: Vec<_> = heads.iter().map(|h| h.view()).collect();
        let o = concatenate(Axis(0), &views).expect("heads have equal length");
        self.o_proj.forward(o.view())
    }
}

pub(crate) fn softmax_inplace(x: &mut Array1<f32>) {
    let max = x.fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    x.mapv_inplace(|v| (v - max).exp());
    let sum = x.sum();
    *x /= sum;
}
