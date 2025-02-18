# Description: Inference functions for the E2K model in numpy
import numpy as np
from typing import Optional, Dict
import importlib.resources
from e2k.constants import SOS_IDX, EOS_IDX, en_phones, kanas, ascii_entries
from functools import partial


class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return np.matmul(x, self.weight.T) + self.bias


class Embedding:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x):
        return self.weight[x]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class GRUCell:
    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):
        self.ih = Linear(weight_ih, bias_ih)
        self.hh = Linear(weight_hh, bias_hh)

    def forward(self, x: np.ndarray, h: Optional[np.ndarray] = None):
        """
        x: [D]
        h: [D]
        """
        if h is None:
            h = np.zeros(
                [
                    self.hh.weight.shape[-1],
                ]
            )
        rzn_ih = self.ih.forward(x)
        rzn_hh = self.hh.forward(h)

        rz_ih, n_ih = (
            rzn_ih[: rzn_ih.shape[-1] * 2 // 3],
            rzn_ih[rzn_ih.shape[-1] * 2 // 3 :],
        )
        rz_hh, n_hh = (
            rzn_hh[: rzn_hh.shape[-1] * 2 // 3],
            rzn_hh[rzn_hh.shape[-1] * 2 // 3 :],
        )

        rz = sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, axis=-1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h


class GRU:
    def __init__(self, cell: GRUCell, reverse: bool = False):
        self.cell = cell
        self.reverse = reverse

    def forward(self, x, h: Optional[np.ndarray] = None):
        """
        x: [T, D], unbatched
        """
        if self.reverse:
            x = np.flip(x, axis=0)
        outputs = []
        for i in range(x.shape[0]):
            h = self.cell.forward(x[i], h)
            outputs.append(h)
        outputs = np.stack(outputs)
        if self.reverse:
            outputs = np.flip(outputs, axis=0)
        return outputs, h


class MHA:
    """
    Multi-head attention
    """

    def __init__(
        self,
        in_proj_weight,
        in_proj_bias,
        out_proj_weight,
        out_proj_bias,
        num_heads: int,
    ):
        [q_w, k_w, v_w] = np.split(in_proj_weight, 3, axis=0)
        [q_b, k_b, v_b] = np.split(in_proj_bias, 3, axis=0)
        self.dim = q_w.shape[-1]
        self.q_proj = Linear(q_w, q_b)
        self.k_proj = Linear(k_w, k_b)
        self.v_proj = Linear(v_w, v_b)
        self.o_proj = Linear(out_proj_weight, out_proj_bias)
        self.num_heads = num_heads
        self.d_heads = self.dim // num_heads
        self.scale = np.sqrt(self.dim)

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray):
        """
        q: [T, D]
        k: [T, D]
        v: [T, D]
        """
        q = self.q_proj.forward(q)
        k = self.k_proj.forward(k)
        v = self.v_proj.forward(v)
        q = np.split(q, self.num_heads, axis=-1)
        q = np.stack(q, axis=0)
        k = np.split(k, self.num_heads, axis=-1)
        k = np.stack(k, axis=0)
        v = np.split(v, self.num_heads, axis=-1)
        v = np.stack(v, axis=0)
        attn = np.matmul(q, np.transpose(k, (0, 2, 1)))
        attn = attn / self.scale
        attn = np.exp(attn)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        o = np.matmul(attn, v)
        o = np.transpose(o, (1, 0, 2))
        o = o.reshape([o.shape[0], -1])
        return self.o_proj.forward(o)


class S2S:
    def __init__(self, weights: Dict[str, np.ndarray], max_len: int = 16):
        # fp32 is usually faster than fp16 on CPU
        new_weight = {}
        for k, v in weights.items():
            new_weight[k] = v.astype(np.float32)
        weights = new_weight
        self.e_emb = Embedding(weights["e_emb.weight"])
        self.k_emb = Embedding(weights["k_emb.weight"])
        self.encoder = GRU(
            GRUCell(
                weights["encoder.weight_ih_l0"],
                weights["encoder.weight_hh_l0"],
                weights["encoder.bias_ih_l0"],
                weights["encoder.bias_hh_l0"],
            )
        )
        self.encoder_reverse = GRU(
            GRUCell(
                weights["encoder.weight_ih_l0_reverse"],
                weights["encoder.weight_hh_l0_reverse"],
                weights["encoder.bias_ih_l0_reverse"],
                weights["encoder.bias_hh_l0_reverse"],
            ),
            reverse=True,
        )
        self.encoder_fc = Linear(
            weights["encoder_fc.0.weight"], weights["encoder_fc.0.bias"]
        )
        self.pre_decoder = GRU(
            GRUCell(
                weights["pre_decoder.weight_ih_l0"],
                weights["pre_decoder.weight_hh_l0"],
                weights["pre_decoder.bias_ih_l0"],
                weights["pre_decoder.bias_hh_l0"],
            )
        )
        self.post_decoder = GRU(
            GRUCell(
                weights["post_decoder.weight_ih_l0"],
                weights["post_decoder.weight_hh_l0"],
                weights["post_decoder.bias_ih_l0"],
                weights["post_decoder.bias_hh_l0"],
            )
        )
        self.attn = MHA(
            weights["attn.in_proj_weight"],
            weights["attn.in_proj_bias"],
            weights["attn.out_proj.weight"],
            weights["attn.out_proj.bias"],
            4,
        )
        self.fc = Linear(weights["fc.weight"], weights["fc.bias"])
        self.max_len = max_len
        self.decode = self.greedy

    def greedy(self, step_dec: np.ndarray):
        """
        step_dec: [N]
        """
        return np.argmax(step_dec, axis=-1)

    def top_k(self, step_dec: np.ndarray, k: int):
        """
        step_dec: [N]
        """
        candidates = np.flip(np.argsort(step_dec, axis=-1), axis=-1)[:k]
        return np.random.choice(candidates)

    def top_p(self, step_dec: np.ndarray, p: float, t: float):
        """
        step_dec: [N]
        p: max probability
        t: temperature
        """
        # softmax
        step_dec = np.exp(step_dec) / t
        step_dec = step_dec / step_dec.sum()
        sorted_idx = np.flip(np.argsort(step_dec, axis=-1), axis=-1)
        i = 0
        cumsum = 0
        while cumsum < p:
            cumsum += step_dec[sorted_idx[i]]
            i += 1
        candidates = sorted_idx[:i]
        return np.random.choice(candidates)

    def set_decode_strategy(self, strategy: str, **kwargs):
        if strategy == "greedy":
            self.decode = self.greedy
        elif strategy == "top_k":
            self.decode = partial(self.top_k, k=kwargs.get("k", 3))
        elif strategy == "top_p":
            self.decode = partial(
                self.top_p, p=kwargs.get("p", 0.9), t=kwargs.get("t", 1)
            )
        else:
            raise ValueError("Invalid decode strategy")

    def forward(self, src):
        """
        In numpy, only inference is supported.
        """
        e_emb = self.e_emb.forward(src)
        enc_out, _ = self.encoder.forward(e_emb)
        enc_out_rev, _ = self.encoder_reverse.forward(e_emb)
        enc_out = np.concatenate([enc_out, enc_out_rev], axis=-1)
        enc_out = self.encoder_fc.forward(enc_out)
        enc_out = tanh(enc_out)
        res = [SOS_IDX]
        h1 = None
        h2 = None
        for _ in range(self.max_len):
            dec_emb = self.k_emb.forward(np.array([res[-1]]))
            dec_out, h1 = self.pre_decoder.forward(dec_emb, h1)
            attn_out = self.attn.forward(dec_out, enc_out, enc_out)
            x = np.concatenate([dec_out, attn_out], axis=-1)
            x, h2 = self.post_decoder.forward(x, h2)
            x = self.fc.forward(x)
            x = x[0]
            res.append(self.decode(x))
            if res[-1] == EOS_IDX:
                break
        return res


class BaseE2K:
    def __init__(self):
        self.s2s = None
        self.in_table = None
        self.out_table = None
        raise NotImplementedError

    def set_decode_strategy(self, strategy: str, **kwargs):
        self.s2s.set_decode_strategy(strategy, **kwargs)

    def __call__(self, src: str) -> str:
        src = list(filter(lambda x: x in self.in_table, src))
        src = [self.in_table[c] for c in src]
        src = [SOS_IDX] + src + [EOS_IDX]
        src = np.array(src)
        tgt = self.s2s.forward(src)
        tgt = [int(v) for v in tgt[1:-1]]
        tgt = [self.out_table[c] for c in tgt]
        return "".join(tgt)


def get_weight_path(filename) -> str:
    return importlib.resources.files("e2k.models").joinpath(filename)


class P2K(BaseE2K):
    def __init__(self, max_len: int = 16):
        weights = np.load(get_weight_path("model-p2k.npz"), allow_pickle=True)
        self.s2s = S2S(weights, max_len)
        self.in_table = {c: i for i, c in enumerate(en_phones)}
        self.out_table = {i: c for i, c in enumerate(kanas)}


class C2K(BaseE2K):
    def __init__(self, max_len: int = 16):
        weights = np.load(get_weight_path("model-c2k.npz"), allow_pickle=True)
        self.s2s = S2S(weights, max_len)
        self.in_table = {c: i for i, c in enumerate(ascii_entries)}
        self.out_table = {i: c for i, c in enumerate(kanas)}


if __name__ == "__main__":
    import argparse
    from g2p_en import G2p

    g2p = G2p()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    p2k = P2K()
    c2k = C2K()
    word = "gutenberg"
    phonemes = g2p(word)
    print(word)
    print("P2K: ", p2k(phonemes))
    print("C2K: ", c2k(word))
    p2k.set_decode_strategy("top_k", k=2)
    c2k.set_decode_strategy("top_k", k=2)
    print("P2K (top_k): ", p2k(phonemes))
    print("C2K (top_k): ", c2k(word))
    p2k.set_decode_strategy("top_p", p=0.3, t=2)
    c2k.set_decode_strategy("top_p", p=0.3, t=2)
    print("P2K (top_p): ", p2k(phonemes))
    print("C2K (top_p): ", c2k(word))
