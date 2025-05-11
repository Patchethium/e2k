# Description: Inference functions for the E2K model in numpy
import zipfile
import numpy as np
from typing import Callable, List, Literal, Optional, Dict, Tuple, Union
import importlib.resources
from functools import partial
import math
import statistics as st
import json


class Linear:
    def __init__(self, weight, bias) -> None:
        self.weight = weight
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weight.T) + self.bias


class Embedding:
    def __init__(self, weight) -> None:
        self.weight = weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.weight[x]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class GRUCell:
    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh):
        self.ih = Linear(weight_ih, bias_ih)
        self.hh = Linear(weight_hh, bias_hh)

    def forward(self, x: np.ndarray, h: Optional[np.ndarray] = None) -> np.ndarray:
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

    def forward(
        self, x, h: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    ) -> None:
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

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
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


def greedy(step_dec: np.ndarray):
    """
    step_dec: [N]
    """
    return np.argmax(step_dec, axis=-1)


def top_k(step_dec: np.ndarray, k: int):
    """
    step_dec: [N]
    """
    candidates = np.flip(np.argsort(step_dec, axis=-1), axis=-1)[:k]
    return np.random.choice(candidates)


def top_p(step_dec: np.ndarray, p: float, t: float):
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


class S2S:
    def __init__(self, weights: Dict[str, np.ndarray], max_len: int = 16):
        # fp32 is usually faster than fp16 on CPU
        new_weight = {}
        for k, v in weights.items():
            if v.dtype == np.float16:
                new_weight[k] = v.astype(np.float32)
            else:
                new_weight[k] = v
        weights = new_weight
        metadata = weights["metadata"].item()
        self.sos_idx = int(metadata["sos_idx"])
        self.eos_idx = int(metadata["eos_idx"])
        self.in_table = list(metadata["in_table"].split("\0"))
        self.out_table = list(metadata["out_table"].split("\0"))

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

    def forward(self, src, decoder: Callable) -> np.ndarray:
        """
        In numpy, only inference is supported.
        """
        e_emb = self.e_emb.forward(src)
        enc_out, _ = self.encoder.forward(e_emb)
        enc_out_rev, _ = self.encoder_reverse.forward(e_emb)
        enc_out = np.concatenate([enc_out, enc_out_rev], axis=-1)
        enc_out = self.encoder_fc.forward(enc_out)
        enc_out = tanh(enc_out)
        res = [self.sos_idx]
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
            res.append(decoder(x))
            if res[-1] == self.eos_idx:
                break
        return res


class BaseE2K:
    def __init__(self, name: str, max_len: int = 16):
        data = np.load(get_asset_path(name), allow_pickle=True)
        self.s2s = S2S(data, max_len)
        self.in_table = {c: i for i, c in enumerate(self.s2s.in_table)}
        self.out_table = self.s2s.out_table

    def set_decode_strategy(self, strategy: str, **kwargs):
        self.s2s.set_decode_strategy(strategy, **kwargs)

    def __call__(
        self,
        src: Union[str, List[str]],
        strategy: Optional[Literal["greedy", "top_k", "top_p"]] = None,
        *,
        k: Optional[int] = None,
        p: Optional[float] = None,
        t: Optional[float] = None,
    ) -> str:
        if isinstance(src, str):
            src = src.lower()
        src = list(filter(lambda x: x in self.in_table, src))
        src = [self.in_table[c] for c in src]
        src = [self.s2s.sos_idx] + src + [self.s2s.eos_idx]
        src = np.array(src)
        match strategy:
            case "greedy" | None:
                tgt = self.s2s.forward(src, greedy)
            case "top_k":
                tgt = self.s2s.forward(src, partial(top_k, k=k if k is not None else 2))
            case "top_p":
                tgt = self.s2s.forward(
                    src,
                    partial(
                        top_p,
                        p=p if p is not None else 0.9,
                        t=t if t is not None else 1.0,
                    ),
                )
            case _:
                raise ValueError(f"Unknown decoding strategy: {strategy=}")
        tgt = [int(v) for v in tgt[1:-1]]
        tgt = [self.out_table[c] for c in tgt]
        return "".join(tgt)


def get_asset_path(filename) -> str:
    return importlib.resources.files("e2k.models").joinpath(filename)


class P2K(BaseE2K):
    def __init__(self, max_len: int = 16):
        super().__init__("model-p2k.npz", max_len)


class C2K(BaseE2K):
    def __init__(self, max_len: int = 16):
        super().__init__("model-c2k.npz", max_len)


class AccentPredictor:
    def __init__(self):
        name = "accent.npz"
        weights = np.load(get_asset_path(name), allow_pickle=True)
        metadata = weights["metadata"].item()
        self.symbols = list(metadata["in_table"].split("\0"))
        self.in_table = {c: i for i, c in enumerate(self.symbols)}
        self.emb = Embedding(weights["emb.weight"])
        self.rnn = GRU(
            GRUCell(
                weights["rnn.weight_ih_l0"],
                weights["rnn.weight_hh_l0"],
                weights["rnn.bias_ih_l0"],
                weights["rnn.bias_hh_l0"],
            )
        )
        self.rnn_rev = GRU(
            GRUCell(
                weights["rnn.weight_ih_l0_reverse"],
                weights["rnn.weight_hh_l0_reverse"],
                weights["rnn.bias_ih_l0_reverse"],
                weights["rnn.bias_hh_l0_reverse"],
            )
        )
        self.head = Linear(weights["head.weight"], weights["head.bias"])

    def forward(self, x: str) -> int:
        """
        x: [T]
        """
        x = list(filter(lambda x: x in self.in_table, x))
        x = [self.in_table[c] for c in x]
        x = np.array(x)
        x = self.emb.forward(x)  # [T,N]
        _, hf = self.rnn.forward(x, None)  # [N]
        _, hr = self.rnn_rev.forward(x, None)  # [N]
        h = np.concatenate([hf, hr], axis=0)
        h = self.head.forward(h)
        h = h[0] * x.shape[0]
        h = int(h + 0.5)  # round operation
        return h

    def __call__(self, x: str) -> int:
        x = x.lower()
        return self.forward(x)


class NGramModel:
    def __init__(self, n, freq={}):
        self.n = n
        self.freq = freq  # {prefix: {appendix: frequency}}

    def score(self, word):
        """Score the log likelihood of a word or phrase being valid."""
        # Handle multi-word phrases
        word = word.lower()
        word = f"^{word}$"
        if len(word) < self.n:
            word = "^"*(self.n - len(word)) + word
        ngrams = [word[i : i + self.n] for i in range(len(word) - self.n + 1)]

        # Calculate score for each n-gram
        scores = []
        for ngram in ngrams:
            prefix, appendix = ngram[:-1], ngram[-1]

            if prefix not in self.freq or appendix not in self.freq[prefix]:
                print(f"Warning: gram `{prefix}-{appendix}` in word `{word}` is not found in {self.n}-gram model")
                return -float("inf")  # Unknown n-gram

            scores.append(math.log(self.freq[prefix][appendix] + 1e-10))
        if len(scores):
            print(f"Warning: no grams extracted from {self.n}-gram model for word `{word}`")
            return -float("inf")
        return st.mean(scores)


class NGramCollection:
    def __init__(self):
        path = get_asset_path("ngram.json.zip")
        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_LZMA) as z:
            with z.open("ngram.json") as f:
                data = json.loads(f.read())

        self.weights = data["weights"]
        self.threshold = data["threshold"]
        self.valid_chars = set(data["valid_chars"] + [" "])

        self.models = {NGramModel(int(k), v) for k, v in data["models"].items()}

        self.spell_table = {
            "a": "エー",
            "b": "ビー",
            "c": "シー",
            "d": "ディー",
            "e": "イー",
            "f": "エフ",
            "g": "ジー",
            "h": "エイチ",
            "i": "アイ",
            "j": "ジェー",
            "k": "ケー",
            "l": "エル",
            "m": "エム",
            "n": "エヌ",
            "o": "オー",
            "p": "ピー",
            "q": "キュー",
            "r": "アール",
            "s": "エス",
            "t": "ティー",
            "u": "ユー",
            "v": "ヴィー",
            "w": "ダブリュー",
            "x": "エックス",
            "y": "ワイ",
            "z": "ゼット",
        }

    def score(self, word):
        """Calculate weighted average score across all n-gram models."""
        scores = [model.score(word) for model in self.models]

        if self.weights:
            return sum(w * s for w, s in zip(self.weights, scores))
        else:
            return st.mean(scores)
    
    def as_is(self, word: str) -> str:
        cleaned = [c for c in word if c in self.spell_table]
        return "".join(self.spell_table[c] for c in cleaned)

    def __call__(self, word: str) -> bool:
        word = word.lower()
        cleaned = "".join(c for c in word if c in self.valid_chars)
        return self.score(cleaned) > self.threshold


if __name__ == "__main__":
    import argparse
    from g2p_en import G2p

    g2p = G2p()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    ngram = NGramCollection()
    p2k = P2K()
    c2k = C2K()
    word = "geogaddi"
    phonemes = g2p(word)
    print(word)
    print(f"Ngram result: {'Valid' if ngram(word) else 'Invalid'}")
    print(f"As is: {ngram.as_is(word)}")
    print("P2K: ", p2k(phonemes))
    print("C2K: ", c2k(word))
    print("P2K (top_k): ", p2k(phonemes, "top_k", k=2))
    print("C2K (top_k): ", c2k(word, "top_k", k=2))
    print("P2K (top_p): ", p2k(phonemes, "top_p", p=0.7, t=2))
    print("C2K (top_p): ", c2k(word, "top_p", p=0.7, t=2))

    katakana = c2k(word)
    ap = AccentPredictor()
    print(f"C2K Katakana: {katakana}, Accent: {ap(katakana)}")
    katakana = p2k(phonemes)
    print(f"P2K Katakana: {katakana}, Accent: {ap(katakana)}")
