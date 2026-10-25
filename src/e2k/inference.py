# Description: Inference functions for the E2K model in numpy
import zipfile
import struct
import numpy as np
from typing import Callable, List, Literal, Optional, Dict, Tuple, Union
import importlib.resources
from functools import partial
import math
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
            # match the weights' dtype, a float64 state would upcast every matmul
            h = np.zeros(
                [
                    self.hh.weight.shape[-1],
                ],
                dtype=self.hh.weight.dtype,
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
        # torch scales by sqrt(head_dim), a python float keeps float32 inputs float32
        self.scale = math.sqrt(self.d_heads)

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        x: [T, D] -> [H, T, D/H]
        """
        return np.stack(np.split(x, self.num_heads, axis=-1), axis=0)

    def memory(self, kv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Projects the keys and values once, so decoding steps can reuse them.
        kv: [T, D] -> ([H, D/H, T], [H, T, D/H])
        """
        k = self.split_heads(self.k_proj.forward(kv))
        v = self.split_heads(self.v_proj.forward(kv))
        return np.transpose(k, (0, 2, 1)), v

    def forward(
        self, q: np.ndarray, memory: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        q: [T, D]
        memory: from `memory()`
        """
        k, v = memory
        q = self.split_heads(self.q_proj.forward(q))
        attn = np.matmul(q, k)
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


def top_k(step_dec: np.ndarray, k: int, rng=np.random):
    """
    step_dec: [N]
    rng: `np.random` or a `np.random.Generator`
    """
    candidates = np.flip(np.argsort(step_dec, axis=-1), axis=-1)[:k]
    return rng.choice(candidates)


def top_p(step_dec: np.ndarray, p: float, t: float, rng=np.random):
    """
    step_dec: [N]
    p: max probability
    t: temperature
    rng: `np.random` or a `np.random.Generator`
    """
    # softmax
    step_dec = np.exp((step_dec - step_dec.max()) / t)
    step_dec = step_dec / step_dec.sum()
    sorted_idx = np.flip(np.argsort(step_dec, axis=-1), axis=-1)
    i = 0
    cumsum = 0
    while cumsum < p and i < len(sorted_idx):
        cumsum += step_dec[sorted_idx[i]]
        i += 1
    candidates = sorted_idx[:i]
    return rng.choice(candidates)


# safetensors dtypes the models may use, all little-endian
SAFETENSORS_DTYPES = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}


def load_safetensors(data: bytes) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Reads a safetensors file with numpy only: a u64 header length, a JSON header
    of `{name: {dtype, shape, data_offsets}}` plus `__metadata__`, then the raw tensors.
    Returns the tensors and the metadata.
    """
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_len])
    metadata = header.pop("__metadata__", {})
    body = memoryview(data)[8 + header_len :]
    weights = {}
    for name, info in header.items():
        if info["dtype"] not in SAFETENSORS_DTYPES:
            raise ValueError(f"tensor `{name}` has unsupported dtype {info['dtype']}")
        dtype = np.dtype(SAFETENSORS_DTYPES[info["dtype"]])
        start, end = info["data_offsets"]
        if end > len(body) or end - start != math.prod(info["shape"]) * dtype.itemsize:
            raise ValueError(f"tensor `{name}` has bad data offsets")
        weights[name] = np.frombuffer(body[start:end], dtype).reshape(info["shape"])
    return weights, metadata


def to_fp32(weights) -> Dict[str, np.ndarray]:
    """
    fp32 is usually faster than fp16 on CPU
    """
    new_weight = {}
    for k, v in weights.items():
        if v.dtype == np.float16:
            new_weight[k] = v.astype(np.float32)
        else:
            new_weight[k] = v
    return new_weight


class S2S:
    def __init__(
        self,
        weights: Dict[str, np.ndarray],
        metadata: Dict[str, str],
        max_len: int = 16,
    ):
        weights = to_fp32(weights)
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
        memory = self.attn.memory(enc_out)
        res = [self.sos_idx]
        h1 = None
        h2 = None
        for _ in range(self.max_len):
            dec_emb = self.k_emb.forward(np.array([res[-1]]))
            dec_out, h1 = self.pre_decoder.forward(dec_emb, h1)
            attn_out = self.attn.forward(dec_out, memory)
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
        self.s2s = S2S(*load_safetensors(read_asset(name)), max_len)
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
        seed: Optional[int] = None,
    ) -> str:
        """
        `seed` makes `top_k` and `top_p` reproducible, without it they draw from `np.random`.
        """
        rng = np.random if seed is None else np.random.default_rng(seed)
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
                tgt = self.s2s.forward(
                    src, partial(top_k, k=k if k is not None else 2, rng=rng)
                )
            case "top_p":
                tgt = self.s2s.forward(
                    src,
                    partial(
                        top_p,
                        p=p if p is not None else 0.9,
                        t=t if t is not None else 1.0,
                        rng=rng,
                    ),
                )
            case _:
                raise ValueError(f"Unknown decoding strategy: {strategy=}")
        tgt = tgt[1:]
        # strip <eos>, absent when decoding stopped at max_len
        if tgt and tgt[-1] == self.s2s.eos_idx:
            tgt = tgt[:-1]
        tgt = [int(v) for v in tgt]
        tgt = [self.out_table[c] for c in tgt]
        return "".join(tgt)


def get_asset_path(filename) -> str:
    return importlib.resources.files("e2k.models").joinpath(filename)


def read_asset(filename) -> bytes:
    return get_asset_path(filename).read_bytes()


# "base": the full width model, "turbo": a width 128 model distilled from it,
# about twice as fast with comparable accuracy
ModelVariant = Literal["base", "turbo"]


def model_file(task: str, model: ModelVariant) -> str:
    match model:
        case "base":
            return f"model-{task}.safetensors"
        case "turbo":
            return f"model-{task}-turbo.safetensors"
        case _:
            raise ValueError(f"Unknown model: {model=}")


class P2K(BaseE2K):
    def __init__(self, max_len: int = 16, model: ModelVariant = "base"):
        super().__init__(model_file("p2k", model), max_len)


class C2K(BaseE2K):
    def __init__(self, max_len: int = 16, model: ModelVariant = "base"):
        super().__init__(model_file("c2k", model), max_len)


class AccentPredictor:
    def __init__(self):
        weights, metadata = load_safetensors(read_asset("accent.safetensors"))
        weights = to_fp32(weights)
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
            ),
            reverse=True,
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


class KneserNey:
    """Interpolated Kneser-Ney character model, see `e2k_train/ngram_kn.py`."""

    def __init__(
        self,
        levels: List[Tuple[float, Dict[str, int], Dict[str, Tuple[int, int]]]],
        vocab: int,
    ):
        # levels[k - 1]: (discount, {gram: count}, {history: (total, types)}) of order k
        self.levels = levels
        self.vocab = vocab

    def prob(self, char: str, hist: str) -> float:
        """P(char | hist), interpolated down to a uniform distribution."""
        p = 1.0 / self.vocab
        for k in range(1, len(hist) + 2):
            discount, counts, hists = self.levels[k - 1]
            h = hist[len(hist) - k + 1 :]
            if h not in hists:
                continue
            total, types = hists[h]
            c = counts.get(h + char, 0)
            p = max(c - discount, 0) / total + discount * types / total * p
        return p

    def score(self, word: str) -> float:
        """Mean log probability per character, including the end of word."""
        s = f"^{word}$"
        n = len(self.levels)
        logps = [
            math.log(self.prob(s[i], s[max(0, i - n + 1) : i]))
            for i in range(1, len(s))
        ]
        return sum(logps) / len(logps)


def kn_level(data: dict) -> Tuple[float, Dict[str, int], Dict[str, Tuple[int, int]]]:
    counts = data["counts"]
    hists = {}
    for gram, c in counts.items():
        total, types = hists.get(gram[:-1], (0, 0))
        hists[gram[:-1]] = (total + c, types + 1)
    return data["discount"], counts, hists


class NGram:
    """
    Decides whether a word is pronounceable, to transliterate it with C2K,
    or should be spelled out letter by letter with `as_is`.
    A linear SVM over the Kneser-Ney scores of orders 1 to 5, see `e2k_train/hybrid.py`.
    """

    def __init__(self):
        path = get_asset_path("ngram.json.zip")
        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_LZMA) as z:
            with z.open("ngram.json") as f:
                data = json.loads(f.read())

        self.weights = data["weights"]
        self.bias = data["bias"]
        self.threshold = data["threshold"]
        # the levels below each model's top order are shared by all models
        lower = [kn_level(d) for d in data["lower"]]
        self.models = [
            KneserNey(lower[: n - 1] + [kn_level(top)], data["vocab"])
            for n, top in zip(data["orders"], data["top"])
        ]
        # the unigrams are the characters seen in training, and the end of word
        self.valid_chars = set(data["top"][0]["counts"]) - {"$"}

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

    def score(self, word: str) -> float:
        """
        How pronounceable `word` is, valid above `threshold`.
        A phrase scores as its least pronounceable word.
        """
        parts = [
            "".join(c for c in part if c in self.valid_chars)
            for part in word.lower().split()
        ]
        parts = [p for p in parts if p]
        if not parts:
            return -math.inf
        return min(self._score(p) for p in parts)

    def _score(self, word: str) -> float:
        scores = [model.score(word) for model in self.models]
        return sum(w * s for w, s in zip(self.weights, scores)) + self.bias

    def as_is(self, word: str) -> str:
        cleaned = [c for c in word if c in self.spell_table]
        return "".join(self.spell_table[c] for c in cleaned)

    def __call__(self, word: str) -> bool:
        return self.score(word) > self.threshold
