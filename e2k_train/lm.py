# character language models, a GRU and a causal transformer, trained on
# pronounced words alone. a word's score is its mean log probability per
# character, the negative log perplexity, like the Kneser-Ney model of
# `ngram_kn.py`, so no negatives are needed for training, only for evaluation
import argparse
import math
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from classifier import held_out_negatives, report_all
from ngram_kn import KneserNeyModel, Scorer, load_cmudict, make_splits, per_word

SEED = 3407
CHARS = "abcdefghijklmnopqrstuvwxyz'-"
PAD, BOS, EOS = 0, 1, 2
CHAR_IDX = {c: i + 3 for i, c in enumerate(CHARS)}
VOCAB = len(CHARS) + 3
MAX_LEN = 48  # cmudict's longest word is 34 characters


class GRULM(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.rnn = nn.GRU(
            dim, dim, layers, batch_first=True, dropout=dropout if layers > 1 else 0
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T], returns next character logits [B, T, VOCAB]"""
        # right padding only follows the real characters, so it can't leak into them
        h, _ = self.rnn(self.drop(self.emb(x)))
        return self.fc(self.drop(h))


class TransformerLM(nn.Module):
    def __init__(self, dim: int, layers: int, dropout: float, heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, VOCAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T], returns next character logits [B, T, VOCAB]"""
        t = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(t, device=x.device))
        # causal, so real characters never attend to the padding after them
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.fc(self.norm(h))


ARCHS = {"gru": GRULM, "transformer": TransformerLM}


def encode(words: List[str]):
    """
    Inputs `^word` and targets `word$` for each word, right padded.
    Characters outside `CHARS` are dropped.
    """
    seqs = [
        [BOS] + [CHAR_IDX[c] for c in w.lower() if c in CHAR_IDX][: MAX_LEN - 2] + [EOS]
        for w in words
    ]
    x = pad_sequence([torch.tensor(s[:-1]) for s in seqs], batch_first=True)
    y = pad_sequence(
        [torch.tensor(s[1:]) for s in seqs], batch_first=True, padding_value=PAD
    )
    return x, y


def token_logps(model: nn.Module, words: List[str]):
    """Log probability of each target character, [B, T], and the mask of real ones."""
    x, y = encode(words)
    logp = model(x).log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    mask = y != PAD
    return logp * mask, mask


def scorer(model: nn.Module, reduce: str = "mean", batch_size: int = 2048) -> Scorer:
    """
    Scores words by the mean log probability per character, including the end
    of word (`mean`), or by the least likely character (`min`).
    """

    @torch.no_grad()
    def score(words: List[str]) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(words), batch_size):
            logp, mask = token_logps(model, words[i : i + batch_size])
            if reduce == "mean":
                out.append(logp.sum(-1) / mask.sum(-1))
            else:
                out.append(logp.masked_fill(~mask, math.inf).min(-1).values)
        return torch.cat(out).numpy()

    return score


def nll(model: nn.Module, words: List[str]) -> float:
    """Mean negative log likelihood per character, in nats."""
    with torch.no_grad():
        model.eval()
        total, count = 0.0, 0
        for i in range(0, len(words), 2048):
            logp, mask = token_logps(model, words[i : i + 2048])
            total -= logp.sum().item()
            count += mask.sum().item()
    return total / count


def train_lm(
    arch: str,
    train: List[str],
    dev: List[str],
    dim: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Module:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = ARCHS[arch](dim, layers, dropout)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, lr, total_steps=steps)

    data = list(train)
    best, best_state = math.inf, None
    for epoch in range(epochs):
        model.train()
        rng.shuffle(data)
        start = time.perf_counter()
        for i in range(0, len(data), batch_size):
            logp, mask = token_logps(model, data[i : i + batch_size])
            loss = -logp.sum() / mask.sum()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
        dev_nll = nll(model, dev)
        print(
            f"  epoch {epoch + 1}: dev nll {dev_nll:.4f} "
            f"(ppl {math.exp(dev_nll):.2f}), {time.perf_counter() - start:.0f}s",
            flush=True,
        )
        if dev_nll < best:
            best = dev_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def load_or_train(args: argparse.Namespace, arch: str, train, dev) -> nn.Module:
    """Loads the LM of this setup from `vendor/`, training it if absent."""
    tag = "all" if args.all else f"s{args.seed}"
    path = (
        f"vendor/lm-{arch}-d{args.dim}-l{args.layers}-e{args.epochs}"
        f"-b{args.batch_size}-{tag}.pth"
    )
    model = ARCHS[arch](args.dim, args.layers, args.dropout)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"  loaded {path}", flush=True)
        return model
    start = time.perf_counter()
    model = train_lm(
        arch,
        train,
        dev,
        args.dim,
        args.layers,
        args.dropout,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    torch.save(model.state_dict(), path)
    print(f"  trained in {time.perf_counter() - start:.0f}s, saved {path}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train character LMs and threshold their perplexity."
    )
    parser.add_argument("--cmudict", default="vendor/cmudict.dict")
    parser.add_argument("--recall", type=float, default=0.95)
    parser.add_argument("--spelled-threshold", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--per-kind", type=int, default=2000)
    parser.add_argument(
        "--all",
        action="store_true",
        help="train on every cmudict word, acronyms included, for shipping; "
        "the evaluation then scores words the model has seen",
    )
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())

    train, dev, sets = make_splits(args.cmudict, args.spelled_threshold, args.seed)
    exclude, held_out = held_out_negatives(args.cmudict, sets, args.per_kind, args.seed)
    lm_train = sorted(load_cmudict(args.cmudict)) if args.all else train

    kn = KneserNeyModel(5)
    kn.train(train)
    models: Dict[str, Scorer] = {"kn-5": per_word(kn.score)}
    for arch in args.archs:
        print(f"{arch}: {len(lm_train)} words", flush=True)
        model = load_or_train(args, arch, lm_train, dev)
        print(f"  test nll {nll(model, sets['test']):.4f}", flush=True)
        models[f"{arch}"] = scorer(model, "mean")
        models[f"{arch}-min"] = scorer(model, "min")

    report_all(models, dev, sets, held_out, args.recall)


if __name__ == "__main__":
    main()
