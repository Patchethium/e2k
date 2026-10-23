# a character GRU that tells pronounceable words from synthetic negatives,
# evaluated on the same splits as the Kneser-Ney model in `ngram_kn.py`
import argparse
import os
import random
import time
from string import ascii_lowercase
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from ngram_kn import (
    KneserNeyModel,
    Scorer,
    calibrate,
    load_cmudict,
    make_splits,
    per_word,
    print_header,
    report,
)

SEED = 3407
CHARS = ascii_lowercase + "'-"
CHAR_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is padding
VOWELS = set("aeiouy")
CONSONANTS = [c for c in ascii_lowercase if c not in VOWELS]


def random_string(rng: random.Random, words: List[str]) -> str:
    return "".join(rng.choice(ascii_lowercase) for _ in rng.choice(words))


def initialism(rng: random.Random, words: List[str]) -> str:
    """First letters of a few words, like an acronym: nba, fbi."""
    return "".join(rng.choice(words)[0] for _ in range(rng.randint(2, 5)))


def devowel(rng: random.Random, words: List[str]) -> str:
    """A word with its vowels dropped, like an abbreviation: management -> mngmnt."""
    return "".join(c for c in rng.choice(words) if c not in VOWELS)


def corrupt(rng: random.Random, words: List[str]) -> str:
    """A word with two edits that pile up consonants."""
    chars = list(rng.choice(words))
    for _ in range(2):
        i = rng.randrange(len(chars))
        op = rng.choice(["sub", "ins", "del"])
        if op == "sub":
            chars[i] = rng.choice(CONSONANTS)
        elif op == "ins":
            chars.insert(i, rng.choice(CONSONANTS))
        elif chars[i] in VOWELS and len(chars) > 2:
            del chars[i]
    return "".join(chars)


def shuffle(rng: random.Random, words: List[str]) -> str:
    chars = list(rng.choice(words))
    rng.shuffle(chars)
    return "".join(chars)


GENERATORS: Dict[str, Callable[[random.Random, List[str]], str]] = {
    "random": random_string,
    "initialism": initialism,
    "devowel": devowel,
    "corrupt": corrupt,
    "shuffle": shuffle,
}


def negatives(
    kind: str, words: List[str], n: int, exclude: Set[str], rng: random.Random
) -> List[str]:
    """`n` distinct negatives of one kind, none of them in `exclude`."""
    # only letters make sensible initials and edits
    words = [w for w in words if w.isalpha()]
    generate = GENERATORS[kind]
    out = set()
    for _ in range(n * 20):
        w = generate(rng, words)
        if len(w) >= 2 and w not in exclude:
            out.add(w)
        if len(out) == n:
            break
    return sorted(out)


class Classifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.emb = nn.Embedding(len(CHARS) + 1, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]
        lengths: [B]
        returns the logit of being pronounceable, [B]
        """
        packed = pack_padded_sequence(
            self.emb(x), lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)  # [2, B, dim]
        return self.fc(torch.cat([h[0], h[1]], dim=-1)).squeeze(-1)


def encode(words: List[str]):
    seqs = [
        torch.tensor([CHAR_IDX[c] for c in w if c in CHAR_IDX] or [0]) for w in words
    ]
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True), lengths


def scorer(model: Classifier, batch_size: int = 1024):
    @torch.no_grad()
    def score(words: List[str]) -> np.ndarray:
        model.eval()
        logits = []
        for i in range(0, len(words), batch_size):
            logits.append(model(*encode(words[i : i + batch_size])))
        return torch.cat(logits).numpy()

    return score


def train_classifier(
    pos: List[str],
    neg: List[str],
    dev_pos: List[str],
    dev_neg: List[str],
    dim: int,
    epochs: int,
    batch_size: int,
) -> Classifier:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = Classifier(dim)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    data = [(w, 1.0) for w in pos] + [(w, 0.0) for w in neg]
    dev_words = dev_pos + dev_neg
    dev_labels = torch.tensor([1.0] * len(dev_pos) + [0.0] * len(dev_neg))
    score = scorer(model)

    best, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        rng.shuffle(data)
        for i in range(0, len(data), batch_size):
            words, labels = zip(*data[i : i + batch_size])
            loss = loss_fn(model(*encode(words)), torch.tensor(labels))
            optim.zero_grad()
            loss.backward()
            optim.step()
        dev_loss = loss_fn(torch.from_numpy(score(dev_words)), dev_labels).item()
        print(f"  epoch {epoch + 1}: dev loss {dev_loss:.4f}", flush=True)
        if dev_loss < best:
            best = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def load_or_train(
    tag: str,
    seed: int,
    pos: List[str],
    neg: List[str],
    dev_pos: List[str],
    dev_neg: List[str],
    dim: int,
    epochs: int,
    batch_size: int,
) -> Classifier:
    """Loads the classifier of this mix and setup from `vendor/`, training it if absent."""
    path = f"vendor/gru-{tag}-d{dim}-e{epochs}-b{batch_size}-s{seed}.pth"
    model = Classifier(dim)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"  loaded {path}", flush=True)
        return model
    start = time.perf_counter()
    model = train_classifier(pos, neg, dev_pos, dev_neg, dim, epochs, batch_size)
    torch.save(model.state_dict(), path)
    print(f"  trained in {time.perf_counter() - start:.0f}s, saved {path}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train pronounceability classifiers.")
    parser.add_argument("--cmudict", default="vendor/cmudict.dict")
    parser.add_argument("--recall", type=float, default=0.95)
    parser.add_argument("--spelled-threshold", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mixes",
        nargs="+",
        default=["random", "random,initialism,devowel", ",".join(GENERATORS)],
        help="comma-separated negative kinds, one classifier per mix",
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--per-kind", type=int, default=2000, help="held-out negatives of each kind"
    )
    args = parser.parse_args()

    train, dev, sets = make_splits(args.cmudict, args.spelled_threshold, args.seed)
    exclude, held_out = held_out_negatives(args.cmudict, sets, args.per_kind, args.seed)

    kn = KneserNeyModel(5)
    kn.train(train)
    models = {"kn-5": per_word(kn.score)}
    for mix in args.mixes:
        tag, neg, dev_neg = mix_negatives(mix, train, dev, exclude, args.seed)
        name = f"gru-{tag}"
        print(f"{name}: {len(train)} positives, {len(neg)} negatives", flush=True)
        model = load_or_train(
            tag,
            args.seed,
            train,
            neg,
            dev,
            dev_neg,
            args.dim,
            args.epochs,
            args.batch_size,
        )
        models[name] = scorer(model)

    report_all(models, dev, sets, held_out, args.recall)


def held_out_negatives(
    cmudict: str, sets: Dict[str, List[str]], per_kind: int, seed: int
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """
    Returns the words no negative may be and the held-out negatives of each
    kind, drawn from test words.
    """
    # no real word, and none of the acronyms used for testing, is a negative
    exclude = set(load_cmudict(cmudict)) | set(sets["spelled"])
    rng = random.Random(seed)
    held_out = {
        k: negatives(k, sets["test"], per_kind, exclude, rng) for k in GENERATORS
    }
    return exclude, held_out


def mix_negatives(
    mix: str, train: List[str], dev: List[str], exclude: Set[str], seed: int
) -> Tuple[str, List[str], List[str]]:
    """Training and dev negatives for a comma-separated mix of kinds, as many as `train`."""
    # seeded by the mix, so a saved classifier matches its negatives in any script
    rng = random.Random(f"{seed}:{mix}")
    kinds = mix.split(",")
    n = len(train) // len(kinds)
    neg = [w for k in kinds for w in negatives(k, train, n, exclude, rng)]
    dev_neg = [w for k in kinds for w in negatives(k, dev, n // 8, exclude, rng)]
    return "+".join(k[0] for k in kinds), neg, dev_neg


def report_all(
    models: Dict[str, Scorer],
    dev: List[str],
    sets: Dict[str, List[str]],
    held_out: Dict[str, List[str]],
    recall: float,
):
    print_header()
    thresholds = {}
    for name, score in models.items():
        thresholds[name] = calibrate(score, dev, recall)
        report(name, score, thresholds[name], sets)

    print("\nrejection of held-out negatives by kind")
    print(f"{'model':<16} " + " ".join(f"{k:>10}" for k in GENERATORS))
    for name, score in models.items():
        rej = [(score(held_out[k]) <= thresholds[name]).mean() for k in GENERATORS]
        print(f"{name:<16} " + " ".join(f"{r:>10.2%}" for r in rej))

    if "kn-5" not in models:
        return
    # negatives the n-gram accepts are either its misses or label noise
    score, threshold = models["kn-5"], thresholds["kn-5"]
    print("\nheld-out negatives that kn-5 accepts")
    for k in GENERATORS:
        accepted = [w for w, s in zip(held_out[k], score(held_out[k])) if s > threshold]
        print(f"{k:<10} {' '.join(accepted[:15])}")


if __name__ == "__main__":
    main()
