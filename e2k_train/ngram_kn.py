# interpolated Kneser-Ney character n-gram model for pronounceability,
# trained and evaluated on cmudict. a candidate replacement for `ngram.py`
import argparse
import contextlib
import io
import math
import random
import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Callable, Dict, List, Tuple

import numpy as np

BOS = "^"
EOS = "$"
WORD_RE = re.compile(r"^[a-z'-]+$")


def load_cmudict(path: str) -> Dict[str, List[List[str]]]:
    """
    Returns {word: [pronunciation, ...]} with stress marks stripped,
    keeping only words made of letters, apostrophes and hyphens.
    """
    prons = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line or line.startswith(";;;"):
                continue
            word, *phones = line.split()
            word = word.split("(")[0]
            if not WORD_RE.match(word):
                continue
            prons[word].append([p.rstrip("012") for p in phones])
    return dict(prons)


def letter_names(prons: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
    """Pronunciation of each letter's name, e.g. `b` -> B IY."""
    names = {c: prons[c][0] for c in ascii_lowercase}
    # `a` is listed as the article AH first, which isn't how acronyms spell it
    names["a"] = ["EY"]
    return names


def edit_distance(a: List[str], b: List[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def is_spelled(
    word: str, phones: List[str], names: Dict[str, List[str]], threshold: float
) -> bool:
    """
    Whether `phones` reads `word` out letter by letter, like fbi -> EH F B IY AY,
    by the edit distance to the spelled-out phonemes, relative to their length.
    """
    if len(word) < 2 or not word.isalpha():
        return False
    spelled = [p for c in word for p in names[c]]
    # the distance is at least the length difference, which rules out most words
    if abs(len(spelled) - len(phones)) / len(spelled) > threshold:
        return False
    return edit_distance(phones, spelled) / len(spelled) <= threshold


def split_cmudict(
    prons: Dict[str, List[List[str]]], threshold: float
) -> Tuple[List[str], List[str]]:
    """
    Splits cmudict into pronounced words and spelled-out words (acronyms).
    Words that have both kinds of pronunciation are dropped.
    """
    names = letter_names(prons)
    pronounced, spelled = [], []
    for word, variants in prons.items():
        flags = [is_spelled(word, p, names, threshold) for p in variants]
        if not any(flags):
            pronounced.append(word)
        elif all(flags):
            spelled.append(word)
    return sorted(pronounced), sorted(spelled)


class KneserNeyModel:
    def __init__(self, n: int):
        self.n = n
        self.counts = [{} for _ in range(n + 1)]  # counts[k]: {gram: adjusted count}
        self.hists = [{} for _ in range(n + 1)]  # hists[k]: {history: (total, types)}
        self.discounts = [0.0] * (n + 1)
        self.vocab = 0

    def _pad(self, word: str) -> str:
        return f"{BOS}{word.lower()}{EOS}"

    def train(self, words: List[str]):
        raw = [defaultdict(int) for _ in range(self.n + 1)]
        preceding = [defaultdict(set) for _ in range(self.n + 1)]
        vocab = set(EOS)
        for word in words:
            s = self._pad(word)
            vocab.update(s[1:])
            for i in range(1, len(s)):
                for k in range(1, min(self.n, i + 1) + 1):
                    gram = s[i - k + 1 : i + 1]
                    raw[k][gram] += 1
                    if k < self.n and gram[0] != BOS:
                        preceding[k][gram].add(s[i - k])
        self.vocab = len(vocab)

        for k in range(1, self.n + 1):
            # lower orders count the distinct characters a gram follows,
            # except for grams at the word start, where nothing comes before
            counts = {
                g: c if k == self.n or g[0] == BOS else len(preceding[k][g])
                for g, c in raw[k].items()
            }
            n1 = sum(1 for c in counts.values() if c == 1)
            n2 = sum(1 for c in counts.values() if c == 2)
            self.discounts[k] = n1 / (n1 + 2 * n2) if n1 + n2 > 0 else 0.5

            hists = defaultdict(lambda: [0, 0])
            for g, c in counts.items():
                hists[g[:-1]][0] += c
                hists[g[:-1]][1] += 1
            self.counts[k] = counts
            self.hists[k] = {h: tuple(v) for h, v in hists.items()}

    def prob(self, char: str, hist: str) -> float:
        """P(char | hist), interpolated down to a uniform distribution."""
        p = 1.0 / self.vocab
        for k in range(1, len(hist) + 2):
            h = hist[len(hist) - k + 1 :]
            if h not in self.hists[k]:
                continue
            total, types = self.hists[k][h]
            d = self.discounts[k]
            c = self.counts[k].get(h + char, 0)
            p = max(c - d, 0) / total + d * types / total * p
        return p

    def score(self, word: str) -> float:
        """Mean log probability per character, including the end of word."""
        s = self._pad(word)
        logps = []
        for i in range(1, len(s)):
            hist = s[max(0, i - self.n + 1) : i]
            logps.append(math.log(self.prob(s[i], hist)))
        return sum(logps) / len(logps)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Chance that a random positive outscores a random negative."""
    scores = np.concatenate([pos, neg])
    ordered = np.sort(scores)
    # tied scores share the mean of the ranks they span
    ranks = (
        np.searchsorted(ordered, scores, "left")
        + np.searchsorted(ordered, scores, "right")
        + 1
    ) / 2
    r = ranks[: len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


Scorer = Callable[[List[str]], np.ndarray]


def per_word(score: Callable[[str], float]) -> Scorer:
    # unseen grams score -inf in the shipped model, which breaks the ranks
    return lambda words: np.nan_to_num([score(w) for w in words], neginf=-1e9)


def make_splits(
    cmudict: str, spelled_threshold: float, seed: int
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Returns the training words, the dev words for calibrating thresholds and
    the evaluation sets: held-out test words, acronyms and random strings.
    """
    pronounced, spelled = split_cmudict(load_cmudict(cmudict), spelled_threshold)
    rng = random.Random(seed)
    rng.shuffle(pronounced)
    n_held = len(pronounced) // 10
    test, dev, train = (
        pronounced[:n_held],
        pronounced[n_held : 2 * n_held],
        pronounced[2 * n_held :],
    )
    randoms = [
        "".join(rng.choice(ascii_lowercase) for _ in range(len(w))) for w in test
    ]
    print(
        f"train {len(train)}, dev {len(dev)}, test {len(test)}, "
        f"spelled-out {len(spelled)}, random {len(randoms)}"
    )
    return train, dev, {"test": test, "spelled": spelled, "random": randoms}


def calibrate(score: Scorer, dev: List[str], recall: float) -> float:
    """The threshold that keeps `recall` of the dev words."""
    return float(np.quantile(score(dev), 1 - recall))


def print_header():
    print(
        f"{'model':<16} {'threshold':>9} {'recall':>8} {'rej.spell':>9} "
        f"{'rej.rand':>9} {'AUC':>8}"
    )


def report(name: str, score: Scorer, threshold: float, sets: Dict[str, List[str]]):
    scores = {k: score(v) for k, v in sets.items()}
    recall = (scores["test"] > threshold).mean()
    rej_spelled = (scores["spelled"] <= threshold).mean()
    rej_random = (scores["random"] <= threshold).mean()
    print(
        f"{name:<16} {threshold:>9.3f} {recall:>8.2%} {rej_spelled:>9.2%} "
        f"{rej_random:>9.2%} {auc(scores['test'], scores['spelled']):>8.4f}"
    )


def shipped_model():
    from e2k import NGram

    ngram = NGram()

    def score(word: str) -> float:
        cleaned = "".join(c for c in word.lower() if c in ngram.valid_chars)
        # the runtime prints a warning for every unseen gram
        with contextlib.redirect_stdout(io.StringIO()):
            return ngram.score(cleaned)

    return per_word(score), ngram.threshold


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kneser-Ney n-gram models.")
    parser.add_argument("--cmudict", default="vendor/cmudict.dict")
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    parser.add_argument(
        "--recall", type=float, default=0.95, help="dev recall the threshold keeps"
    )
    parser.add_argument(
        "--spelled-threshold",
        type=float,
        default=0.18,
        help="relative phoneme edit distance under which a word counts as spelled out",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, dev, sets = make_splits(args.cmudict, args.spelled_threshold, args.seed)
    print_header()

    # the shipped model was trained on all of cmudict, test and acronyms included
    score, threshold = shipped_model()
    report("shipped", score, threshold, sets)
    report("shipped@", score, calibrate(score, dev, args.recall), sets)
    for n in args.orders:
        model = KneserNeyModel(n)
        model.train(train)
        score = per_word(model.score)
        report(f"kn-{n}", score, calibrate(score, dev, args.recall), sets)


if __name__ == "__main__":
    main()
