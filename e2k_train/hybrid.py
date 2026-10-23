# a linear SVM over the Kneser-Ney scores of orders 1 to 5, trained on the
# synthetic negatives of `classifier.py`
import argparse
import random
import time
from typing import Dict, List

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from classifier import (
    held_out_negatives,
    load_or_train,
    mix_negatives,
    report_all,
    scorer,
)
from ngram_kn import KneserNeyModel, make_splits, per_word

ORDERS = [1, 2, 3, 4, 5]


class Featurizer:
    """Per word: the mean log probability per character under each order."""

    def __init__(self, words: List[str]):
        self.models = []
        for n in ORDERS:
            model = KneserNeyModel(n)
            model.train(words)
            self.models.append(model)
        self.cache: Dict[str, List[float]] = {}

    def row(self, word: str) -> List[float]:
        if word not in self.cache:
            self.cache[word] = [m.score(word) for m in self.models]
        return self.cache[word]

    def __call__(self, words: List[str]) -> np.ndarray:
        return np.array([self.row(w) for w in words])


def out_of_fold(pos: List[str], neg: List[str], folds: int, seed: int) -> np.ndarray:
    """
    Features of `pos + neg`, each fold scored by models trained on the
    positives of the other folds, so positives aren't scored by models that saw them.
    """
    rng = random.Random(seed)
    fold_of = [rng.randrange(folds) for _ in range(len(pos) + len(neg))]
    words = pos + neg
    out = np.zeros((len(words), len(ORDERS)))
    for f in range(folds):
        featurize = Featurizer([w for w, k in zip(pos, fold_of) if k != f])
        idx = [i for i, k in enumerate(fold_of) if k == f]
        out[idx] = featurize([words[i] for i in idx])
    return out


def main():
    parser = argparse.ArgumentParser(description="Train SVMs on Kneser-Ney scores.")
    parser.add_argument("--cmudict", default="vendor/cmudict.dict")
    parser.add_argument("--recall", type=float, default=0.95)
    parser.add_argument("--spelled-threshold", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mixes",
        nargs="+",
        default=[
            "random,initialism,devowel",
            "random,initialism,devowel,corrupt,shuffle",
        ],
    )
    parser.add_argument("--per-kind", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    train, dev, sets = make_splits(args.cmudict, args.spelled_threshold, args.seed)
    exclude, held_out = held_out_negatives(args.cmudict, sets, args.per_kind, args.seed)

    start = time.perf_counter()
    featurize = Featurizer(train)
    print(f"n-gram models trained in {time.perf_counter() - start:.0f}s", flush=True)
    models = {"kn-5": per_word(featurize.models[-1].score)}
    for mix in args.mixes:
        tag, neg, dev_neg = mix_negatives(mix, train, dev, exclude, args.seed)
        start = time.perf_counter()
        x = out_of_fold(train, neg, args.folds, args.seed)
        y = np.array([1] * len(train) + [0] * len(neg))
        print(
            f"{tag}: out-of-fold features in {time.perf_counter() - start:.0f}s",
            flush=True,
        )

        clf = make_pipeline(StandardScaler(), LinearSVC(C=args.c)).fit(x, y)
        weights = clf[-1].coef_[0]
        print(
            "  weights " + " ".join(f"kn{n}={w:+.2f}" for n, w in zip(ORDERS, weights))
        )
        models[f"hyb-{tag}"] = lambda words, clf=clf: clf.decision_function(
            featurize(words)
        )
        models[f"gru-{tag}"] = scorer(
            load_or_train(tag, args.seed, train, neg, dev, dev_neg, 64, 5, 256)
        )

    report_all(models, dev, sets, held_out, args.recall)


if __name__ == "__main__":
    main()
