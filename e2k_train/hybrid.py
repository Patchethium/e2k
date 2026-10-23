# a linear SVM over the Kneser-Ney scores of orders 1 to 5, trained on the
# synthetic negatives of `classifier.py`
import argparse
import json
import random
import time
import zipfile
from typing import Dict, List, Tuple

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
from ngram_kn import KneserNeyModel, calibrate, make_splits, per_word

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
    parser.add_argument(
        "--gru",
        action="store_true",
        help="also compare with the GRU of `classifier.py`",
    )
    parser.add_argument(
        "--export",
        help="where to write the runtime model of the first mix, like vendor/ngram.json.zip",
    )
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
        print(
            "  scaled weights "
            + " ".join(f"kn{n}={w:+.2f}" for n, w in zip(ORDERS, clf[-1].coef_[0]))
        )
        weights, bias = fold_scaler(clf)
        score = lambda words, w=weights, b=bias: featurize(words) @ w + b
        models[f"hyb-{tag}"] = score
        if args.gru:
            models[f"gru-{tag}"] = scorer(
                load_or_train(tag, args.seed, train, neg, dev, dev_neg, 64, 5, 256)
            )
        if args.export and mix == args.mixes[0]:
            threshold = calibrate(score, dev, args.recall)
            export(args.export, featurize, weights, bias, threshold)
            print(f"  exported to {args.export}", flush=True)

    report_all(models, dev, sets, held_out, args.recall)


def fold_scaler(clf) -> Tuple[np.ndarray, float]:
    """The weights and bias on raw scores of a scaler + linear SVM pipeline."""
    scaler, svm = clf[0], clf[-1]
    weights = svm.coef_[0] / scaler.scale_
    bias = float(svm.intercept_[0] - weights @ scaler.mean_)
    return weights, bias


def export(
    path: str, featurize: Featurizer, weights: np.ndarray, bias: float, threshold: float
):
    """
    Writes the runtime model: a word is pronounceable when
    `weights @ [score of each order] + bias > threshold`.
    """
    models = featurize.models
    # below its top order, every model counts the same as the highest one does
    lower = models[-1]
    for model in models[:-1]:
        for k in range(1, model.n):
            assert model.counts[k] == lower.counts[k]
            assert model.discounts[k] == lower.discounts[k]

    def level(model: KneserNeyModel, k: int) -> dict:
        return {"discount": model.discounts[k], "counts": model.counts[k]}

    ser = {
        "orders": ORDERS,
        "vocab": lower.vocab,
        # the levels below a model's top order, shared by all models
        "lower": [level(lower, k) for k in range(1, lower.n)],
        # the top level of each model
        "top": [level(model, model.n) for model in models],
        "weights": weights.tolist(),
        "bias": bias,
        "threshold": threshold,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as z:
        with z.open("ngram.json", "w") as f:
            f.write(json.dumps(ser, separators=(",", ":")).encode("utf-8"))


if __name__ == "__main__":
    main()
