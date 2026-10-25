"""
Evaluates models through the numpy runtime of `e2k`, on the test split of
`train.split`: the words no model is trained on.

C2K/P2K: BLEU (3-gram over katakana, against every reference spelling), exact
match against any reference, and CER against the closest one.
Accent: accuracy, and precision, recall and F1 over accent core positions,
weighted by support (macro too, it's dominated by positions of a few words).

Run from the repository root, by default on every shipped model:
    uv run python -m e2k_train.eval
    uv run python -m e2k_train.eval --task c2k --path model-c2k.safetensors
"""

import argparse
import time
from os import path

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import precision_recall_fscore_support

from e2k import C2K, P2K, AccentPredictor
from e2k_train.hp import en_phones
from e2k_train.train import MyDataset, split

VENDOR = path.join(path.dirname(path.abspath(__file__)), "vendor")


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def score(preds, refs):
    """exact match against any reference, CER against the closest one"""
    exact, dist, length = 0, 0, 0
    for pred, ref in zip(preds, refs):
        exact += pred in ref
        d, r = min((edit_distance(pred, r), r) for r in ref)
        dist += d
        length += len(r)
    return exact / len(preds), dist / length


def bleu(preds, refs):
    """
    corpus BLEU over 1 to 3-grams, every katakana is one symbol, so a token;
    corpus level, as many words are shorter than 3 symbols
    """
    refs = [[list(r) for r in ref] for ref in refs]
    return corpus_bleu(refs, [list(p) for p in preds], weights=(1 / 3, 1 / 3, 1 / 3))


def test_s2s(task: str, data: str):
    """the test inputs of `task`, and their reference spellings"""
    p2k = task == "p2k"
    ds = MyDataset(data, torch.device("cpu"), p2k=p2k)
    _, _, test = split(len(ds))
    words = [ds.data[i]["word"] for i in test]
    # P2K takes the phonemes the training data used
    srcs = [[en_phones[j] for j in ds.p2k(w)] for w in words] if p2k else words
    return srcs, [ds.data[i]["kata"] for i in test]


def eval_s2s(model, srcs, refs):
    model(srcs[0])
    t0 = time.perf_counter()
    preds = [model(s) for s in srcs]
    ms = (time.perf_counter() - t0) / len(srcs) * 1000
    em, cer = score(preds, refs)
    return {"BLEU": bleu(preds, refs), "EM": em, "CER": cer, "ms/word": ms}


def eval_accent(model, lex: str):
    # imported here, as it pulls the training code of the accent model
    from e2k_train.accent import AccentDataset

    ds = AccentDataset(lex)
    _, _, test = split(len(ds))
    labels = [int(ds.data[i][1]) for i in test]
    preds = [model(ds.data[i][0]) for i in test]
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    r = {"n": len(test), "accuracy": acc}
    for average in ["weighted", "macro"]:
        prf = precision_recall_fscore_support(
            labels, preds, average=average, zero_division=0
        )
        r[average] = dict(zip(["precision", "recall", "F1"], prf[:3]))
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=["c2k", "p2k", "accent"], nargs="+")
    parser.add_argument("--path", type=str, help="a model file instead of the shipped ones")
    parser.add_argument("--data", default=path.join(VENDOR, "katakana_dict.jsonl"))
    parser.add_argument("--lex", default=path.join(VENDOR, "lex.csv"))
    args = parser.parse_args()
    tasks = args.task or ["c2k", "p2k", "accent"]
    if args.path and len(tasks) != 1:
        parser.error("--path needs a single --task")

    for task in tasks:
        if task == "accent":
            r = eval_accent(AccentPredictor(path=args.path), args.lex)
            print(f"accent ({r['n']} entries): accuracy {r['accuracy']:.4f}")
            for average in ["weighted", "macro"]:
                m = r[average]
                print(
                    f"  {average}: precision {m['precision']:.4f} "
                    f"recall {m['recall']:.4f} F1 {m['F1']:.4f}"
                )
            continue
        srcs, refs = test_s2s(task, args.data)
        cls = P2K if task == "p2k" else C2K
        variants = [("file", cls(path=args.path))] if args.path else [
            (m, cls(model=m)) for m in ["base", "turbo"]
        ]
        for name, model in variants:
            r = eval_s2s(model, srcs, refs)
            print(
                f"{task} {name} ({len(srcs)} words): BLEU {r['BLEU']:.4f} "
                f"EM {r['EM']:.4f} CER {r['CER']:.4f} ({r['ms/word']:.2f} ms/word)"
            )


if __name__ == "__main__":
    main()
