# the n-gram model used to determine if a word is spellable
import statistics as st
from collections import defaultdict
from typing import List, Optional
from string import ascii_lowercase
import json
import os
import argparse
import itertools
import math
import zipfile


VALID_CHARS = list(
    set(ascii_lowercase + "$^-'")
)  # valid characters in the n-gram model


class NGramCollection:
    def __init__(self, ns: List[int], weights: Optional[float]):
        if weights is not None:
            assert sum(weights) - 1.0 < 1e-6  # ensure weights sum to 1
        self.ns = ns
        self.models = [NGramModel(n) for n in ns]
        self.weights = weights
        self.threshold = 0.0

    def train(self, words: List[str]):
        for model in self.models:
            model.train(words)
        scores = []
        for word in words:
            score = self.score(word)
            if score < -100000:
                continue
            scores.append(score)
        # remove scores too low
        scores = [s for s in scores if s > -10]
        if len(scores) == 0:
            self.threshold = 0.0
        else:
            mean = st.mean(scores)
            std = st.stdev(scores)
            self.threshold = mean - 2.0 * std

    def score(self, word: str) -> float:
        """
        Score the log likelihood of a word being spelled correctly.
        """
        scores = []
        for model in self.models:
            score = model.score(word)
            scores.append(score)
        if self.weights is None:
            return st.mean(scores)
        else:
            return sum([w * s for w, s in zip(self.weights, scores)])

    def isvalid(self, word: str) -> bool:
        """
        Check if a word is valid.
        """
        word = word.lower()
        score = self.score(word)
        valid = score > self.threshold
        print(
            f"{word}: {'Valid' if valid else 'Invalid'}, score: {round(score, 4)}, threshold: {round(self.threshold, 4)}"
        )
        return valid

    def serialize(self) -> str:
        models = {n: model.freq for n, model in zip([n for n in self.ns], self.models)}
        ser = {
            "valid_chars": VALID_CHARS,
            "models": models,
            "weights": self.weights,
            "threshold": self.threshold,
        }
        return json.dumps(ser, indent=None, ensure_ascii=False)

    def deserialize(self, ser: str):
        """
        Deserialize the model from a json string.
        """
        ser = json.loads(ser)
        self.weights = ser["weights"]
        self.threshold = ser["threshold"]
        valid_chars = ser["valid_chars"]
        models = ser["models"]
        self.models = {NGramModel(int(k), v, valid_chars) for k, v in models.items()}


class NGramModel:
    def __init__(self, n, freq={}, valid_chars=VALID_CHARS):
        self.n = n
        self.freq = freq  # {prefix: {appendix: count}}
        self.valid_chars = valid_chars

    def _ngram(self, word: str) -> List[str]:
        word = word.lower()
        word = f"^{word}$"
        return [word[i : i + self.n] for i in range(len(word) - self.n + 1)]

    def _possible_ngrams(self):
        """
        Returns all possible prefix given gram number
        """
        return [
            "".join(x) for x in itertools.product(self.valid_chars, repeat=self.n - 1)
        ]

    def train(self, words: List[str]):
        # clear & prepare the freq
        self.freq = defaultdict(lambda: defaultdict(int))
        for word in words:
            ngrams = self._ngram(word)
            for ngram in ngrams:
                prefix = ngram[:-1]
                appendix = ngram[-1]
                self.freq[prefix][appendix] += 1
                self.freq[prefix]["total"] += 1
        # laplace smoothing across all possible ngrams
        possible_ngrams = self._possible_ngrams()
        for prefix in possible_ngrams:
            for appendix in self.valid_chars:
                self.freq[prefix][appendix] += 1
            self.freq[prefix]["total"] += len(self.valid_chars)
        # normalize
        for prefix in self.freq:
            total = self.freq[prefix]["total"]
            for appendix in self.valid_chars:
                self.freq[prefix][appendix] /= total
                self.freq[prefix][appendix] = round(self.freq[prefix][appendix], 4)
            # remove total
            del self.freq[prefix]["total"]
        # convert back to normal dict
        self.freq = dict(self.freq)

    def _score(self, word: str) -> float:
        """
        Score the log likelihood of a word being spelled correctly.
        """
        ngrams = self._ngram(word)
        scores = []
        for ngram in ngrams:
            prefix = ngram[:-1]
            appendix = ngram[-1]
            if prefix not in self.freq or appendix not in self.freq[prefix]:
                print(f"Not found: {prefix} {appendix} in {self.n}-gram model")
                return -float("inf")
            scores.append(
                math.log(
                    self.freq[prefix][appendix] + 1e-10
                )  # add a small value to avoid log(0)
            )
        if len(scores) == 0:
            return -float("inf")
        likelihood = st.mean(scores)
        return likelihood

    def score(self, word: str) -> float:
        """
        Handles the case where whitespace is present in the word.
        we split the word and return the mean
        """
        words = word.split(" ")
        likelihoods = [self._score(w) for w in words]
        if len(likelihoods) == 0:
            return -float("inf")
        return st.mean(likelihoods)


def load_cmudict(path: str) -> List[str]:
    # returns words in cmudict
    with open(path, "r") as f:
        lines = f.readlines()
    words = []
    for line in lines:
        if line.startswith(";;;"):
            continue
        word = line.split(" ")[0]
        word = word.split("(")[0]
        words.append(word)
    return list(set(words))


def main():
    parser = argparse.ArgumentParser(description="Train or load the n-gram model.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()
    ngram = NGramCollection([2, 3], [0.6, 0.4])
    path = "vendor/ngram.json.zip"
    if os.path.exists(path) and not args.train:
        with zipfile.ZipFile(path, "r", compression=zipfile.ZIP_LZMA) as z:
            with z.open("ngram.json") as f:
                ser = f.read()
        ngram.deserialize(ser.decode("utf-8"))
    else:
        words = load_cmudict("vendor/cmudict.dict")
        ngram.train(words)
        ser = ngram.serialize()
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as z:
            with z.open("ngram.json", "w") as f:
                f.write(ser.encode("utf-8"))
        print("Training finished.")
    ngram.isvalid("hello")
    ngram.isvalid("helloo")
    ngram.isvalid("helxo")
    ngram.isvalid("tar")
    ngram.isvalid("tarball")
    ngram.isvalid("dog")
    ngram.isvalid("doggy")
    ngram.isvalid("confield")
    ngram.isvalid("Tri Repetae")
    ngram.isvalid("autechre")
    ngram.isvalid("vordhosbn")
    ngram.isvalid("aphex twin")
    ngram.isvalid("tgc")
    ngram.isvalid("gcc")
    ngram.isvalid("mvp")
    ngram.isvalid("ussr")
    ngram.isvalid("rogbiv")


if __name__ == "__main__":
    main()
