# English to Katakana Translator

[![PyPI version](https://badge.fury.io/py/e2k.svg)](https://badge.fury.io/py/e2k)

`e2k` is a Python library that translates English to Katakana using lightweight RNN models trained on data from Wiktionary and JMdict/EDICT.
It only requires `numpy` as a runtime dependency.
A Rust port with the same models and outputs lives in [`e2k-rs`](./e2k-rs/README.md).

We also provide pretrained models and dictionaries in the **Releases** section (not included in the PyPI package).

## Installation

```bash
pip install e2k
```

## Usage

### 1. Basic Translation

Two models are available:

| Model | Input Type | Description                                  |
| ----- | ---------- | -------------------------------------------- |
| `P2K` | Phonemes   | Converts phonemes (from CMUdict) to Katakana |
| `C2K` | Characters | Converts English text directly to Katakana   |

```python
from e2k import P2K, C2K
from g2p_en import G2p # any g2p library with CMUdict phonemes can work

p2k = P2K()
g2p = G2p()

word = "geogaddi"
print(p2k(g2p(word)))  # ジオダギー

c2k = C2K()
print(c2k(word))       # ジオガディ
```

> [!NOTE]
> While defaulting on `greedy`, both models support `top_k` and `top_p` decoding strategies. It varies the output between each inference, use when you know what you're doing. Pass `seed` to make it reproducible.

```python
c2k(word, "top_p", p=0.9, t=2)
c2k(word, "top_k", k=5, seed=42)
```

Both models also come in a `turbo` variant, a smaller model distilled from the default one. It's about twice as fast with comparable accuracy, though its outputs differ on some words.

```python
c2k = C2K(model="turbo")
p2k = P2K(model="turbo")
```

### 2. Pitch Accent Prediction

The library includes an RNN-based pitch accent predictor trained on [Unidic](https://clrd.ninjal.ac.jp/unidic/).

```python
from e2k import AccentPredictor as Ap, C2K

ap = Ap()
c2k = C2K()

word = "geogaddi"
katakana = c2k(word)
accent = ap(katakana)

print(f"Katakana: {katakana}, Accent: {accent}")
# Katakana: ジオガディ, Accent: 2
```

### 3. Pronunciation Validity (N-Gram Model)

You can check if a word should be transliterated or spelled as-is. A linear SVM over character n-gram scores decides, trained on CMUdict words against synthetic acronyms, abbreviations and random strings.

```python
from e2k import NGram
ngram = NGram()

word = "ussr"
if ngram(word):
    print(c2k(word))
else:
    print(ngram.as_is(word))
```

> Example output:
>
> - `ussr` → ユーエスエスアール
> - `doggy` → ドッギー

### 4. Input/Output Symbol Tables

```python
c2k.in_table  # valid characters
p2k.in_table  # valid phonemes
c2k.out_table # katakana output symbols
p2k.out_table # katakana output symbols
accent.in_table # valid katakana symbols for accent prediction
```

> [!WARNING]
> Symbols not in the `in_table` are ignored and may yield unexpected results.

## Performance

Evaluated with [`e2k_train/eval.py`](./e2k_train/eval.py) on held-out test splits no model was trained on: 1,566 dictionary words for katakana prediction, 37,594 Unidic entries for accent prediction.

### Katakana Prediction

| Model | Variant | BLEU ↑ | Exact Match ↑ | CER ↓ |
| ----- | ------- | ------ | ------------- | ----- |
| C2K   | base    | 0.843  | 67.1%         | 12.8% |
| C2K   | turbo   | 0.846  | 67.9%         | 12.7% |
| P2K   | base    | 0.776  | 56.2%         | 17.9% |
| P2K   | turbo   | 0.778  | 55.8%         | 17.6% |

> [!NOTE]
> BLEU is over 1 to 3-grams of katakana, exact match counts any of a word's reference spellings, and CER is against the closest one.

### Accent Prediction

| Model           | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑  |
| --------------- | ---------- | ----------- | -------- | ----- |
| AccentPredictor | 88.6%      | 89.1%       | 88.6%    | 88.8% |

> [!NOTE]
> Precision, recall and F1 are averaged over accent core positions weighted by their frequency. Unweighted, they're 60.6%, 59.7% and 59.9%, as positions 8 and above appear in few words.

### Pronunciation Validity

| Model | Recall ↑ | Acronym Rejection ↑ |
| ----- | -------- | ------------------- |
| NGram | 98.9%    | 90.5%               |

> [!NOTE]
> The n-gram model is evaluated on held-out CMUdict words for recall, and on the 189 CMUdict acronyms whose pronunciation spells them out letter by letter for rejection.

## Develop

> [!NOTE]
> This section is for development the e2k root package. For training, see [Training](#training)

### Dependencies

We use [`uv`](https://docs.astral.sh/uv/) for dependency management:

- Root package (`e2k` in `src/`) is the PyPI library and only depends on `numpy`.
- Training code lives in the `e2k_train/` workspace member (`e2k_train`) and carries the heavy ML/tooling dependencies.

For root package:

```bash
uv sync
```

### Testing

The model files are tracked in git under `src/e2k/models`, so a fresh clone runs as is.

```bash
# test
uv run src/inference.py
```

## Training

See [here](./e2k_train/README.md) for instructions related to data preparation, training, evaluation and exporting.

## License

- **Code**: [Unlicense](https://unlicense.org/)
- **Dictionaries**: CC-BY-SA (Wiktionary, JMdict/EDICT)
- **Accent Predictor**: Unidic (GPLv2.0/LGPLv2.1/BSD)
- **N-Gram Model**: CMUDict (BSD 2-Clause)

> Machine learning weights may be considered derivative works, check the upstream licenses when distributing.

## Credits

- [Wiktionary](https://www.wiktionary.org/)
- [JMdict / EDICT](http://www.edrdg.org/jmdict/edict.html)
- [Unidic](https://clrd.ninjal.ac.jp/unidic/)
- [CMUDict](https://github.com/cmusphinx/cmudict)
