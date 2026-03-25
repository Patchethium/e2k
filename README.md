# English to Katakana Translator

[![PyPI version](https://badge.fury.io/py/e2k.svg)](https://badge.fury.io/py/e2k)

`e2k` is a Python library that translates English to Katakana using lightweight RNN models trained on data from Wiktionary and JMdict/EDICT.
It only requires `numpy` as a runtime dependency.

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
print(p2k(g2p(word)))  # ジオダギ

c2k = C2K()
print(c2k(word))       # ジオガディ
```

> [!NOTE]
> While defaulting on `greedy`, both models support `top_k` and `top_p` decoding strategies. It varies the output between each inference, use when you know what you're doing.

```python
c2k(word, "top_p", p=0.9, t=2)
c2k(word, "top_k", k=5)
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
# Katakana: ジオガディ, Accent: 3
```

### 3. Pronunciation Validity (N-Gram Model)

You can check if a word should be transliterated or spelled as-is.

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
> - `doggy` → ドギー

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

| Task                | Model           | Metric     | Score |
| ------------------- | --------------- | ---------- | ----- |
| Katakana Prediction | P2K             | BLEU ↑     | 0.89  |
| Katakana Prediction | C2K             | BLEU ↑     | 0.92  |
| Accent Prediction   | AccentPredictor | Accuracy ↑ | 88.4% |

> [!NOTE]
> I'm too dumb to figure out how to properly evaluate the n-gram model.

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

To make it run for debug, you need to download the following files from [Releases](https://github.com/Patchethium/e2k/releases) and put them into `src/models`.

- `c2k` and `p2k` model files ([`model-c2k.npz`](https://github.com/Patchethium/e2k/releases/download/0.4.0/model-c2k.npz) and [`model-p2k.npz`](https://github.com/Patchethium/e2k/releases/download/0.4.0/model-p2k.npz))
- ngrams model [`ngram.json.zip`](https://github.com/Patchethium/e2k/releases/download/0.6.1/ngram.json.zip)
- accent model [`accent.npz`](https://github.com/Patchethium/e2k/releases/download/0.5.0/accent.npz)

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
