# Training E2K Models

> [!NOTE]
> Instructions covered in this README all work under `e2k_train/`, make sure your working directory is `e2k_train`.  
> If you're only for the root `e2k` library, see [here](../README.md).

## Setup

We use [`uv`](https://docs.astral.sh/uv/) for dependency management:

- Root package (`e2k` in `src/`) is the PyPI library and only depends on `numpy`.
- Training code lives in the `e2k_train/` workspace member (`e2k_train`) and carries the heavy ML/tooling dependencies.

### Install dependencies

```bash
# Training environment (CPU)
uv sync --extra cpu

# Training environment (CUDA 12.4)
uv sync --extra cu124
```

## Data Preparation

### 1. Katakana Dictionary

Extracted from **Wiktionary** and **JMdict/EDICT**, resulting ~30k entries.

#### Download Sources

- [Wiktionary dump (ja-extract.jsonl.gz)](https://kaikki.org/dictionary/rawdata.html)
- [JMdict / EDICT (edict2.gz)](https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project)

Unzip the archive files, put `ja-extract.josnl` and `edict2` into `e2k_train/vendor`.

Extract the dictionary:

```bash
uv run extract.py
```

The dictionary is saved in `vendor/katakana_dict.jsonl`.

### 2. Accent Dictionary

Download [Unidic 2.1.2](https://clrd.ninjal.ac.jp/unidic/back_number.html), extract it and place `lex.csv` in `e2k_train/vendor/`.

Train the accent predictor:

```bash
uv run accent.py
```

### 3. N-Gram Model

Download [cmudict.dict](https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict) and place it in `e2k_train/vendor/`.

Train the n-gram model:

```bash
uv run ngram.py --train
```
### 4. E2K Model

```bash
uv run train.py --data ./vendor/katakana_dict.jsonl
```

The checkpoint will be stored as `vendor/model-{p2k|c2k}-e{epoch}.pth`. It's recommended to use [`CUDA`](#Install-dependencies) for faster training.

### Evaluation

Run with `--help` for more details.

```bash
uv run eval.py --data ./vendor/katakana_dict.jsonl --model model.pth --p2k
```

### Export to NumPy

Run with `--help` for more details.

```bash
uv run export.py --model model.pth --p2k --output model-p2k.npz
```