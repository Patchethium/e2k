# English to Katakana Dictionary

It's an extraction of the Wikitionary's Japanese dictionary and `JMdict / EDICT` to create a dictionary of English words and their Katakana representation.

We also provide a GRU model to convert English to Katakana without a dictionary.

```json
{"word": "word", "katakana": "ワード"}
{"word": "another", "katakana": "アナザー"}
{"word": "example", "katakana": "エグザンプル"}
```

It's for the further use in Japanese Text-to-Speech applications or Japanese learning.

This README is divided into 2 parts, first about the dictionary extraction and the second part is about the model.

## Dictionary

### Dependencies

The extraction script has zero dependencies, as long as you have a Python 3 interpreter it should work.

### Download the data

#### Wikitionary

Download the raw dump of the Japanese Wikitionary from https://kaikki.org/dictionary/rawdata.html, they kindly provide the parsed data in a JSONL format.

Look for the `Japanese ja-extract.jsonl.gz (compressed 37.5MB)` entry and download it. If you prefer command line, use

```bash
curl -O https://kaikki.org/dictionary/downloads/ja/ja-extract.jsonl.gz
```

#### JMdict / EDICT

Download the `JMdict` and `EDICT` from https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project.

Look for the `edict2.gz` and download it. Or in command line:

```bash
curl -O http://ftp.edrdg.org/pub/Nihongo/edict2.gz
```

Extract both files into `/vendor` folder.

On Linux, you can use

```bash
gzip -d ja-extract.jsonl.gz
gzip -d edict2.gz
```

### Run the extraction

```bash
python extract.py
# if you have another name for the file
python extract.py --path /path/to/your_file.jsonl
```

By default, a `katakana_dict.jsonl` file will be created in the `vendor` folder.

## E2K model

We also provide a seq2seq GRU model to convert English automatically to Katakana. It's trained on the aforementioned dictionary.

We provide 2 types of models, one converts phoneme to Katakana and one that converts character to Katakana. Choose the one that fits your use case.

### Usage

`e2k` is available on PyPI.

```bash
pip install e2k
```

Usage:

```python
from e2k import P2K, C2K
from g2p_en import G2p # or `phonemizer`, as long as it outputs CMUdict phoneme

# cmudict phoneme to katakana
p2k = P2K()

g2p = G2p()

katakana = p2k(g2p("word"))

print(katakana) # "ワード"

# character directly to katakana
c2k = C2K()

katakana = c2k("word")

print(katakana) # "ワード"

# decode strategy
# greedy by default, top_k and top_p are available
katakana = c2k("word", strategy="top_k", top_k=5)
# or
katakana = c2k("word", strategy="top_p", top_p=0.6, t=0.8)
# top_k and top_p are sampling strategies, which means
# each time you run the model, you may get different results
# for further information, see
# https://huggingface.co/docs/transformers/en/generation_strategies
# TODO: add beam searh
print(katakana) # "ワード"
```

We rewrite the inference of GRU model in `numpy`, minimizing the dependencies to `numpy` only.

> [!Note]
> The sections below require more than `numpy`, but they're not required for the end user.

## Development

### Install the dependencies

I use [`uv`](https://docs.astral.sh/uv/) to manage the dependencies and publish the package.

```bash
uv sync
```

Then activate the virtual environment with `source .venv/bin/activate` or add `uv run` before the commands.

### Benchmark

```bash
# --p2k for phoneme to katakana, if not provided, it will be character to katakana
python eval.py --data ./vendor/katakana_dict.jsonl --model /path/to/your/model.pth --p2k
```

| Model                 | BLEU Score ↑ |
| --------------------- | ------------ |
| Phoneme to Katakana   | 0.85         |
| Character to Katakana | 0.90         |

### Train

After installing the dependencies, `torch` will be added as a development dependency. You can train the model using

```bash
python train.py --data ./vendor/katakana_dict.jsonl
```

It takes around 10 minutes on a desktop CPU. The model will be saved as `vendor/model-{p2k/c2k}-e-{epoch}.pth`.

Also, you'll need to either download the `katakana_dict.jsonl` from the releases or create it yourself using the `extract.py` script.

### Export

The model should be exported to `numpy` format for production use.

```bash
# --p2k for phoneme to katakana, if not provided, it will be character to katakana
# --fp16 for half precision, there's no reason not to use it
# --output to specify the output file, in this project it's `model-{p2k/c2k}.npz`
python export.py --model /path/to/your/model.pth --p2k --fp16 --output /path/to/your/model.npz
```

> [!Note]
> The model is not included in the Git registry, I uploaded exported models to the releases, but didn't upload the `pth` format.

> [!Note]
> The benchmarking/training script is not included in the PyPI package, you'll need to clone this repository to train the model.

## License

The code is released under WTFPL.

The dictionary should follow the [Wikimedia's license](https://dumps.wikimedia.org/legal.html) and the [JMdict / EDICT's Copyright](https://www.edrdg.org/) license.

In short, as long as you provide the attribution, you can use it for both commercial and non-commercial purposes.

The model weights are released under the same license as the dictionary.

## Credits

- [Wikitionary](https://www.wiktionary.org/)
- [JMdict / EDICT](http://www.edrdg.org/jmdict/edict.html)
