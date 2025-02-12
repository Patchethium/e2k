# English to Katakana Dictionary

It's an extraction of the Wikitionary's Japanese dictionary and `JMdict / EDICT` to create a dictionary of English words and their Katakana representation.

We also provide a GRU model to convert English to Katakana without a dictionary.

```json
{"word": "word", "katakana": "ワード"}
{"word": "another", "katakana": "アナザー"}
{"word": "example", "katakana": "エグザンプル"}
```

It's for the purpose of converting English words to Katakana and further use in Japanese Text-to-Speech applications or Japanese learning.

## Re-creating the dictionary

The script has zero dependencies, as long as you have a Python 3 interpreter it should work.

### Download the dictionary

Download the raw dump of the Japanese Wikitionary from https://kaikki.org/dictionary/rawdata.html, they kindly provide the parsed data in a JSONL format.

Look for the `Japanese ja-extract.jsonl.gz (compressed 37.5MB)` entry and download it. If you prefer command line, use

```bash
curl -O https://kaikki.org/dictionary/downloads/ja/ja-extract.jsonl.gz
```

### Extract it

Extract it into `/vendor` folder.

On Linux, you can use

```bash
gzip -d ja-extract.jsonl.gz
```

### Run the script

```bash
python extract.py
# if you have another name for the file
python extract.py --path /path/to/your_file.jsonl
```

And a `katakana_dict.jsonl` file will be created in the root folder.

## E2K model

We also provide a seq2seq GRU model to convert English automatically to Katakana. It's trained on the aforementioned dictionary.

We provide 2 types of models, one converts phoneme to Katakana and one that converts character to Katakana. Choose the one that fits your use case.

```bash
pip install e2k
```

The weights are included in PyPI, you can simply use the model like this:

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

# since it's an autoregressive model, you can set the decoding strategy
# to greedy, top-k or nucleus sampling (top-p)
# see https://huggingface.co/docs/transformers/en/generation_strategies
c2k.set_decoding_strategy("top-p", p=0.3)

katakana = c2k("word")

print(katakana) # "ワード"
```

We rewrite the inference of GRU model in `numpy`, minimizing the dependencies to `numpy` only.

### Benchmark

```bash
# --p2k for phoneme to katakana, if not provided, it will be character to katakana
python eval.py --data ./vendor/katakana_dict.jsonl --model /path/to/your/model.pth --p2k
```

| Model                 | BLEU Score |
| --------------------- | ---------- |
| Phoneme to Katakana   | 0.85       |
| Character to Katakana | 0.90       |

### Training the model

You'll need `torch` and `g2p_en`. After that, you can run the `train.py` script to train the model.

```bash
python train.py --data ./vendor/kanji_dict.jsonl
```

It takes around 10 minutes to train the model on a desktop CPU. The model will be saved as `model.pth` in the root folder.

Also, you'll need to either download the `kanji_dict.jsonl` from the releases or create it yourself using the `extract.py` script.

Be noted that the training script is not included in the PyPI package, you'll need to clone the repository to train the model.

## Development

### Install the dependencies

I use [`uv`](https://docs.astral.sh/uv/) to manage the dependencies and publish the package.

```bash
uv sync
```

## License

The code is released under WTFPL.

The dictionary should follow the [Wikimedia's license](https://dumps.wikimedia.org/legal.html) and the [JMdict / EDICT's Copyright](https://www.edrdg.org/) license.

In short, as long as you provide the attribution, you can use it for both commercial and non-commercial purposes.

The model weights are released under the same license as the dictionary.

## Credits

- [Wikitionary](https://www.wiktionary.org/)
- [JMdict / EDICT](http://www.edrdg.org/jmdict/edict.html)
