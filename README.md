# English to Katakana Translator

[![PyPI version](https://badge.fury.io/py/e2k.svg)](https://badge.fury.io/py/e2k)


`e2k` is a Python library that translates English to Katakana. It's based on a RNN model trained on a dictionary extracted from Wikitionary and JMdict / EDICT. It only requires `numpy` as a dependency.

We also provide a English to Katakana dictionary in the releases (not available in the PyPI package).

## Usage

`e2k` is available on PyPI.

```bash
pip install e2k
```

Usage:

2 types of models are provided, one converts phoneme to Katakana and one that converts character to Katakana. Choose the one that fits your use case.

```python
from e2k import P2K, C2K
from g2p_en import G2p # any g2p library with CMUdict will work

# cmudict phoneme to katakana
p2k = P2K()

g2p = G2p()

word = "vordhosbn" # track 2 from Aphex Twin's "Drukqs"

word = word.lower()

katakana = p2k(g2p(word))

print(katakana) # "ボードヒッチン"

# characters directly to katakana
c2k = C2K()

katakana = c2k(word)

print(katakana) # "ボードホスン"

# we provide top_k and top_p decoding strategies
katakana = c2k(word, "top_k", k=5) # top_k sampling
katakana = c2k(word, "top_p", p=0.9, t=2) # top_p sampling
# see https://huggingface.co/docs/transformers/en/generation_strategies
# for more details

# you can check the accepted symbols using
in_table = c2k.in_table # `c2k` accepts lowercase characters, space and apostrophe
in_table = p2k.in_table # `p2k` accepts phonemes from the CMUdict and space
# for output symbols
out_table = c2k.out_table
out_table = p2k.out_table
```

> [!WARNING]
> For any symbols not in the `in_table`, the model will ignore them and may produce unexpected results.

### Performance

The BLEU score is calculated on a random subset with size of 10% of the dataset.

| Model | BLEU Score ↑ |
| ----- | ------------ |
| P2K   | 0.89         |
| C2K   | 0.92         |

## Dictionary

We train the model on a dictionary extracted from `Wikitionary` and `JMdict / EDICT`. The dictionary contains 30k entries, you can also find it in the releases.

> [!Note]
> The dictionary is not included in the PyPI package. Either download it from the releases or create it yourself following the instructions below.

### Dependencies

The extraction script has zero dependencies, as long as you have a Python 3 interpreter it should work.

However, it's not included in the PyPI package, you need to clone this repository to use it.

```bash
git clone https://github.com/Patchethium/e2k.git
```

### Download data

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

## Development

### Install the dependencies

I use [`uv`](https://docs.astral.sh/uv/) to manage the dependencies and publish the package.

```bash
uv sync
```

Then activate the virtual environment with `source .venv/bin/activate` or add `uv run` before the commands.

### Benchmark

The scores in [Performance](#performance) are obtained using the `eval.py` script.

```bash
# --p2k for phoneme to katakana, if not provided, it will be character to katakana
python eval.py --data ./vendor/katakana_dict.jsonl --model /path/to/your/model.pth --p2k
```

### Train

After installing the dependencies, `torch` will be added as a development dependency. You can train the model using

```bash
python train.py --data ./vendor/katakana_dict.jsonl
```

It takes around 10 minutes on a desktop CPU. The model will be saved as `vendor/model-{p2k/c2k}-e{epoch}.pth`.

Also, you'll need to either download the `katakana_dict.jsonl` from the releases or create it yourself using the `extract.py` script.

#### CUDA

The `cpu` version is capable to train this little model, if you prefer to use GPU, use `--extra` to install the `torch` with CUDA support,

```bash
uv sync --extra cu124
# or
uv sync --extra cu121
```

depending on your CUDA version.

### Export

The model should be exported to `numpy` format for production use.

```bash
# --p2k for phoneme to katakana, if not provided, it will be character to katakana
# --fp32 for double precision, by default we use fp16 to save space
# --output to specify the output file, in this project it's `model-{p2k/c2k}.npz`
# --safetenors to use safe tensors, it's for easier binding in some languages
python export.py --model /path/to/your/model.pth --p2k --output /path/to/your/model.npz
```

> [!Note]
> The pretrained weights are not included in the Git registry, you can find them in the releases.

## License

 - The code is released under WTFPL, you can do WTF you want with it.
 - The dictionary follows the [Wikimedia's license](https://dumps.wikimedia.org/legal.html) and the [JMdict / EDICT's Copyright](https://www.edrdg.org/) license.
   - In short, they both fall into CC-BY-SA.
   - The model weights are trained using the dictionary. I am not a lawyer, whether the machine learning weights is considered as a derivative work is up to you.

## Credits

- [Wikitionary](https://www.wiktionary.org/)
- [JMdict / EDICT](http://www.edrdg.org/jmdict/edict.html)
