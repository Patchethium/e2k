"""
Converts `.npz` models from e2k <= 0.7 to `.safetensors`.

The `.npz` metadata is a pickled dict, and unpickling can run arbitrary code,
so only convert files you trust.

    uv run npz2safetensors.py model-c2k.npz [more.npz ...]
"""

import argparse

import numpy as np
from safetensors.numpy import save_file


def convert(src: str, dst: str):
    data = np.load(src, allow_pickle=True)
    metadata = data["metadata"].item()
    weights = {k: data[k] for k in data.files if k != "metadata"}
    save_file(weights, dst, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", nargs="+")
    args = parser.parse_args()
    for src in args.npz:
        dst = src.removesuffix(".npz") + ".safetensors"
        convert(src, dst)
        print(f"{src} -> {dst}")
