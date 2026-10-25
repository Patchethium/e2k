"""
Exports torch checkpoints to the `.safetensors` model files both runtimes load:
fp16 tensors (fp32 with `--fp32`) plus the vocabulary as string metadata.

    uv run export.py --model model.pth --p2k --output model-p2k.safetensors
"""

import argparse
from typing import Dict

import numpy as np
import torch
from torch import nn
from safetensors.numpy import save_file

from e2k_train.train import Model
from e2k_train.accent import AccentPredictor
from e2k_train.hp import kanas, en_phones, ascii_entries, SOS_IDX, EOS_IDX


def load_model(path: str, p2k: bool, accent: bool) -> nn.Module:
    """loads a checkpoint, a S2S model takes its width from the checkpoint"""
    state = torch.load(path, map_location="cpu")
    if accent:
        model = AccentPredictor()
    else:
        model = Model(p2k=p2k, dim=state["e_emb.weight"].shape[1])
    model.load_state_dict(state)
    return model.eval()


def model_weights(model: nn.Module, fp32: bool = False) -> Dict[str, np.ndarray]:
    return {
        name: (p.detach().float() if fp32 else p.detach().half()).cpu().numpy()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def model_metadata(p2k: bool, accent: bool) -> Dict[str, str]:
    # safetensors only accepts str in metadata
    if accent:
        return {"in_table": "\0".join(kanas[3:])}
    return {
        "in_table": "\0".join(en_phones if p2k else ascii_entries),
        "out_table": "\0".join(kanas),
        "sos_idx": str(SOS_IDX),
        "eos_idx": str(EOS_IDX),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--p2k", action="store_true")
    parser.add_argument("--accent", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model, args.p2k, args.accent)
    weights = model_weights(model, args.fp32)
    for name, w in weights.items():
        print(name, w.shape)
    output = (
        args.output
        if args.output.endswith(".safetensors")
        else f"{args.output}.safetensors"
    )
    save_file(weights, output, metadata=model_metadata(args.p2k, args.accent))
