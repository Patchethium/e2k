"""
Exports the torch weights to numpy npy files.
"""

import numpy as np
import torch
from safetensors.numpy import save_file
import argparse
from train import Model
from accent import AccentPredictor
from hp import kanas, en_phones, ascii_entries, SOS_IDX, EOS_IDX, PAD_IDX

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--p2k", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--safetensors", action="store_true", help="Use safe tensors instead of numpy"
    )
    parser.add_argument("--accent", action="store_true")

    args = parser.parse_args()

    in_table = en_phones if args.p2k else ascii_entries
    out_table = kanas

    model = Model(p2k=args.p2k) if not args.accent else AccentPredictor()
    model.load_state_dict(torch.load(args.model))
    model.eval()

    if not args.fp32:
        model = model.half()

    weights = {}

    metadata = {}

    if not args.accent:
        metadata["in_table"] = "\0".join(in_table)
        metadata["out_table"] = "\0".join(out_table)
        metadata["sos_idx"] = str(SOS_IDX)  # safetensors only accepts str in metadata
        metadata["eos_idx"] = str(EOS_IDX)
    else:
        metadata["in_table"] = "\0".join(kanas[3:])

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
            weights[name] = param.data.cpu().numpy()

    if args.safetensors:
        output = (
            args.output
            if args.output.endswith(".safetensors")
            else f"{args.output}.safetensors"
        )
        save_file(weights, output, metadata=metadata)
    else:
        weights["metadata"] = metadata
        output = args.output if args.output.endswith(".npz") else f"{args.output}.npz"
        np.savez(output, **weights)
