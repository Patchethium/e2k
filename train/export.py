"""
Exports the torch weights to numpy npy files.
"""

import numpy as np
import torch
from safetensors.numpy import save_file
import argparse
from train import Model

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, required=True)
parser.add_argument("--p2k", action="store_true")
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--fp32", action="store_true")
parser.add_argument(
    "--safetensors", action="store_true", help="Use safe tensors instead of numpy"
)

args = parser.parse_args()

model = Model(p2k=args.p2k)
model.load_state_dict(torch.load(args.model))
model.eval()

if not args.fp32:
    model = model.half()

weights = {}

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
    save_file(weights, output)
else:
    output = args.output if args.output.endswith(".npz") else f"{args.output}.npz"
    np.savez(output, **weights)
