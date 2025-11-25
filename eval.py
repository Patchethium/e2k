# Description: Evaluate the model on the full dataset.
# and calculate the accuracy.
import os
import torch
import argparse
from torcheval.metrics import BLEUScore
from tqdm.auto import tqdm
from train import E2KDataset, MLME2KLightningModule, CausalE2KLightningModule
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", type=str, required=True)
parser.add_argument("--model", type=str, required=True)

args = parser.parse_args()
c = OmegaConf.load(args.config)

test_path = os.path.join(c.tmp_dir, "test_dataset.jsonl")
train_path = os.path.join(c.tmp_dir, "train_dataset.jsonl")

if not all([os.path.exists(path) for path in [test_path, train_path]]):
    raise FileNotFoundError(f"Test dataset not found at {test_path}")


test_ds = E2KDataset(test_path, eval=True, causal=c.model.causal)
train_ds = E2KDataset(train_path, eval=True, causal=c.model.causal)
test_ds.update_symbols(train_ds.input_symbols, train_ds.output_symbols)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if c.model.causal:
    module = CausalE2KLightningModule.load_from_checkpoint(
        args.model, in_symbols=test_ds.input_symbols, out_symbols=test_ds.output_symbols, cfg=c
    )
else:
    module = MLME2KLightningModule.load_from_checkpoint(
        args.model, in_symbols=test_ds.input_symbols, out_symbols=test_ds.output_symbols, cfg=c
    )
model = module.model.to(device)
model.eval()
bleu = BLEUScore(n_gram=2)

with torch.no_grad():
    for src, tgt in tqdm(test_ds, desc="Evaluating"):
        tgt = [" ".join(list(t)) for t in tgt]
        if c.model.causal:
            src = ["<sos>", *list(src), "<eos>"]
        else:
            src = list(src)
        src_idx = torch.tensor(
                [test_ds.input_symbol_to_idx[ch] for ch in src], dtype=torch.long
            )
        src = src_idx.unsqueeze(0).to(device)

        output, _ = model.generate(src, None)
        output = " ".join([test_ds.output_symbols[idx] for idx in output.tolist()])
        bleu.update(output, [tgt])

print(f"BLEU Score: {bleu.compute().item()*100:.2f}")
