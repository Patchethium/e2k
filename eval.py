# Description: Evaluate the model on the full dataset.
# and calculate the accuracy.
import torch
import argparse
from torchtext.data.metrics import bleu_score
from tqdm.auto import tqdm
from train import Model, MyDataset

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="data.jsonl")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--p2k", action="store_true")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(p2k=args.p2k).to(device)

model.load_state_dict(torch.load(args.model))

model.eval()

dataset = MyDataset(args.data, device, p2k=args.p2k)
dataset.set_return_full(True) # bleu score test

candidates = []
references = []

def tensor2str(t):
    return [str(int(x)) for x in t]

for i in tqdm(range(len(dataset))):
    eng, kata = dataset[i]
    res = model.inference(eng)
    candidates.append(tensor2str(res))
    references.append([tensor2str(k) for k in kata])

print(f"BLEU score: {bleu_score(candidates, references)}")
