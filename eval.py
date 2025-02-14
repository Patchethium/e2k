# Description: Evaluate the model on the full dataset.
# and calculate the accuracy.
import torch
import argparse
from torcheval.metrics import BLEUScore
from tqdm.auto import tqdm
from train import Model, MyDataset, random_split

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="data.jsonl")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--p2k", action="store_true")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(p2k=args.p2k).to(device)

model.load_state_dict(torch.load(args.model))

model.eval()

torch.manual_seed(3407)

dataset = MyDataset(args.data, device, p2k=args.p2k)
test_ds, _ = random_split(dataset, [0.1, 0.9])
dataset.set_return_full(True)  # bleu score test

bleu = BLEUScore(n_gram=3)


def tensor2str(t):
    return " ".join([str(int(x)) for x in t])


for i in tqdm(range(len(test_ds))):
    eng, kata = test_ds[i]
    res = model.inference(eng)
    pred_kana = tensor2str(res)
    kana = [[tensor2str(k) for k in kata]]
    bleu.update(pred_kana, kana)


print(f"BLEU score: {bleu.compute()}")
