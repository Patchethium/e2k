# Description: Evaluate the model on the full dataset.
# and calculate the accuracy.
import torch
import argparse
from torcheval.metrics import BLEUScore
from tqdm.auto import tqdm
from train import Model, MyDataset, random_split, SEED
from e2k import kanas

acc_idx = kanas.index("<acc>")

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="data.jsonl")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--p2k", action="store_true")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(p2k=args.p2k).to(device)

model.load_state_dict(torch.load(args.model))

model.eval()

torch.manual_seed(SEED)

dataset = MyDataset(args.data, device, p2k=args.p2k)
train_ds, val_ds = random_split(dataset, [0.95, 0.05])
dataset.set_return_full(True)  # bleu score test

bleu = BLEUScore(n_gram=3)

class Accuracy:
    def __init__(self):
        self.acc = 0
        self.count = 0
    
    def update(self, pred, true):
        self.acc += 1 if pred == true else 0
        self.count += 1
    
    def compute(self):
        return (self.acc / self.count) if self.count > 0 else 0

def tensor2str(t):
    idx = [int(x) for x in t]
    idx = list(filter(lambda x: x != acc_idx, idx))
    return " ".join([str(i) for i in idx])

accuracy = Accuracy()

for i in tqdm(range(len(val_ds))):
    eng, kata = val_ds[i]
    res = model.inference(eng)
    pred_kana = tensor2str(res)
    try:
        acc = kata[0].tolist().index(acc_idx)
        pred_acc = res.index(acc_idx)
        accuracy.update(acc, pred_acc)
    except Exception:
        accuracy.count += 1
    kana = [[tensor2str(k) for k in kata]]
    bleu.update(pred_kana, kana)


print(f"BLEU score: {bleu.compute()}")
print(f"Accent accuracy: {accuracy.compute() * 100}%")