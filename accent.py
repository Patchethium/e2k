import csv
import torch

from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics import MulticlassAccuracy

from hp import kanas


DIM = 256
SEED = 3407


class AccentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(kanas[3:]), DIM) # we don't need the special tokens
        self.rnn = nn.GRU(DIM, DIM, 1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2*DIM, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        _, h = self.rnn(x)
        h = torch.cat([h[0], h[1]], dim=-1)
        acc = self.head(h)
        acc = acc.squeeze(-1)
        return acc


class AccentDataset(Dataset):
    def __init__(self):
        path = "vendor/lex.csv"
        with open(path) as f:
            lines = f.readlines()
        reader = csv.reader(lines, delimiter=",")
        self.data = []
        for row in reader:
            phoneme = row[24]
            accent_core = row[27]
            if any([phoneme == "*", accent_core == "*"]):
                continue
            if len(accent_core.split(",")) > 1:
                accent_core = accent_core.split(",")[0]
            accent_core = float(accent_core)
            accent_length = float(len(phoneme))
            self.data.append((phoneme, accent_core, accent_length))
        self.input_map = {c: i for i, c in enumerate(kanas[3:])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        phoneme, accent_core, accent_length = self.data[index]
        ph_idx = [self.input_map[c] for c in phoneme]
        ph_idx = torch.tensor(ph_idx)
        accent_core = torch.tensor(accent_core)
        accent_length = torch.tensor(accent_length)
        return ph_idx, accent_core, accent_length


def collate_fn(batch):
    ph_idx, accent_core, accent_length = zip(*batch)
    ph_idx = pad_sequence(ph_idx, padding_value=0, batch_first=True)
    accent_core = torch.stack(accent_core)
    accent_length = torch.stack(accent_length)
    accent = accent_core / accent_length
    return ph_idx, accent, accent_core, accent_length


def train():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccentPredictor().to(device)
    dataset = AccentDataset()
    train_ds, val_ds = random_split(dataset, [0.9, 0.1])
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterian = nn.MSELoss()
    epochs = 20
    scheduler = ExponentialLR(optim, 0.9)
    writer = SummaryWriter()
    step = 0
    for e in range(1, epochs + 1):
        for ph, acc, _, _ in train_dl:
            step += 1
            ph = ph.to(device)
            acc = acc.to(device)
            optim.zero_grad()
            pred_acc = model.forward(ph)
            loss = criterian.forward(pred_acc, acc)
            loss.backward()
            optim.step()
            writer.add_scalar("train/loss", loss.item(), step)
        else:
            with torch.no_grad():
                accuracy = MulticlassAccuracy().to(device)
                for ph, _, core, length in val_dl:
                    ph = ph.to(device)
                    core = core.to(device)
                    length = length.to(device)
                    pred_acc = model(ph)
                    pred_core = (pred_acc * length).round().long()
                    core = core.long()
                    accuracy.update(pred_core, core)
                print(f"Epoch {e}, Accuracy: {accuracy.compute()}")
            scheduler.step()
    torch.save(model.state_dict(), "vendor/accent.pth")


if __name__ == "__main__":
    train()
