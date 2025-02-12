# we train a s2s model to predict the katakana phonemes from
# English phonemes
from g2p_en import G2p
import json
import torch
from torch import nn
import argparse
from random import randint
from torch.utils.data import random_split, Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

# scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from string import ascii_letters

katas = [
    "ァ",
    "ア",
    "ィ",
    "イ",
    "ゥ",
    "ウ",
    "ェ",
    "エ",
    "ォ",
    "オ",
    "カ",
    "ガ",
    "キ",
    "ギ",
    "ク",
    "グ",
    "ケ",
    "ゲ",
    "コ",
    "ゴ",
    "サ",
    "ザ",
    "シ",
    "ジ",
    "ス",
    "ズ",
    "セ",
    "ゼ",
    "ソ",
    "ゾ",
    "タ",
    "ダ",
    "チ",
    "ヂ",
    "ッ",
    "ツ",
    "テ",
    "デ",
    "ト",
    "ド",
    "ナ",
    "ニ",
    "ヌ",
    "ネ",
    "ノ",
    "ハ",
    "バ",
    "パ",
    "ヒ",
    "ビ",
    "ピ",
    "フ",
    "ブ",
    "プ",
    "ヘ",
    "ベ",
    "ペ",
    "ホ",
    "ボ",
    "ポ",
    "マ",
    "ミ",
    "ム",
    "メ",
    "モ",
    "ャ",
    "ヤ",
    "ュ",
    "ユ",
    "ョ",
    "ヨ",
    "ラ",
    "リ",
    "ル",
    "レ",
    "ロ",
    "ワ",
    "ヰ",
    "ヱ",
    "ヲ",
    "ン",
    "ヴ",
    "ー",
]

engs = [
    "AA0",
    "AA1",
    "AA2",
    "AE0",
    "AE1",
    "AE2",
    "AH0",
    "AH1",
    "AH2",
    "AO0",
    "AO1",
    "AO2",
    "AW0",
    "AW1",
    "AW2",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH0",
    "EH1",
    "EH2",
    "ER0",
    "ER1",
    "ER2",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH0",
    "IH1",
    "IH2",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW0",
    "OW1",
    "OW2",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH0",
    "UH1",
    "UH2",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

ascii_entries = ["<pad>", "<sos>", "<eos>"] + list(ascii_letters)

katas = ["<pad>", "<sos>", "<eos>"] + katas
engs = ["<pad>", "<sos>", "<eos>"] + engs


class Model(nn.Module):
    def __init__(self, p2k: bool = False):
        super(Model, self).__init__()
        if p2k:
            self.e_emb = nn.Embedding(len(engs), 256)
        else:
            self.e_emb = nn.Embedding(len(ascii_entries), 256)
        self.k_emb = nn.Embedding(len(katas), 256)
        self.encoder = nn.GRU(256, 256, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Sequential(
            nn.Linear(2 * 256, 256),
            nn.Tanh(),
        )
        self.pre_decoder = nn.GRU(256, 256, batch_first=True)
        self.post_decoder = nn.GRU(2 * 256, 256, batch_first=True)
        self.attn = nn.MultiheadAttention(256, 4, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(256, len(katas))

    def forward(self, src, tgt):
        e_emb = self.e_emb(src)
        k_emb = self.k_emb(tgt)
        k_emb = F.pad(k_emb, (0, 0, 1, 0))
        enc_out, _ = self.encoder(e_emb)
        enc_out = self.encoder_fc(enc_out)
        dec_out, _ = self.pre_decoder(k_emb)
        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        x = torch.cat([dec_out, attn_out], dim=-1)
        x, _ = self.post_decoder(x)
        x = self.fc(x)
        # strip the padded tokens
        x = x[:, :-1]
        return x

    def inference(self, src):
        # Assume both src and tgt are unbatched
        sos_idx = 1
        eos_idx = 2
        src = src.unsqueeze(0)
        src_emb = self.e_emb(src)
        enc_out, _ = self.encoder(src_emb)
        enc_out = self.encoder_fc(enc_out)
        res = [sos_idx]
        h1 = None
        h2 = None
        count = 0
        while res[-1] != eos_idx and count < 16:
            dec = torch.tensor([res[-1]]).unsqueeze(0)
            dec_emb = self.k_emb(dec)
            dec_out, h1 = self.pre_decoder(dec_emb, h1)
            attn_out, _ = self.attn(dec_out, enc_out, enc_out)
            x = torch.cat([dec_out, attn_out], dim=-1)
            x, h2 = self.post_decoder(x, h2)
            x = self.fc(x)
            idx = torch.argmax(x, dim=-1)
            res.append(idx.item())
            count += 1
        return res


class MyDataset(Dataset):
    def __init__(self, path, device, p2k: bool = True):
        """
        reads a json line file
        """
        super().__init__()
        self.g2p = G2p()
        with open(path, "r") as file:
            lines = file.readlines()
        self.data = [json.loads(line) for line in lines]
        self.device = device
        self.eng_dict = {c: i for i, c in enumerate(engs)}
        self.c_dict = {c: i for i, c in enumerate(ascii_entries)}
        self.kata_dict = {c: i for i, c in enumerate(katas)}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.p2k_flag = p2k
        self.return_full = False

    def __len__(self):
        return len(self.data)

    def p2k(self, eng):
        phonemes = self.g2p(eng)
        # phonemes = [p[:-1] if p[-1] in "012" else p for p in phonemes]
        phonemes = list(filter(lambda x: x in self.eng_dict, phonemes))
        eng = [self.eng_dict[c] for c in phonemes]
        return eng

    def c2k(self, eng):
        eng = [self.c_dict[c] for c in eng]
        return eng

    def set_return_full(self, flag: bool):
        """
        Returns the full dataset, it's for bleu score calculation
        """
        self.return_full = flag

    def __getitem__(self, idx):
        item = self.data[idx]
        eng = item["word"]
        katas = item["kata"]
        if self.p2k_flag:
            eng = self.p2k(eng)
        else:
            eng = self.c2k(eng)
        eng = [self.sos_idx] + eng + [self.eos_idx]
        # katas is a list of katakana words
        # if not return_full, we randomly select one of them
        # else we return all of them
        if not self.return_full:
            kata = katas[randint(0, len(katas) - 1)]
            kata = [self.kata_dict[c] for c in kata]
            kata = [self.sos_idx] + kata + [self.eos_idx]
            return torch.tensor(eng).to(self.device), torch.tensor(kata).to(self.device)
        else:
            kata = []
            for k in katas:
                k = [self.kata_dict[c] for c in k]
                k = [self.sos_idx] + k + [self.eos_idx]
                kata.append(torch.tensor(k).to(self.device))
            return torch.tensor(eng).to(self.device), kata


def lens2mask(lens, max_len):
    mask = torch.zeros(len(lens), max_len).bool()
    for i, l in enumerate(lens):
        mask[i, :l] = True
    return mask


def collate_fn(batch):
    engs = [x[0] for x in batch]
    katas = [x[1] for x in batch]
    eng_lens = [len(x) for x in engs]
    kata_lens = [len(x) for x in katas]
    eng_mask = lens2mask(eng_lens, max(eng_lens))
    kata_mask = lens2mask(kata_lens, max(kata_lens))
    engs = pad_sequence(engs, batch_first=True, padding_value=0)
    katas = pad_sequence(katas, batch_first=True, padding_value=0)
    return engs, katas, eng_mask, kata_mask


def infer(src, model, p2k):
    model = model.eval()
    res = model.inference(src)
    # return to words
    res = [katas[i] for i in res]
    # also for english phonemes
    if p2k:
        src = [engs[i] for i in src]
    else:
        src = [ascii_entries[i] for i in src]
    return src, res


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data.jsonl")
    parser.add_argument("--p2k", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(p2k=args.p2k).to(device)
    dataset = MyDataset(args.data, device, p2k=args.p2k)
    train_ds, val_ds = random_split(dataset, [0.9, 0.1])

    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, 0.85)
    writer = SummaryWriter()
    epochs = 15
    steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for eng, kata, e_mask, k_mask in train_dl:
            optimizer.zero_grad()
            out = model(eng, kata)
            loss = criterion(out.transpose(1, 2), kata)
            writer.add_scalar("Loss/train", loss.item(), steps)
            loss.backward()
            optimizer.step()
            steps += 1
        model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for eng, kata, e_mask, k_mask in val_dl:
                out = model(eng, kata)
                loss = criterion(out.transpose(1, 2), kata)
                total_loss += loss.item()
                count += 1
        # take a sample and inference it
        sample = val_ds[randint(0, len(val_ds) - 1)]
        src, tgt = sample
        src, pred = infer(src, model, args.p2k)
        print(f"Epoch {epoch} Sample: {src} -> {pred}")
        writer.add_scalar("Loss/val", total_loss / count, epoch)
        print(f"Epoch {epoch} Loss: {total_loss / count}")
        scheduler.step()
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
