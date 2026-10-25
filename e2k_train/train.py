# we train a s2s model to predict the katakana phonemes from
# English phonemes
import json
import time
import random
import argparse
from functools import partial
from os import path
from random import randint

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR

from g2p_en import G2p

from e2k_train.hp import kanas, en_phones, ascii_entries, PAD_IDX, SOS_IDX, EOS_IDX


SEED = 3407
DIM = 256


class Model(nn.Module):
    def __init__(self, p2k: bool = False, dim: int = DIM):
        super(Model, self).__init__()
        if p2k:
            self.e_emb = nn.Embedding(len(en_phones), dim)
        else:
            self.e_emb = nn.Embedding(len(ascii_entries), dim)
        self.k_emb = nn.Embedding(len(kanas), dim)
        self.encoder = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
        )
        self.pre_decoder = nn.GRU(dim, dim, batch_first=True)
        self.post_decoder = nn.GRU(2 * dim, dim, batch_first=True)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(dim, len(kanas))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [B, Ts]
        tgt: [B, Tt]
        src_mask: [B, Ts]
        tgt_mask: [B, Tt]
        """
        e_emb = self.e_emb(src)
        k_emb = self.k_emb(tgt)
        k_emb = k_emb[:, :-1]
        enc_out, _ = self.encoder(e_emb)
        enc_out = self.encoder_fc(enc_out)
        dec_out, _ = self.pre_decoder(k_emb)
        attn_out, _ = self.attn.forward(
            dec_out, enc_out, enc_out, key_padding_mask=~src_mask
        )
        x = torch.cat([dec_out, attn_out], dim=-1)
        x, _ = self.post_decoder(x)
        x = self.fc(x)
        return x

    def inference(self, src):
        # Assume both src and tgt are unbatched
        sos_idx = SOS_IDX
        eos_idx = EOS_IDX
        src = src.unsqueeze(0)
        src_emb = self.e_emb(src)
        enc_out, _ = self.encoder(src_emb)
        enc_out = self.encoder_fc(enc_out)
        res = [sos_idx]
        h1 = None
        h2 = None
        count = 0
        while res[-1] != eos_idx and count < 16:
            dec = torch.tensor([res[-1]]).unsqueeze(0).to(src.device)
            dec_emb = self.k_emb(dec)
            dec_out, h1 = self.pre_decoder(dec_emb, h1)
            attn_out, _ = self.attn(dec_out, enc_out, enc_out)
            x = torch.cat([dec_out, attn_out], dim=-1)
            x, h2 = self.post_decoder(x, h2)
            x = self.fc(x)
            idx = torch.argmax(x, dim=-1)
            res.append(idx.cpu().item())
            count += 1
        return res


class MyDataset(Dataset):
    def __init__(self, path, device, p2k: bool = True):
        """
        reads a json line file
        """
        super().__init__()
        # G2p loads nltk data, only P2K needs it
        self.g2p = G2p() if p2k else None
        with open(path, "r") as file:
            lines = file.readlines()
        self.data = [json.loads(line) for line in lines]
        self.device = device
        self.eng_dict = {c: i for i, c in enumerate(en_phones)}
        self.c_dict = {c: i for i, c in enumerate(ascii_entries)}
        self.kata_dict = {c: i for i, c in enumerate(kanas)}
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.cache_en = {}
        self.cache_kata = {}
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
        if idx in self.cache_en:
            return self.cache_en[idx], self.cache_kata[idx]
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
            en = torch.tensor(eng).to(self.device)
            kana = torch.tensor(kata).to(self.device)
            self.cache_en[idx] = en
            self.cache_kata[idx] = kana
            return en, kana
        else:
            kata = []
            for k in katas:
                k = [self.kata_dict[c] for c in k]
                k = [self.sos_idx] + k + [self.eos_idx]
                kata.append(torch.tensor(k).to(self.device))
            en = torch.tensor(eng).to(self.device)
            self.cache_en[idx] = en
            self.cache_kata[idx] = kata
            return en, kata


def lens2mask(lens, max_len):
    mask = torch.zeros(len(lens), max_len).bool()
    for i, le in enumerate(lens):
        mask[i, :le] = True
    return mask


def collate_fn(batch, device):
    engs = [x[0] for x in batch]
    katas = [x[1] for x in batch]
    eng_lens = [len(x) for x in engs]
    kata_lens = [len(x) for x in katas]
    eng_mask = lens2mask(eng_lens, max(eng_lens))
    kata_mask = lens2mask(kata_lens, max(kata_lens))
    engs = pad_sequence(engs, batch_first=True, padding_value=0)
    katas = pad_sequence(katas, batch_first=True, padding_value=0)
    engs, katas, eng_mask, kata_mask = [
        x.to(device) for x in [engs, katas, eng_mask, kata_mask]
    ]
    return engs, katas, eng_mask, kata_mask


def infer(src, model, p2k):
    model = model.eval()
    res = model.inference(src)
    # return to words
    res = [kanas[i] for i in res]
    # also for english phonemes
    if p2k:
        src = [en_phones[i] for i in src]
    else:
        src = [ascii_entries[i] for i in src]
    return src, res


class Words(Dataset):
    """
    Encodes every entry once. Unlike `MyDataset`, a training item samples a
    fresh spelling on every access; a `fixed` one keeps a seeded spelling, so
    validation losses are comparable across epochs.
    """

    def __init__(self, base: MyDataset, indices, fixed: bool, seed: int = SEED):
        self.base = base
        self.indices = list(indices)
        self.fixed = fixed
        self.rng = random.Random(seed)
        self.choice = [self.rng.randrange(len(base.katas[i])) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        katas = self.base.katas[idx]
        j = self.choice[i] if self.fixed else self.rng.randrange(len(katas))
        return self.base.srcs[idx], katas[j]


def split(n: int, seed: int = SEED):
    """
    fixed 90/5/5 train/val/test indices, independent of the global torch seed,
    so the test words stay unseen by every model trained on `train`
    """
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = n_test = n // 20
    test = perm[:n_test]
    val = perm[n_test : n_test + n_val]
    train = perm[n_test + n_val :]
    return train, val, test


def load_data(data_path, p2k):
    base = MyDataset(data_path, torch.device("cpu"), p2k=p2k)
    base.srcs, base.katas = [], []
    for item in base.data:
        src = base.p2k(item["word"]) if p2k else base.c2k(item["word"])
        base.srcs.append(torch.tensor([SOS_IDX] + src + [EOS_IDX]))
        base.katas.append(
            [
                torch.tensor([SOS_IDX] + [base.kata_dict[c] for c in k] + [EOS_IDX])
                for k in item["kata"]
            ]
        )
    train_idx, val_idx, test_idx = split(len(base))
    return base, train_idx, val_idx, test_idx


def loader(ds, shuffle, batch_size=64):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED) if shuffle else None,
        collate_fn=partial(collate_fn, device=torch.device("cpu")),
    )


@torch.no_grad()
def val_ce(model, val_dl):
    """token weighted cross entropy"""
    model.eval()
    ce_sum, count = 0.0, 0
    for src, tgt, src_mask, tgt_mask in val_dl:
        out = model(src, tgt, src_mask, tgt_mask)
        n = (tgt[:, 1:] != 0).sum().item()
        ce = F.cross_entropy(out.transpose(1, 2), tgt[:, 1:], ignore_index=0)
        ce_sum += ce.item() * n
        count += n
    return ce_sum / count


def train_ce(model, train_dl, val_dl, epochs=10):
    """trains a model from scratch on the data, also the teacher of `shrink.py`"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, 0.8)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for src, tgt, src_mask, tgt_mask in train_dl:
            optimizer.zero_grad()
            out = model(src, tgt, src_mask, tgt_mask)
            loss = F.cross_entropy(out.transpose(1, 2), tgt[:, 1:], ignore_index=0)
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss = val_ce(model, val_dl)
        history.append(loss)
        print(
            f"  epoch {epoch}: val CE {loss:.4f} ({time.time() - t0:.0f}s)", flush=True
        )
    return history


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="vendor/katakana_dict.jsonl")
    parser.add_argument("--p2k", action="store_true")
    parser.add_argument(
        "--output", type=str, help="defaults to vendor/model-{c2k|p2k}.pth"
    )
    args = parser.parse_args()
    task = "p2k" if args.p2k else "c2k"

    # the same data, split and seeding as the teacher of `shrink.py`
    base, train_idx, val_idx, test_idx = load_data(args.data, args.p2k)
    print(f"split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    train_dl = loader(Words(base, train_idx, fixed=False), shuffle=True)
    val_dl = loader(Words(base, val_idx, fixed=True), shuffle=False)
    torch.manual_seed(SEED)
    model = Model(p2k=args.p2k)
    train_ce(model, train_dl, val_dl)
    output = args.output or path.join("vendor", f"model-{task}.pth")
    torch.save(model.state_dict(), output)
    print(f"saved {output}")


if __name__ == "__main__":
    train()
