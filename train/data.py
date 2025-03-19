import json
from typing import List
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from e2k.constants import src_tokens, tgt_tokens


SEED = 3407  # make sure the reproducibility

"""
Self note

"""


class E2KDataset(Dataset):
    def __init__(self, p_path: str, k_path: str):
        self.src_idx = {c: i for i, c in enumerate(src_tokens)}
        self.tgt_idx = {c: i for i, c in enumerate(tgt_tokens)}
        ascii_set = set(src_tokens)
        p_dict = {}
        # process the english phoneme dictionary
        with open(p_path, "r") as f:
            for line in f:
                if line.startswith(";;;"):
                    continue
                line = line.split("#")[0].strip()
                word, phoneme = line.split("  ")
                word = word.split("(")[0].strip().lower()
                phoneme = list(phoneme.split(" "))
                if not all([c in ascii_set for c in word]):
                    continue
                if word not in p_dict:
                    p_dict[word] = phoneme  # take only the first phoneme
        # process the katakana dictionary
        # katakana dictionary is a JSON lines file with "word" and "kata" keys
        # kata is in the form of List[str]
        k_dict = []
        with open(k_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                word = entry["word"]
                word = word.lower()
                katakanas = entry["kata"]
                for katakana in katakanas:
                    k_dict.append((word, katakana))
        self.p_dict = p_dict
        self.k_dict = k_dict

    def __len__(self):
        return len(self.k_dict)

    def __getitem__(self, idx: int):
        """
        If the english word has a phoneme representation in the phoneme dictionary,
        we insert it before the katakana word with a special token <cot>(Chain of Thought)
        else, it would be just katakana word with <sos> and <eos> tokens
        """
        e, k = self.k_dict[idx]
        e_idx = [self.src_idx[c] for c in e]
        if e in self.p_dict:
            p = self.p_dict[e]
            k = ["<cot>", *p, "<sos>", *k, "<eos>"]
        else:
            k = ["<sos>", *k, "<eos>"]
        k_idx = [self.tgt_idx[c] for c in k]

        e_idx = torch.tensor(e_idx, dtype=torch.long)
        k_idx = torch.tensor(k_idx, dtype=torch.long)
        return e_idx, k_idx


def len2mask(lens: List[int]) -> Tensor:
    """
    lens: [B]
    return: [B, max_len]
    """
    max_len = max(lens)
    mask = torch.arange(max_len).expand(len(lens), max_len) < torch.tensor(
        lens
    ).unsqueeze(1)
    return mask


def collate_fn(batch):
    src, tgt = zip(*batch)
    src_lens = [len(s) for s in src]
    tgt_lens = [len(t) for t in tgt]
    src = pad_sequence(src, batch_first=True, padding_value=0)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
    src_mask = len2mask(src_lens)
    tgt_mask = len2mask(tgt_lens)
    return src, tgt, src_mask, tgt_mask


def get_loaders(num_workers: int = 4):
    P_PATH = "vendor/cmudict"
    K_PATH = "vendor/katakana_dict.jsonl"
    ds = E2KDataset(P_PATH, K_PATH)
    train, val, test = random_split(ds, [0.9, 0.09, 0.01])
    train_loader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(test, batch_size=1, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
