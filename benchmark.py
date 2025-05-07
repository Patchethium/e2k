import time

from e2k import P2K, C2K
from train import MyDataset
from extract import Welford
from tqdm.auto import tqdm


def main():
    p2k = P2K()
    c2k = C2K()
    pds = MyDataset("vendor/katakana_dict.jsonl", p2k=True, device="cpu")
    cds = MyDataset("vendor/katakana_dict.jsonl", p2k=False, device="cpu")
    # data preparation
    words = []
    phonemes = []
    for i in range(1000):
        word, _ = cds[i]
        words.append(word)
        phoneme, _ = pds[i]
        phonemes.append(phoneme)
    # benchmark
    c2k_t = Welford()
    p2k_t = Welford()
    for i in tqdm(range(200)):
        start = time.time()
        p2k(phonemes[i])
        end = time.time()
        p2k_t.update(end - start)
        start = time.time()
        c2k(words[i])
        end = time.time()
        c2k_t.update(end - start)
    print(f"P2K: mean {p2k_t.mean() * 1000} ms, std {p2k_t.std() * 1000} ms")
    print(f"C2K: mean {c2k_t.mean() * 1000} ms, std {c2k_t.std() * 1000} ms")


if __name__ == "__main__":
    main()
