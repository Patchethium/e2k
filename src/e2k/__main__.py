import argparse
import time

from .inference import P2K, C2K, AccentPredictor, NGramCollection

if __name__ == "__main__":
    from g2p_en import G2p

    g2p = G2p()
    parser = argparse.ArgumentParser()
    parser.add_argument("word", nargs="?", default="vox")
    parser.add_argument("--model", choices=["base", "turbo"], default="base")
    args = parser.parse_args()
    ngram = NGramCollection()
    p2k = P2K(model=args.model)
    c2k = C2K(model=args.model)
    word = args.word
    phonemes = g2p(word)
    print(word)
    print(f"Ngram result: {'Valid' if ngram(word) else 'Invalid'}")
    print(f"As is: {ngram.as_is(word)}")
    start = time.perf_counter()
    katakana = p2k(phonemes)
    print(f"P2K:  {katakana} ({(time.perf_counter() - start) * 1000:.2f} ms)")
    start = time.perf_counter()
    katakana = c2k(word)
    print(f"C2K:  {katakana} ({(time.perf_counter() - start) * 1000:.2f} ms)")
    print("P2K (top_k): ", p2k(phonemes, "top_k", k=2))
    print("C2K (top_k): ", c2k(word, "top_k", k=2))
    print("P2K (top_p): ", p2k(phonemes, "top_p", p=0.7, t=2))
    print("C2K (top_p): ", c2k(word, "top_p", p=0.7, t=2))

    ap = AccentPredictor()
    katakana = p2k(phonemes)
    print(f"P2K Katakana: {katakana}, Accent: {ap(katakana)}")
    katakana = c2k(word)
    print(f"C2K Katakana: {katakana}, Accent: {ap(katakana)}")
