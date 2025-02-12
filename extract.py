"""
This script will extract Japanese katakana words and its English correspondences from the Japanese Wiktionary dump.
The extracted data will be stored in a JSON file, with
{
    "English word": ["katakana1", "katakana2", ...],
}
format.
"""

import json
import re
from collections import defaultdict
from typing import Dict, List
import argparse
from string import ascii_letters


katakana_re = re.compile(r"[\u30A1-\u30F4\u30FC]+")
en_re = re.compile(r"[a-zA-Z\-\s\+]+")
kata = set()


def extract_wiki(path) -> Dict[str, List[str]]:
    file = open(path, "r")
    katakana_dict = defaultdict(list)
    for line in file:
        data = json.loads(line)
        if "word" not in data:
            continue
        word = data["word"]
        if katakana_re.fullmatch(word):
            if "etymology_texts" in data:
                # the dictionary doesn't directly specify the source word,
                # it's usually included in the etymology_texts with
                # example: "etymology_texts": ["英語: freelance"]
                # we just try and match the en_re to get the English word
                etyomology = data["etymology_texts"]
                # find the English word
                match = en_re.search(etyomology[0])
                if not match:
                    continue
                en_word = match.group(0)
                # strip the start-end whitespace
                en_word = en_word.strip()
                # remove the + and - characters
                en_word = en_word.replace("+", " ")
                en_word = en_word.replace("-", " ")
                # combine multiple spaces into one
                en_word = re.sub(r"\s+", " ", en_word)
                # normalize to lowercase
                en_word = en_word.lower().strip()
                # the en_re will match whitespace, we filter out those
                # too short or too long
                if len(en_word) > 20 or len(en_word) < 2 or any([c not in ascii_letters for c in en_word]):
                    continue
                if en_word:
                    katakana_dict[en_word].append(word)
    print(
        f"Extracted {len(katakana_dict)} katakana words from the Japanese Wiktionary."
    )
    return katakana_dict


def extract_jmdict(path) -> Dict[str, List[str]]:
    file = open(path, "r", encoding="euc-jp")
    katakana_dict = defaultdict(list)
    for line in file:
        # JMDICT is a csv file with internal commas replaced by `/`.
        data = line.split("/")
        # in JMDICT, we don't look for the full match, as the katakana words are usually
        # followed by a (P) for its part of speech
        # instead
        if data and katakana_re.match(data[0]):
            katakana = data[0]
            kanas = katakana.split(";")  # alternative readings are separated by ;
            n_kanas = set()
            # remove (P) and (n) suffixes
            for kana in kanas:
                # remove those with `・`
                if "・" in kana:
                    continue
                # remove (*) and {*}
                kana = re.sub(r"\(.*?\)", "", kana)
                kana = re.sub(r"\{.*?\}", "", kana)
                kana = kana.strip()
                if not katakana_re.fullmatch(kana) or len(kana) > 20 or len(kana) < 2:
                    continue
                match = katakana_re.match(kana)
                if match:
                    n_kanas.add(match.group(0))
            en_word = data[1]
            # remove the derogatories
            if "(derog)" in en_word:
                continue
            # remove the (n) and (n,adj) suffixes
            # but keep the (wasei: word) suffixes
            # wasei means a closer-to-katakana word
            wasei = re.search(r"\(wasei:.*?\)", en_word)
            if wasei:
                # search for the wasei word
                en_word = wasei.group(0).replace("(wasei:", "").replace(")", "").strip()
            else:
                en_word = re.sub(r"\(.*?\)", "", en_word)
                en_word = re.sub(r"\{.*?\}", "", en_word)
                en_word = en_word.strip()
            en_word = en_word.lower().replace("-", " ").strip()
            if (
                len(en_word) > 20
                or len(n_kanas) == 0
                or any([c in en_word for c in "()012345678"])
                or len(en_word) < 2
                or any([c not in ascii_letters for c in en_word])
            ):
                continue
            katakana_dict[en_word].extend(list(n_kanas))
    print(f"Extracted {len(katakana_dict)} katakana words from the JMDICT.")
    return katakana_dict


def post_processing(
    wiki_dict: Dict[str, List[str]], jmdict_dict: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    global kata
    katakana_dict = defaultdict(list)
    # combine the two dictionaries
    for en_word, katakana_words in wiki_dict.items():
        katakana_dict[en_word].extend(katakana_words)
    for en_word, katakana_words in jmdict_dict.items():
        katakana_dict[en_word].extend(katakana_words)
    for en_word, katakana_words in katakana_dict.items():
        katakana_dict[en_word] = list(set(katakana_words))
    # remove substrings from a set of katakana words
    # for example, "アメリカ" is a substring of "アメリカ合衆国"
    # if both are present, we remove the shorter "アメリカ" and keep "アメリカ合衆国"
    for en_word, katakana_words in katakana_dict.items():
        katakana_dict[en_word] = sorted(
            katakana_words, key=lambda x: len(x), reverse=True
        )
        n_katakana_words = []
        for katakana in katakana_dict[en_word]:
            # yeah it's dumb O(n^2) but it's not that big
            if not any(katakana in n_kata for n_kata in n_katakana_words):
                for k in katakana:
                    kata.add(k)
                n_katakana_words.append(katakana)
        katakana_dict[en_word] = n_katakana_words
    return katakana_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the Japanese Wiktionary dump.",
        default="vendor/ja-extract.jsonl",
        required=False,
    )
    args = parser.parse_args()
    wiki_dict = extract_wiki(args.path)
    jmdict_dict = extract_jmdict("vendor/edict2")
    katakana_dict = post_processing(wiki_dict, jmdict_dict)
    print(f"Total katakana characters: {len(kata)}")
    print(sorted(list(kata)))
    # save as jsonl
    with open("vendor/katakana_dict.jsonl", "w") as f:
        for en_word, katakana_words in katakana_dict.items():
            f.write(
                json.dumps(
                    {"word": en_word, "kata": katakana_words}, ensure_ascii=False
                )
                + "\n"
            )
