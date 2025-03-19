import json

path = "vendor/katakana_dict.jsonl"

with open(path, "r") as f:
    for line in f:
        line = line.rstrip()
        line = json.loads(line)
        word = line["word"]