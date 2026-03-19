"""
Splits dataset.jsonl into train / validation / test JSONL files.
Run once:  python data/instruction_dataset/prepare_splits.py
"""

import json, random, pathlib

HERE = pathlib.Path(__file__).resolve().parent
SRC  = HERE / "dataset.jsonl"

random.seed(42)

with open(SRC, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

random.shuffle(rows)

n = len(rows)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

splits = {
    "train.jsonl": rows[:train_end],
    "val.jsonl":   rows[train_end:val_end],
    "test.jsonl":  rows[val_end:],
}

for name, data in splits.items():
    out = HERE / name
    with open(out, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(data)} examples")

print("Done.")
