import json
import random
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_FILES = [
    "data/datasets/caring_baby_young_child_cleaned.jsonl",
    "data/datasets/therapy_and_the_postpartum_women_cleaned.jsonl",
    "data/datasets/womanly_art_of_breastfeeding_cleaned.jsonl",
]
SOURCE_NAMES = [
    "caring_baby_young_child",
    "therapy_postpartum_woman",
    "womanly_art_breastfeeding",
]
OUTPUT_DIR = "data/datasets/merged_splits"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42

# -----------------------------
# Helper functions
# -----------------------------
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def save_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def stratified_split(items, train_ratio, val_ratio, test_ratio):
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

# -----------------------------
# MAIN
# -----------------------------
def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    global_train, global_val, global_test = [], [], []

    for file_path, source_name in zip(INPUT_FILES, SOURCE_NAMES):
        print(f"Loading: {file_path} (source={source_name})")
        items = load_jsonl(file_path)

        # tag source if not already present
        for ex in items:
            ex.setdefault("source", source_name)

        print(f"  Items from {source_name}: {len(items)}")

        train, val, test = stratified_split(
            items, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )

        print(
            f"  Split {source_name} -> "
            f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
        )

        global_train.extend(train)
        global_val.extend(val)
        global_test.extend(test)

    # Final shuffle within each split
    random.shuffle(global_train)
    random.shuffle(global_val)
    random.shuffle(global_test)

    print(
        f"TOTAL after merge:\n"
        f"  Train: {len(global_train)}\n"
        f"  Val:   {len(global_val)}\n"
        f"  Test:  {len(global_test)}"
    )

    save_jsonl(f"{OUTPUT_DIR}/train.jsonl", global_train)
    save_jsonl(f"{OUTPUT_DIR}/val.jsonl", global_val)
    save_jsonl(f"{OUTPUT_DIR}/test.jsonl", global_test)

    print("Saved:")
    print(f"- {OUTPUT_DIR}/train.jsonl")
    print(f"- {OUTPUT_DIR}/val.jsonl")
    print(f"- {OUTPUT_DIR}/test.jsonl")


if __name__ == "__main__":
    main()
