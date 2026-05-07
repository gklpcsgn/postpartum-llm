# scripts/data/build_severity_datasets.py

import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

LABELS = ["green", "yellow", "red"]
label2id = {l: i for i, l in enumerate(LABELS)}

SPLITS_DIR = Path("data/original/merged_splits")
AUGMENTED_DIR = Path("data/augmented")


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_conversation_text(ex: dict) -> str:
    """
    Format a training example as a conversation string.

    Edinburgh-style examples (instruction starts with 'A:') are kept as-is.
    Regular QA examples are formatted as:
        User: <instruction>
        A: <input>        (omitted when input is empty)
        Draft: <output>   (omitted when output is empty)
    """
    instruction = ex.get("instruction") or ""
    inp = ex.get("input") or ""
    output = ex.get("output") or ""
    if instruction.startswith("A:"):
        return instruction
    parts = [f"User: {instruction}"]
    if inp:
        parts.append(f"A: {inp}")
    if output:
        parts.append(f"Draft: {output}")
    return "\n".join(parts)


def to_hf_dict(examples: list[dict]) -> dict:
    texts, labels = [], []
    for ex in examples:
        if ex.get("severity") not in label2id:
            continue
        texts.append(build_conversation_text(ex))
        labels.append(label2id[ex["severity"]])
    return {"text": texts, "label": labels}


def print_split_stats(name: str, examples: list[dict]) -> None:
    counts = Counter(ex["severity"] for ex in examples if ex.get("severity") in label2id)
    total = sum(counts.values())
    print(f"  {name}: {total} examples")
    for label in LABELS:
        n = counts.get(label, 0)
        print(f"    {label:8s}: {n:5d}  ({100 * n / total:.1f}%)")


# --- Load all data ---

all_examples: list[dict] = []

for split_file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    examples = load_jsonl(SPLITS_DIR / split_file)
    all_examples.extend(examples)
    print(f"Loaded original {split_file}: {len(examples)} examples")

if AUGMENTED_DIR.is_dir():
    for jsonl_file in sorted(AUGMENTED_DIR.glob("*.jsonl")):
        extra = load_jsonl(jsonl_file)
        all_examples.extend(extra)
        print(f"Loaded augmented {jsonl_file.name}: {len(extra)} examples")
else:
    print(f"No augmented dir found at {AUGMENTED_DIR}, skipping.")

# Drop examples with unknown severity before splitting
all_examples = [ex for ex in all_examples if ex.get("severity") in label2id]
print(f"\nTotal after filtering: {len(all_examples)} examples")

# --- Stratified 80 / 10 / 10 split ---

severity_labels = [ex["severity"] for ex in all_examples]

train_examples, temp_examples = train_test_split(
    all_examples,
    test_size=0.20,
    random_state=42,
    stratify=severity_labels,
)
temp_labels = [ex["severity"] for ex in temp_examples]
val_examples, test_examples = train_test_split(
    temp_examples,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels,
)

print("\nSplit sizes and label distributions:")
print_split_stats("train", train_examples)
print_split_stats("val  ", val_examples)
print_split_stats("test ", test_examples)

# --- Save ---

conv_ds = DatasetDict({
    "train":      Dataset.from_dict(to_hf_dict(train_examples)),
    "validation": Dataset.from_dict(to_hf_dict(val_examples)),
    "test":       Dataset.from_dict(to_hf_dict(test_examples)),
})

conv_ds.save_to_disk("data/augmented/severity_dataset_conversation")
print(f"\nSaved to data/augmented/severity_dataset_conversation")
print(conv_ds)
