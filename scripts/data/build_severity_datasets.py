# scripts/data/build_severity_datasets.py

import json
from pathlib import Path

from datasets import Dataset, DatasetDict

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


# Train: merged_splits/train + all .jsonl files under data/augmented/
train_examples = load_jsonl(SPLITS_DIR / "train.jsonl")
print(f"Loaded train split: {len(train_examples)} examples")

if AUGMENTED_DIR.is_dir():
    for jsonl_file in sorted(AUGMENTED_DIR.glob("*.jsonl")):
        extra = load_jsonl(jsonl_file)
        train_examples.extend(extra)
        print(f"Loaded augmented: {jsonl_file} ({len(extra)} examples)")
else:
    print(f"No augmented dir found at {AUGMENTED_DIR}, skipping.")

val_examples = load_jsonl(SPLITS_DIR / "val.jsonl")
test_examples = load_jsonl(SPLITS_DIR / "test.jsonl")
print(f"Loaded val: {len(val_examples)}, test: {len(test_examples)} examples")

conv_ds = DatasetDict({
    "train": Dataset.from_dict(to_hf_dict(train_examples)),
    "validation": Dataset.from_dict(to_hf_dict(val_examples)),
    "test": Dataset.from_dict(to_hf_dict(test_examples)),
})

conv_ds.save_to_disk("severity_dataset_conversation")
print(f"\nSaved to severity_dataset_conversation")
print(conv_ds)
