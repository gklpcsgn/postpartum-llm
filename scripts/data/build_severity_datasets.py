# scripts/build_severity_datasets.py

from datasets import load_dataset, DatasetDict

data_files = {
    "train": "data/datasets/merged_splits/train.jsonl",
    "validation": "data/datasets/merged_splits/val.jsonl",
    "test": "data/datasets/merged_splits/test.jsonl",
}

raw = load_dataset("json", data_files=data_files)

LABELS = ["green", "yellow", "red"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

def build_input_example(ex):
    # user side
    text = ex["instruction"] or ""
    if ex.get("input"):
        text = text + "\n\n" + ex["input"]
    return {
        "text": text,
        "label": label2id[ex["severity"]],
    }

def build_output_example(ex):
    # assistant side
    return {
        "text": ex["output"],
        "label": label2id[ex["severity"]],
    }

input_ds = DatasetDict({
    split: raw[split].map(build_input_example, remove_columns=raw[split].column_names)
    for split in raw
})

output_ds = DatasetDict({
    split: raw[split].map(build_output_example, remove_columns=raw[split].column_names)
    for split in raw
})

input_ds.save_to_disk("severity_dataset_input")
output_ds.save_to_disk("severity_dataset_output")
