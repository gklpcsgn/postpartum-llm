# scripts/classifiers/evaluate_saved_model.py
#
# Usage:
#   python scripts/classifiers/evaluate_saved_model.py \
#       --model_dir models/classifiers/severity_conversation/deberta_flat_wce \
#       --dataset_path data/augmented/severity_dataset_conversation

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import classification_report, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABELS = ["green", "yellow", "red"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved severity classifier on the test split.")
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing the saved model and tokenizer")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to pre-built HuggingFace DatasetDict (saved with save_to_disk)")
    parser.add_argument("--model_name", default=None,
                        help="HuggingFace model name, used as tokenizer fallback if not saved in model_dir")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # --- Dataset ---
    ds = load_from_disk(args.dataset_path)
    ds = ds.rename_column("label", "labels")

    # --- Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except Exception:
        if not args.model_name:
            raise ValueError(
                "No tokenizer found in model_dir and --model_name was not provided."
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=512)

    tokenized_test = ds["test"].map(tokenize, batched=True, remove_columns=["text"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    # --- Inference ---
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    preds_output = trainer.predict(tokenized_test)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    # --- Print report ---
    report_str = classification_report(labels, preds, target_names=LABELS, digits=3, zero_division=0)
    print("\n=== TEST CLASSIFICATION REPORT ===")
    print(report_str)

    # --- Read existing summary for metadata fields ---
    summary_path = os.path.join(args.model_dir, "results_summary.json")
    existing: dict = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            existing = json.load(f)

    model_name = existing.get("model_name") or args.model_name or args.model_dir

    # --- Build and save summary ---
    report_dict = classification_report(
        labels, preds, target_names=LABELS, zero_division=0, output_dict=True
    )

    summary = {
        "model_name": model_name,
        "loss_type": existing.get("loss_type", "unknown"),
        "alpha": existing.get("alpha"),
        "gamma": existing.get("gamma"),
        "focal_with_weights": existing.get("focal_with_weights", False),
        "best_epoch": existing.get("best_epoch"),
        "test_accuracy": report_dict["accuracy"],
        "test_macro_f1": report_dict["macro avg"]["f1-score"],
        "test_macro_precision": report_dict["macro avg"]["precision"],
        "test_macro_recall": report_dict["macro avg"]["recall"],
        "test_mcc": float(matthews_corrcoef(labels, preds)),
        "test_red_precision": report_dict["red"]["precision"],
        "test_red_recall": report_dict["red"]["recall"],
        "test_red_f1": report_dict["red"]["f1-score"],
        "test_green_precision": report_dict["green"]["precision"],
        "test_green_recall": report_dict["green"]["recall"],
        "test_green_f1": report_dict["green"]["f1-score"],
        "test_yellow_precision": report_dict["yellow"]["precision"],
        "test_yellow_recall": report_dict["yellow"]["recall"],
        "test_yellow_f1": report_dict["yellow"]["f1-score"],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results summary to {summary_path}")


if __name__ == "__main__":
    main()
