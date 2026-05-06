# scripts/classifiers/train_severity_classifier.py

import argparse
import os
import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss

from datasets import Dataset, DatasetDict
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

LABELS = ["green", "yellow", "red"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


class WeightedTrainer(Trainer):
    """Trainer using class-weighted cross-entropy."""

    def __init__(self, class_weights: torch.Tensor, num_labels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class AsymmetricLossTrainer(Trainer):
    """Standard CE + differentiable penalty on red false negatives only.

    For each example where the true label is red (2), adds alpha*(1 - p_red)
    to the loss, pushing the model to assign higher probability to class 2.
    Non-red examples receive no extra penalty.
    """

    def __init__(self, alpha: float, num_labels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss()
        ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = torch.softmax(logits, dim=-1)
        red_mask = (labels == 2).float()
        prob_red = probs[:, 2]
        asymm_penalty = self.alpha * (red_mask * (1.0 - prob_red)).mean()

        loss = ce_loss + asymm_penalty
        return (loss, outputs) if return_outputs else loss


class FocalLossTrainer(Trainer):
    """Focal loss, optionally combined with class weighting via --focal_with_weights."""

    def __init__(
        self,
        gamma: float,
        num_labels: int,
        class_weights: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.num_labels = num_labels
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        ce_fct = CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            reduction="none",
        )
        ce_per = ce_fct(logits.view(-1, self.num_labels), labels.view(-1))

        probs = torch.softmax(logits, dim=-1)
        p_t = probs.gather(1, labels.view(-1, 1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = (focal_weight * ce_per).mean()
        return (loss, outputs) if return_outputs else loss


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def build_conversation_text(ex: dict) -> str:
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


def load_jsonl(path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_dataset(splits_dir: str, augmented_dir: str | None) -> DatasetDict:
    splits_path = Path(splits_dir)
    train_examples = load_jsonl(splits_path / "train.jsonl")
    print(f"Loaded train split: {len(train_examples)} examples")

    if augmented_dir:
        aug_path = Path(augmented_dir)
        if aug_path.is_dir():
            for jsonl_file in sorted(aug_path.glob("*.jsonl")):
                extra = load_jsonl(jsonl_file)
                train_examples.extend(extra)
                print(f"Loaded augmented: {jsonl_file} ({len(extra)} examples)")
        else:
            print(f"Warning: augmented_dir {augmented_dir!r} not found, skipping.")

    val_examples = load_jsonl(splits_path / "val.jsonl")
    test_examples = load_jsonl(splits_path / "test.jsonl")

    def to_hf_dict(examples: list[dict]) -> dict:
        texts, labels = [], []
        for ex in examples:
            if ex.get("severity") not in label2id:
                continue
            texts.append(build_conversation_text(ex))
            labels.append(label2id[ex["severity"]])
        return {"text": texts, "labels": labels}

    return DatasetDict({
        "train": Dataset.from_dict(to_hf_dict(train_examples)),
        "validation": Dataset.from_dict(to_hf_dict(val_examples)),
        "test": Dataset.from_dict(to_hf_dict(test_examples)),
    })


def train(
    dataset_path: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    lr: float = 2e-5,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    num_epochs: int = 8,
    loss_type: str = "weighted_ce",
    alpha: float = 1.0,
    gamma: float = 2.0,
    focal_with_weights: bool = False,
    augmented_dir: str | None = None,
):
    ds = build_dataset(dataset_path, augmented_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=512,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    num_labels = len(LABELS)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        probs = _softmax_np(logits)
        labels_bin = label_binarize(labels, classes=list(range(num_labels)))
        auprc_per_class = average_precision_score(labels_bin, probs, average=None)

        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            "auprc_green": float(auprc_per_class[0]),
            "auprc_yellow": float(auprc_per_class[1]),
            "auprc_red": float(auprc_per_class[2]),
            "mcc": float(matthews_corrcoef(labels, preds)),
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    os.makedirs(output_dir, exist_ok=True)

    # class weights from train split
    train_labels_raw = ds["train"]["labels"]
    counts = Counter(train_labels_raw)
    total = len(train_labels_raw)
    weights = []
    for i in range(num_labels):
        freq = counts[i] / total
        weights.append(1.0 / freq)
    class_weights = torch.tensor(weights, dtype=torch.float)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    common_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0,
            )
        ],
    )

    if loss_type == "weighted_ce":
        trainer = WeightedTrainer(
            class_weights=class_weights,
            num_labels=num_labels,
            **common_kwargs,
        )
    elif loss_type == "asymmetric":
        trainer = AsymmetricLossTrainer(
            alpha=alpha,
            num_labels=num_labels,
            **common_kwargs,
        )
    elif loss_type == "focal":
        trainer = FocalLossTrainer(
            gamma=gamma,
            num_labels=num_labels,
            class_weights=class_weights if focal_with_weights else None,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    trainer.train()

    # save training log history
    log_history = trainer.state.log_history
    log_path = os.path.join(output_dir, "training_log_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    # collect curves
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []
    eval_f1_epochs, eval_f1_vals = [], []
    eval_acc_epochs, eval_acc_vals = [], []

    for record in log_history:
        if "loss" in record and "epoch" in record and "eval_loss" not in record:
            train_epochs.append(record["epoch"])
            train_losses.append(record["loss"])
        if "eval_loss" in record and "epoch" in record:
            eval_epochs.append(record["epoch"])
            eval_losses.append(record["eval_loss"])
        if "eval_f1_macro" in record and "epoch" in record:
            eval_f1_epochs.append(record["epoch"])
            eval_f1_vals.append(record["eval_f1_macro"])
        if "eval_accuracy" in record and "epoch" in record:
            eval_acc_epochs.append(record["epoch"])
            eval_acc_vals.append(record["eval_accuracy"])

    # loss plot
    if train_epochs or eval_epochs:
        plt.figure()
        if train_epochs:
            plt.plot(train_epochs, train_losses, marker="o", label="Train loss")
        if eval_epochs:
            plt.plot(eval_epochs, eval_losses, marker="o", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and validation loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
        plt.close()

    # validation metrics plot
    if eval_f1_epochs or eval_acc_epochs:
        plt.figure()
        if eval_f1_epochs:
            plt.plot(eval_f1_epochs, eval_f1_vals, marker="o", label="Validation F1 (macro)")
        if eval_acc_epochs:
            plt.plot(eval_acc_epochs, eval_acc_vals, marker="o", label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Validation metrics over epochs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "val_metrics_curve.png"), dpi=300)
        plt.close()

    # final eval on test (for overall numbers)
    trainer.evaluate(tokenized["test"])

    # === per-class reports ===
    def save_class_report(split_name: str):
        preds_output = trainer.predict(tokenized[split_name])
        logits = preds_output.predictions
        preds = np.argmax(logits, axis=-1)
        labels = preds_output.label_ids

        report = classification_report(
            labels,
            preds,
            target_names=LABELS,
            digits=3,
            zero_division=0,
        )
        print(f"\n=== {split_name.upper()} CLASSIFICATION REPORT ===")
        print(report)

        out_path = os.path.join(output_dir, f"{split_name}_classification_report.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)

    save_class_report("validation")
    save_class_report("test")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train severity classifier")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to merged_splits directory containing train/val/test.jsonl")
    parser.add_argument("--augmented_dir", default="data/augmented",
                        help="Directory of extra .jsonl files merged into train (default: data/augmented)")
    parser.add_argument("--output_dir", default="models/classifiers/severity_conversation/",
                        help="Where to save checkpoints and artifacts")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument(
        "--loss_type",
        choices=["weighted_ce", "asymmetric", "focal"],
        default="weighted_ce",
        help="Loss function: weighted_ce (default), asymmetric, or focal",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Penalty weight for red false negatives (asymmetric loss only)",
    )
    parser.add_argument(
        "--gamma", type=float, default=2.0,
        help="Focusing parameter for focal loss",
    )
    parser.add_argument(
        "--focal_with_weights", action="store_true",
        help="Combine focal loss with inverse-frequency class weighting",
    )
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        loss_type=args.loss_type,
        alpha=args.alpha,
        gamma=args.gamma,
        focal_with_weights=args.focal_with_weights,
        augmented_dir=args.augmented_dir,
    )
