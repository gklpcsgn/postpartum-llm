# scripts/train_severity_classifier.py

import os
import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from evaluate import load as load_metric
from sklearn.metrics import classification_report 

LABELS = ["green", "yellow", "red"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

MODEL_NAME = "distilbert-base-uncased"


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


def train(dataset_path: str, output_dir: str):
    ds = load_from_disk(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=256,
        )

    tokenized = ds.map(tokenize, batched=True)

    # make labels explicit for Trainer
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    os.makedirs(output_dir, exist_ok=True)

    # class weights from train split
    train_labels_raw = ds["train"]["label"]
    counts = Counter(train_labels_raw)
    num_labels = len(LABELS)
    total = len(train_labels_raw)
    weights = []
    for i in range(num_labels):
        freq = counts[i] / total
        weights.append(1.0 / freq)
    class_weights = torch.tensor(weights, dtype=torch.float)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        num_labels=num_labels,
        model=model,
        args=args,
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

    # validation + test per-class metrics
    save_class_report("validation")
    save_class_report("test")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # 1) train input severity model
    train("severity_dataset_input", "models/severity_input")

    # 2) train output severity model (comment out if you want to run separately)
    train("severity_dataset_output", "models/severity_output")
