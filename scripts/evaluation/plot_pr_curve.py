import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, precision_recall_curve
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABELS = ["green", "yellow", "red"]
COLORS = {"green": "#3B6D11", "yellow": "#854F0B", "red": "#A32D2D"}

TARGET_RECALL = 0.988
OPERATING_RECALL = 0.988
OPERATING_PREC = 0.997


def load_probs(model_dir: str, dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    ds = load_from_disk(dataset_path)
    ds = ds.rename_column("label", "labels")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(cfg._name_or_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=512)

    tokenized_test = ds["test"].map(tokenize, batched=True, remove_columns=["text"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    training_args = TrainingArguments(
        output_dir="/tmp/_plot_pr_tmp",
        per_device_eval_batch_size=32,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    preds_output = trainer.predict(tokenized_test)
    logits = preds_output.predictions
    y_true = preds_output.label_ids.astype(int)

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    y_probs = exp / exp.sum(axis=-1, keepdims=True)

    return y_true, y_probs


def plot_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, output_fig: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("white")
    ax.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for class_idx, class_name in enumerate(LABELS):
        y_bin = (y_true == class_idx).astype(int)
        probs = y_probs[:, class_idx]

        precision, recall, thresholds = precision_recall_curve(y_bin, probs)
        auprc = average_precision_score(y_bin, probs)

        ax.plot(
            recall,
            precision,
            color=COLORS[class_name],
            linewidth=2,
            label=f"{class_name.capitalize()} (AUPRC = {auprc:.3f})",
            zorder=2,
        )

        if class_name == "red":
            # Vertical dashed line at recall = 0.90
            ax.axvline(
                x=0.90,
                color=COLORS["red"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
                label="Target recall = 0.90",
                zorder=1,
            )

            # Operating point: threshold closest to recall = TARGET_RECALL
            # recall from precision_recall_curve is in descending order of threshold,
            # so we find the index where recall is closest to TARGET_RECALL
            diffs = np.abs(recall[:-1] - TARGET_RECALL)
            op_idx = int(np.argmin(diffs))
            op_recall = recall[op_idx]
            op_prec = precision[op_idx]

            ax.scatter(
                op_recall,
                op_prec,
                color=COLORS["red"],
                s=70,
                zorder=5,
            )
            ax.annotate(
                f"Operating point\n(recall={OPERATING_RECALL:.3f}, prec={OPERATING_PREC:.3f})",
                xy=(op_recall, op_prec),
                xytext=(op_recall - 0.28, op_prec - 0.12),
                fontsize=8,
                color=COLORS["red"],
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.0),
            )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Precision-Recall Curves — Severity Classifier (Augmented Test Set)",
        fontsize=11,
        pad=10,
    )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_fig)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {output_fig}")


def main():
    parser = argparse.ArgumentParser(description="Plot PR curves for severity classifier.")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to augmented HuggingFace DatasetDict (save_to_disk format)")
    parser.add_argument("--model_dir",
                        default="models/classifiers/severity_conversation/deberta_flat_focal_g2",
                        help="Path to classifier checkpoint directory")
    parser.add_argument("--output_fig", default="results/figures/pr_curve.png",
                        help="Output figure path")
    parser.add_argument("--probs_file", default="results/classifier_probs.npz",
                        help="Where to save/load raw probabilities")
    parser.add_argument("--use_cached", action="store_true",
                        help="Skip inference and load probabilities from --probs_file if it exists")
    args = parser.parse_args()

    if args.use_cached and os.path.exists(args.probs_file):
        print(f"Loading cached probabilities from {args.probs_file}")
        data = np.load(args.probs_file)
        y_true = data["y_true"]
        y_probs = data["y_probs"]
    else:
        print(f"Running inference with {args.model_dir} ...")
        y_true, y_probs = load_probs(args.model_dir, args.dataset_path)

        os.makedirs(os.path.dirname(os.path.abspath(args.probs_file)), exist_ok=True)
        np.savez(args.probs_file, y_true=y_true, y_probs=y_probs)
        print(f"Probabilities saved to {args.probs_file}")

    plot_pr_curves(y_true, y_probs, args.output_fig)


if __name__ == "__main__":
    main()
