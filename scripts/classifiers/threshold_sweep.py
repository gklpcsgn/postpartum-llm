# scripts/classifiers/threshold_sweep.py

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import precision_recall_curve

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_logits(
    checkpoint: str, dataset_path: str, split: str
) -> tuple[np.ndarray, np.ndarray]:
    ds = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=256)

    tokenized = ds.map(tokenize, batched=True)

    if "label" in tokenized[split].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    tmp_args = TrainingArguments(
        output_dir="/tmp/_threshold_sweep",
        per_device_eval_batch_size=32,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=tmp_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    pred_output = trainer.predict(tokenized[split])
    return pred_output.predictions, pred_output.label_ids


def sweep(
    logits: np.ndarray, labels: np.ndarray, thresholds: list[float]
) -> list[dict]:
    probs = _softmax_np(logits)
    prob_red = probs[:, 2]

    results = []
    for thr in thresholds:
        pred_is_red = prob_red >= thr
        # Among non-red predictions, pick the higher-scoring of green/yellow
        non_red_preds = np.argmax(probs[:, :2], axis=-1)
        final_preds = np.where(pred_is_red, 2, non_red_preds)

        tp = int(np.sum((final_preds == 2) & (labels == 2)))
        fp = int(np.sum((final_preds == 2) & (labels != 2)))
        fn = int(np.sum((final_preds != 2) & (labels == 2)))

        red_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        red_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        red_f1 = (
            2 * red_prec * red_rec / (red_prec + red_rec)
            if (red_prec + red_rec) > 0
            else 0.0
        )

        non_red_mask = labels != 2
        far = (
            float(np.sum((final_preds == 2) & non_red_mask)) / float(np.sum(non_red_mask))
            if np.sum(non_red_mask) > 0
            else 0.0
        )

        accuracy = float(np.mean(final_preds == labels))

        results.append(
            {
                "threshold": thr,
                "red_precision": float(red_prec),
                "red_recall": float(red_rec),
                "red_f1": float(red_f1),
                "false_alarm_rate": float(far),
                "accuracy": accuracy,
            }
        )
    return results


def compute_precision_at_90_recall(
    prob_red: np.ndarray, labels: np.ndarray
) -> float | None:
    y_bin = (labels == 2).astype(int)
    prec_arr, rec_arr, _ = precision_recall_curve(y_bin, prob_red)

    # prec_arr/rec_arr have length n+1; the last element is the sklearn-added anchor
    # (precision=1.0, recall=0.0) with no corresponding threshold — exclude it.
    prec_work = prec_arr[:-1]
    rec_work = rec_arr[:-1]

    # rec_work is decreasing (higher index = higher threshold = lower recall).
    # Among all indices where recall >= 0.90, take the last one: the highest
    # threshold that still achieves >= 90% recall, which also gives the highest
    # precision at that recall level.
    idx_90 = np.where(rec_work >= 0.90)[0]
    if len(idx_90) == 0:
        return None
    return float(prec_work[idx_90[-1]])


def plot_pr_curve(
    prob_red: np.ndarray,
    labels: np.ndarray,
    sweep_results: list[dict],
    out_path: str,
) -> None:
    y_bin = (labels == 2).astype(int)
    prec_arr, rec_arr, _ = precision_recall_curve(y_bin, prob_red)

    plt.figure(figsize=(7, 5))
    plt.plot(rec_arr, prec_arr, lw=2, label="PR curve (red class)")

    for r in sweep_results:
        plt.scatter(r["red_recall"], r["red_precision"], zorder=5, s=60)
        plt.annotate(
            f"  thr={r['threshold']:.2f}",
            xy=(r["red_recall"], r["red_precision"]),
            fontsize=8,
        )

    plt.xlabel("Recall (red)")
    plt.ylabel("Precision (red)")
    plt.title("Precision-Recall curve — red class")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"PR curve saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep for red-class detection")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained classifier checkpoint (load_from_disk compatible)")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to HuggingFace arrow dataset")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output_json", default="results/cl1_threshold_sweep.json")
    parser.add_argument("--plot_path", default="results/cl1_pr_curve.png")
    args = parser.parse_args()

    print(f"Loading model from: {args.checkpoint}")
    logits, labels = load_logits(args.checkpoint, args.dataset_path, args.split)
    print(f"Loaded {len(labels)} examples from '{args.split}' split")

    probs = _softmax_np(logits)
    prob_red = probs[:, 2]

    print("Running threshold sweep...")
    sweep_results = sweep(logits, labels, THRESHOLDS)
    prec_90 = compute_precision_at_90_recall(prob_red, labels)

    output = {
        "checkpoint": args.checkpoint,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "thresholds": sweep_results,
        "precision_at_90_recall": prec_90,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output_json}")

    print(
        f"\n{'Threshold':>10} {'RedPrec':>9} {'RedRec':>8} "
        f"{'RedF1':>7} {'FAR':>8} {'Acc':>8}"
    )
    for r in sweep_results:
        print(
            f"{r['threshold']:>10.2f} {r['red_precision']:>9.3f} {r['red_recall']:>8.3f} "
            f"{r['red_f1']:>7.3f} {r['false_alarm_rate']:>8.3f} {r['accuracy']:>8.3f}"
        )

    if prec_90 is not None:
        print(f"\nPrecision @ 90% red recall: {prec_90:.3f}")
    else:
        print("\nPrecision @ 90% red recall: null (no threshold achieves >= 90% recall)")

    plot_pr_curve(prob_red, labels, sweep_results, args.plot_path)


if __name__ == "__main__":
    main()
