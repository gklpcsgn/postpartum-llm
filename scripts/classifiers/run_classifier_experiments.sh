#!/bin/bash
# run_classifier_experiments.sh
# Runs all remaining classifier experiments consecutively.
# Each experiment saves results_summary.json to its output directory.
# Run from repo root: bash scripts/classifiers/run_classifier_experiments.sh

set -e  # stop on first error

DATASET="data/augmented/severity_dataset_conversation"
BASE="models/classifiers/severity_conversation"
DEBERTA="microsoft/deberta-v3-base"
MENTAL="mental/mental-roberta-base"

echo "================================================================"
echo "Starting classifier experiment sweep"
echo "$(date)"
echo "================================================================"

# ── DeBERTa asymmetric α=50 ──────────────────────────────────────────
echo ""
echo "[1/10] DeBERTa asymmetric α=50"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $DEBERTA \
    --dataset_path $DATASET \
    --output_dir $BASE/deberta_flat_asym_a50 \
    --loss_type asymmetric \
    --alpha 50
echo "[1/10] Done — $(date)"

# ── DeBERTa asymmetric α=100 ─────────────────────────────────────────
echo ""
echo "[2/10] DeBERTa asymmetric α=100"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $DEBERTA \
    --dataset_path $DATASET \
    --output_dir $BASE/deberta_flat_asym_a100 \
    --loss_type asymmetric \
    --alpha 100
echo "[2/10] Done — $(date)"

# ── DeBERTa focal γ=1 ────────────────────────────────────────────────
echo ""
echo "[3/10] DeBERTa focal γ=1"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $DEBERTA \
    --dataset_path $DATASET \
    --output_dir $BASE/deberta_flat_focal_g1 \
    --loss_type focal \
    --gamma 1
echo "[3/10] Done — $(date)"

# ── DeBERTa focal γ=2 ────────────────────────────────────────────────
echo ""
echo "[4/10] DeBERTa focal γ=2"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $DEBERTA \
    --dataset_path $DATASET \
    --output_dir $BASE/deberta_flat_focal_g2 \
    --loss_type focal \
    --gamma 2
echo "[4/10] Done — $(date)"

# ── DeBERTa focal γ=5 ────────────────────────────────────────────────
echo ""
echo "[5/10] DeBERTa focal γ=5"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $DEBERTA \
    --dataset_path $DATASET \
    --output_dir $BASE/deberta_flat_focal_g5 \
    --loss_type focal \
    --gamma 5
echo "[5/10] Done — $(date)"

# ── MentalRoBERTa asymmetric α=10 ────────────────────────────────────
echo ""
echo "[6/10] MentalRoBERTa asymmetric α=10"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $MENTAL \
    --dataset_path $DATASET \
    --output_dir $BASE/mentalroberta_flat_asym_a10 \
    --loss_type asymmetric \
    --alpha 10
echo "[6/10] Done — $(date)"

# ── MentalRoBERTa asymmetric α=50 ────────────────────────────────────
echo ""
echo "[7/10] MentalRoBERTa asymmetric α=50"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $MENTAL \
    --dataset_path $DATASET \
    --output_dir $BASE/mentalroberta_flat_asym_a50 \
    --loss_type asymmetric \
    --alpha 50
echo "[7/10] Done — $(date)"

# ── MentalRoBERTa asymmetric α=100 ───────────────────────────────────
echo ""
echo "[8/10] MentalRoBERTa asymmetric α=100"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $MENTAL \
    --dataset_path $DATASET \
    --output_dir $BASE/mentalroberta_flat_asym_a100 \
    --loss_type asymmetric \
    --alpha 100
echo "[8/10] Done — $(date)"

# ── MentalRoBERTa focal γ=1 ──────────────────────────────────────────
echo ""
echo "[9/10] MentalRoBERTa focal γ=1"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $MENTAL \
    --dataset_path $DATASET \
    --output_dir $BASE/mentalroberta_flat_focal_g1 \
    --loss_type focal \
    --gamma 1
echo "[9/10] Done — $(date)"

# ── MentalRoBERTa focal γ=2 ──────────────────────────────────────────
echo ""
echo "[10/10] MentalRoBERTa focal γ=2"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $MENTAL \
    --dataset_path $DATASET \
    --output_dir $BASE/mentalroberta_flat_focal_g2 \
    --loss_type focal \
    --gamma 2
echo "[10/10] Done — $(date)"

# ── MentalRoBERTa focal γ=5 ──────────────────────────────────────────
echo ""
echo "[11/11] MentalRoBERTa focal γ=5"
python scripts/classifiers/train_severity_classifier.py \
    --model_name $MENTAL \
    --dataset_path $DATASET \
    --output_dir $BASE/mentalroberta_flat_focal_g5 \
    --loss_type focal \
    --gamma 5
echo "[11/11] Done — $(date)"

echo ""
echo "================================================================"
echo "All experiments complete — $(date)"
echo "================================================================"