#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN="$SCRIPT_DIR/train_generative.py"
LOG_DIR="$REPO_ROOT/results/generative_training_logs"

mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_model() {
    local name="$1"
    local model_name="$2"
    local output_dir="$3"
    local log_file="$4"

    echo "[$(ts)] Starting $name"
    python "$TRAIN" \
        --model_name  "$model_name" \
        --output_dir  "$REPO_ROOT/$output_dir" \
        --train_file  "$REPO_ROOT/data/original/merged_splits/train.jsonl" \
        --val_file    "$REPO_ROOT/data/original/merged_splits/val.jsonl" \
        --seed        42 \
        2>&1 | tee "$log_file"
    echo "[$(ts)] Finished $name"
}

run_model \
    "llama31_8b" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "models/generative/llama31_8b_postpartum_qlora" \
    "$LOG_DIR/llama31_8b.log"

run_model \
    "gemma2_9b" \
    "google/gemma-2-9b-it" \
    "models/generative/gemma2_9b_postpartum_qlora" \
    "$LOG_DIR/gemma2_9b.log"

run_model \
    "phi4_14b" \
    "microsoft/phi-4" \
    "models/generative/phi4_14b_postpartum_qlora" \
    "$LOG_DIR/phi4_14b.log"

run_model \
    "qwen25_7b" \
    "Qwen/Qwen2.5-7B-Instruct" \
    "models/generative/qwen25_7b_postpartum_qlora" \
    "$LOG_DIR/qwen25_7b.log"

echo "All four models trained. Logs in results/generative_training_logs/"
