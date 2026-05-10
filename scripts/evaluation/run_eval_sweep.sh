#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL="$SCRIPT_DIR/eval_generation.py"
LOG_DIR="$REPO_ROOT/results/generative_eval_logs"

mkdir -p "$LOG_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_eval() {
    local name="$1"
    local model_dir="$2"
    local log_file="$LOG_DIR/${name}.log"

    echo "[$(ts)] Starting eval: $name"
    python "$EVAL" \
        --model_dir         "$REPO_ROOT/$model_dir" \
        --model_key         "$name" \
        --test_file         "$REPO_ROOT/data/original/merged_splits/test.jsonl" \
        --results_file      "$REPO_ROOT/results/generative_results.json" \
        --save_predictions \
        2>&1 | tee "$log_file"
    echo "[$(ts)] Finished eval: $name"
    echo ""
}

run_eval "llama31_8b"  "models/generative/llama31_8b_postpartum_qlora"
run_eval "gemma2_9b"   "models/generative/gemma2_9b_postpartum_qlora"
run_eval "phi4_14b"    "models/generative/phi4_14b_postpartum_qlora"
run_eval "qwen25_7b"   "models/generative/qwen25_7b_postpartum_qlora"

echo "All four evals complete. Results in results/generative_results.json"
echo "Logs in results/generative_eval_logs/"
