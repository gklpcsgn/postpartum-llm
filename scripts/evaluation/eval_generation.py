import json
import os
from pathlib import Path

import nltk
import torch
from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from peft import PeftConfig, PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


SYSTEM_PROMPT = (
    "You are a concise, factual postpartum and newborn-care assistant. "
    "Use only the provided information from the books. "
    "Answer clearly in a few sentences, without repeating the same sentence "
    "or adding extra filler."
)


def build_prompt(instr: str, inp: str | None, tokenizer) -> str:
    inp = inp or ""
    user_text = f"Instruction: {instr}\nInput: {inp}" if inp else f"Instruction: {instr}"

    if tokenizer.chat_template is not None:
        supports_system_role = "gemma" not in tokenizer.name_or_path.lower()
        if supports_system_role:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"},
            ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        if inp:
            return SYSTEM_PROMPT + "\n\n" + f"Instruction: {instr}\nInput: {inp}\nResponse:"
        else:
            return SYSTEM_PROMPT + "\n\n" + f"Instruction: {instr}\nResponse:"


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def precision_recall_f1(pred_tokens, gold_tokens):
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1

    if common == 0:
        return 0.0, 0.0, 0.0

    prec = common / len(pred_tokens) if pred_tokens else 0.0
    rec = common / len(gold_tokens) if gold_tokens else 0.0
    if prec + rec == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def postprocess_pred(pred_raw: str) -> str:
    """
    Heuristic trimming: cut after first blank line, or if
    another 'Instruction:' appears later in the text.
    """
    pred = pred_raw.strip()

    # Cut at first blank line
    parts = pred.split("\n\n", 1)
    pred = parts[0]

    # If it somehow starts another instruction block, cut there
    idx = pred.find("Instruction:")
    if idx > 0:
        pred = pred[:idx]

    return pred.strip()


def evaluate_model(
    model_dir: str,
    test_file: str,
    max_new_tokens: int = 256,
    limit: int | None = None,
    debug_n: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading PEFT adapter from: {model_dir}")
    config = PeftConfig.from_pretrained(model_dir)

    # Tokenizer: prefer model_dir (training saves it there), fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()
    model.eval()

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Model first parameter device:", next(model.parameters()).device)

    print(f"Loading test data from: {test_file}")
    data = load_jsonl(test_file)
    if limit is not None:
        data = data[:limit]
    print(f"Number of examples: {len(data)}")

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    smooth_fn = SmoothingFunction().method1

    n = 0
    exact = 0
    f1_sum = 0.0
    rouge1_sum = 0.0
    rouge2_sum = 0.0
    rougeL_sum = 0.0
    bleu_sum = 0.0
    meteor_sum = 0.0
    pred_lens = []
    gold_lens = []

    gold_texts = []
    pred_texts = []
    predictions_records = []

    for ex in tqdm(data, desc="Evaluating", total=len(data)):
        instr = ex.get("instruction", "")
        inp = ex.get("input", "")
        gold_raw = ex.get("output", "")

        prompt = build_prompt(instr, inp, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        stop_token_ids = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "additional_special_tokens_ids"):
            stop_token_ids.extend(tokenizer.additional_special_tokens_ids)
        stop_token_ids = [t for t in stop_token_ids if t is not None]

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
            )

        # Decode only the newly generated tokens to avoid prompt-stripping issues
        # with chat-template models (special tokens are stripped differently from
        # the raw prompt string used to build the input).
        input_len = inputs["input_ids"].shape[1]
        new_tokens = gen_ids[0][input_len:]
        pred_raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_raw = postprocess_pred(pred_raw)

        pred_norm = normalize(pred_raw)
        gold_norm = normalize(gold_raw)

        if pred_norm == gold_norm and pred_norm != "":
            exact += 1

        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()

        _, _, f1 = precision_recall_f1(pred_tokens, gold_tokens)
        f1_sum += f1

        rouge_scores = rouge.score(gold_norm, pred_norm)
        rouge1_sum += rouge_scores["rouge1"].fmeasure
        rouge2_sum += rouge_scores["rouge2"].fmeasure
        rougeL_sum += rouge_scores["rougeL"].fmeasure

        if gold_tokens and pred_tokens:
            bleu = sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smooth_fn)
            meteor = nltk_meteor([gold_tokens], pred_tokens)
        else:
            bleu = 0.0
            meteor = 0.0
        bleu_sum += bleu
        meteor_sum += meteor

        pred_lens.append(len(pred_tokens))
        gold_lens.append(len(gold_tokens))

        gold_texts.append(gold_raw)
        pred_texts.append(pred_raw)
        predictions_records.append({
            "instruction": instr,
            "input": inp,
            "gold": gold_raw,
            "prediction": pred_raw,
        })

        if n < debug_n:
            print(f"\n---- EXAMPLE {n} ----")
            print("INSTRUCTION:", instr)
            if inp:
                print("INPUT:", inp)
            print("GOLD:", gold_raw)
            print("PRED:", pred_raw)
            print("---------------------")

        n += 1

    print("Computing BERTScore on all examples...")
    P, R, F = bertscore(pred_texts, gold_texts, lang="en", verbose=False)

    avg_pred_len = sum(pred_lens) / n if n else 0.0
    avg_gold_len = sum(gold_lens) / n if n else 0.0
    length_ratio = avg_pred_len / avg_gold_len if avg_gold_len > 0 else 0.0

    results = {
        "model_dir": model_dir,
        "n_examples": n,
        "exact_match": exact / n if n else 0.0,
        "f1": f1_sum / n if n else 0.0,
        "rouge1": rouge1_sum / n if n else 0.0,
        "rouge2": rouge2_sum / n if n else 0.0,
        "rougeL": rougeL_sum / n if n else 0.0,
        "bleu": bleu_sum / n if n else 0.0,
        "meteor": meteor_sum / n if n else 0.0,
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F.mean()),
        "avg_pred_len": avg_pred_len,
        "avg_gold_len": avg_gold_len,
        "length_ratio": length_ratio,
    }
    return results, predictions_records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to PEFT adapter directory")
    parser.add_argument("--model_key", type=str, required=True,
                        help="Key to write results under in generative_results.json "
                             "(e.g. llama31_8b, gemma2_9b, phi4_14b, qwen25_7b)")
    parser.add_argument("--test_file", type=str,
                        default="data/original/merged_splits/test.jsonl")
    parser.add_argument("--results_file", type=str,
                        default="results/generative_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of test examples")
    parser.add_argument("--debug_n", type=int, default=0,
                        help="Print first N examples with gold/pred for sanity check")
    parser.add_argument("--save_predictions", action="store_true", default=False,
                        help="Save per-example predictions to results/predictions/<model_key>.jsonl")

    args = parser.parse_args()

    res, predictions = evaluate_model(
        model_dir=args.model_dir,
        test_file=args.test_file,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        debug_n=args.debug_n,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

    if args.save_predictions:
        pred_dir = os.path.join(os.path.dirname(args.results_file), "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pred_path = os.path.join(pred_dir, f"{args.model_key}.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for rec in predictions:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {pred_path}")

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    if os.path.exists(args.results_file):
        with open(args.results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[args.model_key] = res

    with open(args.results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results written to {args.results_file} under key '{args.model_key}'")
