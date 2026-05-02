import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from tqdm import tqdm

from bert_score import score as bertscore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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
    """
You are a postpartum and newborn-care assistant.

Keep answers brief: 1–3 sentences. Stay strictly within the information stated
in the question.

Do NOT add or infer any symptoms, behaviors, emotions, risks, consequences,
or outcomes unless the question explicitly names them. Do not speculate or
describe what "may" or "might" happen.

Do NOT change or reinterpret any detail of the scenario, including who the
caregiver is, where the child is, or what the parent is doing.

If the question asks “what happens,” respond only with what the parent already
stated. If the question does not state a specific reaction, give a general
statement such as “Babies can react differently,” without adding any new
behaviors or outcomes.

Do NOT add nursing patterns, feeding behavior, milk supply details, or any
physiological explanations unless the question explicitly mentions them.

Do NOT add organizations, studies, statistics, or numbers unless already named
in the question.

If the parent expresses worry, you may add ONE short, general reassurance that
does not repeat wording and does not add facts.

Do not make general statements about what “many parents” do unless the question explicitly refers to other parents.


Answer only the question as written. No speculation, no added details.



"""


) 

def build_prompt(instr: str, inp: str | None) -> str:
    """
    Must match your training format:

      Instruction: ...
      Input: ...
      Response:

    or without Input if empty.
    """
    inp = inp or ""
    if inp:
        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + f"Instruction: {instr}\nInput: {inp}\nResponse:"
        )
    else:
        prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + f"Instruction: {instr}\nResponse:"
        )
    return prompt



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

    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
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
    pred_lens = []
    gold_lens = []

    gold_texts = []
    pred_texts = []

    for ex in tqdm(data, desc="Evaluating", total=len(data)):
        instr = ex.get("instruction", "")
        inp = ex.get("input", "")
        gold_raw = ex.get("output", "")

        prompt = build_prompt(instr, inp)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Remove prompt echo if present
        if full_text.startswith(prompt):
            pred_raw = full_text[len(prompt):]
        else:
            pred_raw = full_text

        pred_raw = postprocess_pred(pred_raw)

        pred_norm = normalize(pred_raw)
        gold_norm = normalize(gold_raw)

        if pred_norm == gold_norm and pred_norm != "":
            exact += 1

        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()

        _, _, f1 = precision_recall_f1(pred_tokens, gold_tokens)
        f1_sum += f1

        # ROUGE
        rouge_scores = rouge.score(gold_norm, pred_norm)
        rouge1_sum += rouge_scores["rouge1"].fmeasure
        rouge2_sum += rouge_scores["rouge2"].fmeasure
        rougeL_sum += rouge_scores["rougeL"].fmeasure

        # BLEU
        if gold_tokens and pred_tokens:
            bleu = sentence_bleu(
                [gold_tokens],
                pred_tokens,
                smoothing_function=smooth_fn,
            )
        else:
            bleu = 0.0
        bleu_sum += bleu

        pred_lens.append(len(pred_tokens))
        gold_lens.append(len(gold_tokens))

        gold_texts.append(gold_raw)
        pred_texts.append(pred_raw)

        if n < debug_n:
            print(f"\n---- EXAMPLE {n} ----")
            print("INSTRUCTION:", instr)
            if inp:
                print("INPUT:", inp)
            print("GOLD:", gold_raw)
            print("PRED:", pred_raw)
            print("---------------------")

        n += 1

    # BERTScore over the whole set
    print("Computing BERTScore on all examples...")
    P, R, F = bertscore(
        pred_texts,
        gold_texts,
        lang="en",
        verbose=False,
    )

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
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F.mean()),
        "avg_pred_len": avg_pred_len,
        "avg_gold_len": avg_gold_len,
        "length_ratio": length_ratio,
    }
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to finetuned (or base) model directory",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/datasets/merged_splits/test.jsonl",
        help="Path to test JSONL file",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of test examples",
    )
    parser.add_argument(
        "--debug_n",
        type=int,
        default=0,
        help="Print first N examples with gold/pred for sanity check",
    )

    args = parser.parse_args()

    res = evaluate_model(
        model_dir=args.model_dir,
        test_file=args.test_file,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        debug_n=args.debug_n,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
