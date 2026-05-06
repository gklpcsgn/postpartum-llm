"""
convert_edinburgh.py

Convert edinburgh_responses.json into classifier training examples.

Format:
    Top-level responses (paired with every question_variant):
        A: <question_variant>
        User: <response_example>

    Followup responses (paired with every followup_question, prior context sampled):
        A: <sampled question_variant>       ← one sample, not full cartesian product
        User: <sampled moderate_example>    ← one sample, provides context
        A: <followup_question>              ← all followup_questions enumerated
        User: <followup_response_example>   ← all followup response examples enumerated

Rationale: enumerating all (question_variant × moderate_example × followup_question ×
followup_response) combinations causes a combinatorial explosion (>1M rows for Q1 alone).
The prior context (question + moderate response) exists only to make the followup response
interpretable — one representative sample is sufficient for that purpose.

Output schema:
    instruction : conversation string
    input       : ""
    output      : ""
    topic       : question title (snake_cased)
    severity    : green / yellow / red (severity of the final user turn)

Usage:
    python scripts/data/convert_edinburgh.py \
        --input  data/original/edinburgh_responses.json \
        --output data/original/edinburgh_converted.jsonl \
        --seed   42
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

PLACEHOLDERS = {
    "any indication of self-harm or suicidality",
    "any responses indicating self-harm or suicidality",
}


def is_placeholder(text: str) -> bool:
    return text.strip().lower() in PLACEHOLDERS


def snake_case(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def process_question(q: dict, severity_mapping: dict) -> list[dict]:
    rows = []
    topic = snake_case(q["title"])
    question_variants = q.get("question_variants", [])
    responses = q.get("responses", {})

    for label, data in responses.items():
        if not isinstance(data, dict):
            continue
        severity = severity_mapping.get(label.capitalize())
        if not severity:
            continue

        top_level_examples = [e for e in data.get("examples", []) if not is_placeholder(e)]
        followup_questions = data.get("followup_questions", [])
        followup_responses = data.get("followup_responses", {})

        # ── Top-level responses ───────────────────────────────────────────────
        # Pair each response example with every question_variant
        for example in top_level_examples:
            for question in question_variants:
                rows.append({
                    "instruction": f"A: {question}\nUser: {example}",
                    "input": "",
                    "output": "",
                    "topic": topic,
                    "severity": severity,
                })

        # ── Followup responses ────────────────────────────────────────────────
        # Pair each followup response with every followup_question.
        # Sample one question_variant and one moderate_example as prior context.
        if not followup_questions or not top_level_examples or not question_variants:
            continue

        for fu_label, fu_data in followup_responses.items():
            if not isinstance(fu_data, dict):
                continue
            fu_severity = severity_mapping.get(fu_label.capitalize())
            if not fu_severity:
                continue
            fu_examples = [e for e in fu_data.get("examples", []) if not is_placeholder(e)]

            for fu_example in fu_examples:
                for fu_question in followup_questions:
                    # Sample one representative prior context
                    prior_question = random.choice(question_variants)
                    prior_response = random.choice(top_level_examples)
                    conversation = (
                        f"A: {prior_question}\n"
                        f"User: {prior_response}\n"
                        f"A: {fu_question}\n"
                        f"User: {fu_example}"
                    )
                    rows.append({
                        "instruction": conversation,
                        "input": "",
                        "output": "",
                        "topic": topic,
                        "severity": fu_severity,
                    })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/original/edinburgh_responses.json")
    parser.add_argument("--output", default="data/original/edinburgh_converted.jsonl")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    severity_mapping = data["meta"]["severity_mapping"]
    questions = data["questions"]

    all_rows = []
    for q in questions:
        all_rows.extend(process_question(q, severity_mapping))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    counts = Counter(r["severity"] for r in all_rows)
    print(f"Wrote {len(all_rows)} examples to {output_path}")
    print(f"  green  : {counts.get('green', 0)}")
    print(f"  yellow : {counts.get('yellow', 0)}")
    print(f"  red    : {counts.get('red', 0)}")


if __name__ == "__main__":
    main()
