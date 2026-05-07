"""
augment_synthetic_red.py

Generate synthetic red-class examples using the Gemini API.
Output is appended to data/augmented/synthetic_red.jsonl.

Usage:
    python scripts/data/augment_synthetic_red.py \
        --target 300 \
        --output data/augmented/synthetic_red.jsonl \
        --model gemini-1.5-flash

Schema (matches existing dataset):
    instruction : str   — user message
    input       : str   — optional context (usually "")
    output      : str   — assistant response
    topic       : str   — topic label
    severity    : str   — always "red" here
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You are a data-generation assistant helping build a postpartum support chatbot dataset.
Your task is to write realistic user messages that describe postpartum or newborn emergencies.
These messages will be used to train a safety classifier to detect high-risk situations.

Rules:
- Write only the user message (instruction field). Do NOT write an assistant response.
- The situation must be genuinely emergent — something requiring immediate medical attention.
- Cover maternal emergencies (hemorrhage, eclampsia, pulmonary embolism, sepsis, severe PPD,
  suicidal ideation, psychosis) and newborn emergencies (apnea, cyanosis, seizures,
  high fever in newborns under 3 months, not feeding for extended periods, limpness).
- Messages should sound like a real worried parent or patient — informal, first-person, urgent.
- Vary length, vocabulary, and level of medical knowledge across examples.
- Do NOT include clinical recommendations or diagnoses in the user message.
- Return ONLY a JSON array of objects. No markdown, no commentary.
"""

GENERATION_PROMPT = """\
Generate {batch_size} user messages describing postpartum or newborn emergency situations.

Return a JSON array where each element has exactly these fields:
{{
  "instruction": "<the user message>",
  "input": "",
  "output": "",
  "topic": "<one of: maternal_emergency | newborn_emergency | mental_health_emergency>",
  "severity": "red"
}}

Vary the situations across maternal and newborn emergencies. Do not repeat the same scenario.
"""

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"instruction", "input", "output", "topic", "severity"}
VALID_TOPICS = {"maternal_emergency", "newborn_emergency", "mental_health_emergency"}


def validate_example(ex: dict) -> tuple[bool, str]:
    if not isinstance(ex, dict):
        return False, "not a dict"
    missing = REQUIRED_FIELDS - ex.keys()
    if missing:
        return False, f"missing fields: {missing}"
    if ex.get("severity") != "red":
        return False, f"severity is not red: {ex.get('severity')}"
    if ex.get("topic") not in VALID_TOPICS:
        return False, f"invalid topic: {ex.get('topic')}"
    if not isinstance(ex.get("instruction"), str) or len(ex["instruction"].strip()) < 10:
        return False, "instruction too short or not a string"
    return True, "ok"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> list[dict]:
    """Extract a JSON array from model output, stripping markdown fences if present."""
    text = text.strip()
    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def generate_batch(client, model_name: str, batch_size: int, retries: int = 3) -> list[dict]:
    prompt = GENERATION_PROMPT.format(batch_size=batch_size)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )
            candidates = parse_json_response(response.text)
            if not isinstance(candidates, list):
                raise ValueError("Response is not a JSON array")
            return candidates
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    print("  All retries exhausted for this batch.")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

    parser = argparse.ArgumentParser(description="Generate synthetic red-class examples via Gemini.")
    parser.add_argument("--target",    type=int, default=300, help="Number of examples to generate")
    parser.add_argument("--batch_size",type=int, default=10,  help="Examples per API call")
    parser.add_argument("--output",    default="data/augmented/synthetic_red.jsonl")
    parser.add_argument("--model",     default="gemini-1.5-flash")
    parser.add_argument("--rpm",       type=int, default=15, help="Max requests per minute")
    args = parser.parse_args()

    client = genai.Client(api_key=api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-generated examples to avoid re-running from scratch
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = [json.loads(l) for l in f if l.strip()]
        print(f"Found {len(existing)} existing examples in {output_path}")

    collected = len(existing)
    delay = 60.0 / args.rpm  # seconds between requests to stay under rate limit

    print(f"Target: {args.target} examples. Need {max(0, args.target - collected)} more.")

    with open(output_path, "a") as out_f:
        while collected < args.target:
            remaining = args.target - collected
            batch_size = min(args.batch_size, remaining)
            print(f"[{collected}/{args.target}] Generating batch of {batch_size}...")

            candidates = generate_batch(client, args.model, batch_size)
            accepted = 0
            for ex in candidates:
                ok, reason = validate_example(ex)
                if ok:
                    out_f.write(json.dumps(ex) + "\n")
                    accepted += 1
                else:
                    print(f"  Skipped example: {reason}")

            collected += accepted
            print(f"  Accepted {accepted}/{len(candidates)} examples.")

            if collected < args.target:
                time.sleep(delay)

    print(f"\nDone. {collected} red examples saved to {output_path}")


if __name__ == "__main__":
    main()
