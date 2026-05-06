"""
augment_back_translate.py

Augment existing red-class examples via back-translation using the Gemini API.
Each example is translated to an intermediate language and back to English,
producing surface-form variation while preserving semantic content.

Usage:
    python scripts/data/augment_back_translate.py \
        --input data/original/merged_splits/train.jsonl \
        --output data/augmented/back_translated_red.jsonl \
        --languages Spanish French German Turkish \
        --model gemini-1.5-flash

Only red-class examples from the input file are processed.
Each example is back-translated once per specified intermediate language.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

BACK_TRANSLATE_PROMPT = """\
You will be given a user message written in English that describes a postpartum or newborn emergency situation.

Step 1: Translate the message to {language}.
Step 2: Translate it back to English.

Return ONLY the final English text — no labels, no explanation, no quotes.
The result should read naturally as a first-person user message.

Original message:
{text}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_red_examples(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("severity") == "red":
                examples.append(ex)
    return examples


def back_translate(model, text: str, language: str, retries: int = 3) -> str | None:
    prompt = BACK_TRANSLATE_PROMPT.format(language=language, text=text)
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            result = response.text.strip()
            # Basic sanity: result should be non-empty and different from input
            if result and result.lower() != text.lower():
                return result
            print(f"  Result identical to input or empty, skipping.")
            return None
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return None


def validate_example(ex: dict) -> bool:
    required = {"instruction", "input", "output", "topic", "severity"}
    return (
        required.issubset(ex.keys())
        and ex.get("severity") == "red"
        and isinstance(ex.get("instruction"), str)
        and len(ex["instruction"].strip()) >= 10
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

    parser = argparse.ArgumentParser(description="Back-translate red-class examples via Gemini.")
    parser.add_argument("--input",      default="data/original/merged_splits/train.jsonl")
    parser.add_argument("--output",     default="data/augmented/back_translated_red.jsonl")
    parser.add_argument("--languages",  nargs="+",
                        default=["Spanish", "French", "German", "Turkish"],
                        help="Intermediate languages for back-translation")
    parser.add_argument("--model",      default="gemini-1.5-flash")
    parser.add_argument("--rpm",        type=int, default=15, help="Max requests per minute")
    args = parser.parse_args()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=args.model)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    red_examples = load_red_examples(input_path)
    print(f"Found {len(red_examples)} red examples in {input_path}")
    print(f"Languages: {args.languages}")
    total_target = len(red_examples) * len(args.languages)
    print(f"Target: {total_target} back-translated examples\n")

    # Track already-done (instruction, language) pairs to allow resuming
    done_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    done_keys.add((ex.get("_source_instruction", ""), ex.get("_language", "")))
        print(f"Resuming: {len(done_keys)} pairs already done.\n")

    delay = 60.0 / args.rpm
    total_written = len(done_keys)

    with open(output_path, "a") as out_f:
        for i, source in enumerate(red_examples):
            src_instruction = source["instruction"]

            for language in args.languages:
                key = (src_instruction, language)
                if key in done_keys:
                    continue

                print(f"[{total_written}/{total_target}] Example {i+1}/{len(red_examples)} · {language}...")
                translated = back_translate(model, src_instruction, language)

                if translated:
                    new_example = {
                        "instruction": translated,
                        "input":       source.get("input", ""),
                        "output":      source.get("output", ""),
                        "topic":       source.get("topic", ""),
                        "severity":    "red",
                        # Internal tracking fields (stripped before training if needed)
                        "_source_instruction": src_instruction,
                        "_language": language,
                    }
                    if validate_example(new_example):
                        out_f.write(json.dumps(new_example) + "\n")
                        out_f.flush()
                        total_written += 1
                        done_keys.add(key)
                        print(f"  OK: {translated[:80]}{'...' if len(translated) > 80 else ''}")
                    else:
                        print(f"  Validation failed, skipping.")
                else:
                    print(f"  Back-translation failed, skipping.")

                time.sleep(delay)

    print(f"\nDone. {total_written} back-translated examples saved to {output_path}")
    print("Note: strip '_source_instruction' and '_language' fields before training.")


if __name__ == "__main__":
    main()
