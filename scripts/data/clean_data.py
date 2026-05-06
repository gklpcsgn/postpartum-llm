import os
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
client = OpenAI(api_key=api_key)


def build_messages(examples_batch: list[dict]):
    """
    Build messages for the chat completion call for CLEANING.
    We feed a JSONL block of existing examples and ask the model to clean them.
    """

    system_msg = {
        "role": "system",
        "content": (
            "You are cleaning and standardizing a medical Q&A training dataset about postpartum and newborn care.\n\n"
            "Each example is a JSON object with at least the fields:\n"
            '  - \"instruction\": the user question\n'
            '  - \"input\": usually an empty string\n'
            '  - \"output\": the assistant answer\n'
            '  - \"severity\": one of [\"red\", \"yellow\", \"green\"]\n'
            "There may be other metadata fields (e.g., topic, subtopic, source); keep them as-is unless they are clearly wrong.\n\n"
            "IMPORTANT CARDINALITY RULE:\n"
            "- For every input example, you must output EXACTLY ONE cleaned JSON object.\n"
            "- The ONLY time you may output fewer examples is when two or more input examples are TRUE duplicates.\n"
            "- TRUE duplicates mean: effectively the same instruction AND effectively the same answer.\n"
            "- Do NOT drop examples for any other reason.\n"
            "- Do NOT merge examples that are similar but not truly the same.\n\n"
            "Your tasks for EACH batch you receive:\n\n"
            "1) Remove or rewrite any references to source text.\n"
            "   - The instruction and output must NOT mention: 'in the passage above', 'in the text', "
            "'according to the excerpt', 'in this book', 'on this page', chapter names, or similar.\n"
            "   - Rewrite them into standalone clinical questions and answers that make sense without any book or excerpt.\n\n"
            "2) Fix severity labels when needed.\n"
            "   Use ONLY these three labels:\n"
            "   - red: Immediate danger; instructions to call emergency services (e.g., 911) or go to ER now.\n"
            "   - yellow: Needs medical attention soon (same day or within 24 hours) but not clearly 911-level.\n"
            "   - green: Routine care, education, reassurance, or non-urgent issues.\n"
            "   If severity is wrong, correct it. If missing and inferable, add it. If ambiguous, choose the least severe reasonable category.\n\n"
            "3) Deduplicate ONLY when examples are truly duplicates.\n"
            "   - If two or more examples have the same meaning in both instruction and output, output only ONE cleaned version.\n"
            "   - Otherwise, keep each example separate.\n"
            "   - Do NOT remove examples unless they are true duplicates.\n\n"
            "4) Keep answers clinically safe and conservative.\n"
            "   - Do NOT add new medical facts or recommendations.\n"
            "   - You may clarify wording slightly and remove references to the text, but do not change medical meaning.\n"
            "   - Maintain original intent and precautions unless clearly wrong.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY cleaned examples, one per line, as JSONL.\n"
            "Each line must be a valid JSON object with the same basic fields you received (instruction, input, output, severity, and any extra fields).\n"
            "Do NOT include any explanations, commentary, or extra text.\n"
            "The number of output lines must match the number of input examples, EXCEPT when duplicates are merged.\n"
        ),
    }


    # Build JSONL block of the input examples
    jsonl_lines = [json.dumps(ex, ensure_ascii=False) for ex in examples_batch]
    jsonl_block = "\n".join(jsonl_lines)

    user_msg = {
        "role": "user",
        "content": (
            "You are given a batch of existing training examples in JSONL format.\n"
            "Each line is ONE example.\n\n"
            "INPUT JSONL:\n"
            "```jsonl\n"
            f"{jsonl_block}\n"
            "```\n\n"
            "Now output ONLY the cleaned examples, one per line, as JSONL, with NO extra text."
        ),
    }

    return [system_msg, user_msg]


def clean_examples_for_batch(
    examples_batch: list[dict],
    model: str,
    error_fh=None,
    batch_index: int | None = None,
):
    """
    Call the ChatGPT API for a single batch of existing examples and
    return parsed cleaned JSONL examples.
    """
    print(f"  Batch size: {len(examples_batch)} examples")
    messages = build_messages(examples_batch)

    api_t0 = time.perf_counter()
    print("  Calling API...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    api_t1 = time.perf_counter()
    print(f"  API response received. API time: {api_t1 - api_t0:.3f} s")

    content = response.choices[0].message.content.strip()
    print(f"  Raw response length: {len(content)} characters")

    # We minimally require instruction/input/output/severity to keep the example.
    required_keys = {"instruction", "input", "output", "severity"}
    cleaned_examples = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        def log_error(reason: str):
            if error_fh is not None:
                record = {
                    "reason": reason,
                    "batch_index": batch_index,
                    "raw_line": line,
                }
                error_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            print(f"    Skipping malformed line: {line[:80]}...")
            log_error("json_decode_error")
            continue

        if not isinstance(ex, dict):
            print("    Skipping non-dict line")
            log_error("non_dict")
            continue

        if not required_keys.issubset(ex.keys()):
            print(f"    Skipping line with missing keys: {line[:80]}...")
            log_error("missing_keys")
            continue

        cleaned_examples.append(ex)

    print(f"  Parsed {len(cleaned_examples)} valid cleaned examples from this batch.")
    return cleaned_examples


def iter_jsonl_batches(jsonl_path: str, batch_size: int):
    """
    Yield batches of examples from a JSONL file.
    Each batch is a list[dict] with up to batch_size items.
    """
    batch = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                # If you want to log these separately, you can extend this.
                print(f"[WARN] Skipping invalid JSON line in source: {line[:80]}...")
                continue

            batch.append(ex)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def main(
    input_jsonl: str,
    output_jsonl: str,
    model: str,
    batch_size: int = 100,
    max_batches: int | None = None,
):
    """
    Walk through an existing JSONL dataset in batches,
    call the API to clean each batch, and write a consolidated cleaned JSONL.
    Also log malformed lines from model output into a separate file.
    """
    in_path = Path(input_jsonl)
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    error_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")

    # Count how many examples for logging
    n_source = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_source += 1
    print(f"Source file {input_jsonl} has {n_source} examples (non-empty lines).")

    processed_batches = 0
    total_cleaned = 0
    t_global_start = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as f_out, \
         error_path.open("w", encoding="utf-8") as f_err:

        for batch_index, batch in enumerate(
            iter_jsonl_batches(input_jsonl, batch_size=batch_size), start=1
        ):
            if max_batches is not None and processed_batches >= max_batches:
                break

            print(f"\n=== Processing batch {batch_index} (size {len(batch)}) ===")
            t_batch_start = time.perf_counter()

            cleaned = clean_examples_for_batch(
                batch,
                model=model,
                error_fh=f_err,
                batch_index=batch_index,
            )

            if not cleaned:
                print("  No valid cleaned examples parsed for this batch.")
            else:
                print(f"  Writing {len(cleaned)} cleaned examples to {out_path}...")
                for ex in cleaned:
                    line = json.dumps(ex, ensure_ascii=False)
                    f_out.write(line + "\n")
                f_out.flush()
                total_cleaned += len(cleaned)
                print("  Write + flush complete for this batch.")

            t_batch_end = time.perf_counter()
            print(f"=== Finished batch {batch_index} in {t_batch_end - t_batch_start:.3f} s ===")

            processed_batches += 1

    t_global_end = time.perf_counter()
    print(f"\nDone. Processed {processed_batches} batch(es) into {output_jsonl}")
    print(f"Total cleaned examples written: {total_cleaned}")
    print(f"Total wall-clock time: {t_global_end - t_global_start:.3f} s")
    print(f"Model output errors logged to {error_path}")


if __name__ == "__main__":
    main(
        input_jsonl="data/datasets/therapy_and_the_postpartum_women.jsonl",
        output_jsonl="data/datasets/therapy_and_the_postpartum_women_cleaned.jsonl",
        model="gpt-5-mini",
        batch_size=100,
        max_batches=None,     # set an integer to limit for testing
    )
