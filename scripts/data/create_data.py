import os
import json
from pathlib import Path
import time


from openai import OpenAI

# Set your API key in the environment:
# export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key="")
def build_messages(chunk_text: str):
    """
    Build messages for the chat completion call.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are helping build a medical Q&A dataset for a postpartum and newborn-care assistant.\n\n"
            "You will receive a text excerpt from a parenting / medical book.\n"
            "Use ONLY this text to create high-quality training data.\n\n"
            "You must do TWO internal steps:\n\n"
            "STEP 1 — INTERNAL FACT OUTLINE (DO NOT OUTPUT)\n"
            "- Read the excerpt and mentally summarize it into a concise factual outline.\n"
            "- The outline must contain ONLY facts stated directly in the text.\n"
            "- Do NOT infer, embellish, generalize, or add outside medical knowledge.\n"
            "- You will NOT print this outline; you will only use it to ground your answers.\n\n"
            "NARRATIVE FILTERING WITHIN STEP 1:\n"
            "- When building the internal outline, ignore and exclude any content that is narrative, anecdotal, metaphorical, or case-based.\n"
            "- Treat as narrative any material that focuses on personal stories, specific clients, scenes, therapist reflections, dialogues (e.g., \"she said...\", \"I told her...\"), or symbolic/psychodynamic metaphors.\n"
            "- Do NOT carry narrative details, character names, quoted speech, or therapy-process descriptions into the factual outline.\n"
            "- Only retain generalizable, factual, clinical, or symptom-pattern information that applies broadly (e.g., definitions, symptom lists, risk factors, red flags, typical trajectories, recommended actions described in the text).\n"
            "- If an excerpt contains only narrative material and no generalizable clinical facts, you may output ZERO examples for that excerpt instead of forcing questions.\n\n"
            
            "STEP 2 — GENERATE 10–20 JSONL EXAMPLES (OUTPUT ONLY)\n"
            "- Produce between 10 and 20 examples.\n"
            "- Each example is a SINGLE LINE valid JSON object of this exact form:\n\n"
            '  {"instruction":"...","input":"","output":"...","topic":"...","subtopic":"...","severity":"green|yellow|red"}\n\n'
            "- Output ONLY these JSON lines, nothing else. No prose, no comments, no arrays, no surrounding JSON.\n\n"

            "INSTRUCTION FIELD (\"instruction\"):\n"
            "- Write natural questions as if from a parent or caregiver.\n"
            "- Prefer common, broadly relevant concerns reflected in the excerpt.\n"
            "- Use everyday language unless the excerpt itself is highly medical.\n"
            "- Each question MUST be fully answerable from the factual outline alone.\n"
            "- You may include questions that express worry, fear, guilt, or uncertainty; they should sound emotionally natural, not theatrical.\n\n"
            "OUTPUT FIELD (\"output\"):\n"
            "- Answer as the assistant, speaking with direct knowledge.\n"
            "- NEVER reference the text, outline, pages, or “the excerpt.”\n"
            "- Stay STRICTLY within the facts from the outline.\n"
            "- Use simple, calm, supportive language.\n"
            "- If the question shows worry or emotion:\n"
            "  - Include EXACTLY ONE brief emotional acknowledgment in the answer\n"
            "    (e.g., “It’s common to have questions like this.”,\n"
            "          “Many parents wonder about this.”,\n"
            "          “It makes sense to feel unsure about this.”)\n"
            "  - Keep that acknowledgment short; do not overdo reassurance.\n"
            "- If the question is purely factual, keep the tone neutral and informative.\n"
            "- Do NOT add new medical facts, probabilities, or diagnoses not clearly in the text.\n"
            "- Do NOT add generic safety disclaimers or “talk to your doctor” advice unless the text itself clearly recommends it.\n"
            "- Do not supply explanations unless they appear directly in the excerpt.\n\n"
            "- Do NOT include narrative elements such as patient stories, therapist self-reflection, dialogues, or metaphorical/psychodynamic interpretations in the answer.\n\n"
            "INPUT FIELD (\"input\"):\n"
            "- Always set \"input\" to the empty string: \"\".\n\n"
            "TOPIC / SUBTOPIC FIELDS:\n"
            "- \"topic\": a short high-level label (e.g., \"pediatrician\", \"telehealth\", \"immunizations\", \"feeding\", \"smoking\").\n"
            "- \"subtopic\": a more specific label (e.g., \"choosing pediatrician\", \"telehealth privacy\", \"hospital newborn exam\").\n"
            "- Use simple, consistent labels that could be reused across examples in the same excerpt.\n\n"
            "SEVERITY FIELD (\"severity\"):\n"
            "- Choose severity ONLY from what the excerpt conveys.\n"
            "- Use:\n"
            "  - \"green\": routine information, logistics, expectations, low concern, no implied risk.\n"
            "  - \"yellow\": situations requiring caution, non-emergency risks, or where the text suggests follow-up, monitoring, or professional input without urgency.\n"
            "  - \"red\": clearly serious dangers explicitly described in the text (e.g., severe harm, emergencies, high-risk warnings).\n"
            "- Do NOT mark \"red\" unless the text itself clearly indicates serious or urgent risk.\n\n"
            "GENERAL CONSTRAINTS:\n"
            "- Use ONLY information that can be justified by the excerpt.\n"
            "- Never invent numbers, timelines, or recommendations that are not given.\n"
            "- Do NOT mention “book,” “chapter,” “excerpt,” “author,” page numbers, or any source-related detail.\n"
            "- Ensure every line is valid JSON and can be parsed independently.\n"
            "- Do not break answers across multiple lines; each example must be a single-line JSON object.\n\n"
            "The user message will contain:\n\n"
            "TEXT EXCERPT:\n"
            "<excerpt here>\n"
        ),
    }

    user_msg = {
        "role": "user",
        "content": chunk_text,
    }

    return [system_msg, user_msg]

def generate_examples_for_chunk(
    chunk_text: str,
    model: str,
    error_fh=None,
    source_path: str | None = None,
):
    """
    Call the ChatGPT API for a single chunk and return parsed JSONL examples.
    Optionally log skipped lines to error_fh.
    """
    print(f"  Chunk text length: {len(chunk_text)} characters")
    messages = build_messages(chunk_text)

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

    required_keys = {"instruction", "input", "output", "topic", "subtopic", "severity"}
    examples = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        def log_error(reason: str):
            if error_fh is not None:
                record = {
                    "reason": reason,
                    "chunk_file": source_path,
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

        examples.append(ex)

    print(f"  Parsed {len(examples)} valid examples from this chunk.")
    # if 1 minute has not passed, wait to avoid rate limits
    # elapsed = time.perf_counter() - api_t0
    # if elapsed < 60:
    #     wait_time = 60 - elapsed
    #     print(f"  Waiting {wait_time:.1f} s to avoid rate limits...")
    #     time.sleep(wait_time)
    return examples


def iter_chunk_files(chunks_root: str):
    """
    Yield all .txt chunk files under chunks_root, sorted.
    """
    root = Path(chunks_root)
    for path in sorted(root.rglob("*.txt")):
        yield path


def main(
    chunks_root: str,
    output_jsonl: str,
    model: str,
    max_chunks: int | None = None,
):
    """
    Walk through chunk files, call the API, and write a consolidated JSONL dataset.
    Also log skipped / malformed lines into a separate file.
    """
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    error_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")


    # Collect all chunk files once so we can count and log
    chunk_files = list(iter_chunk_files(chunks_root))
    total_chunks = len(chunk_files)
    if max_chunks is not None:
        total_planned = min(total_chunks, max_chunks)
    else:
        total_planned = total_chunks

    print(f"Found {total_chunks} chunk file(s) under {chunks_root}.")
    print(f"Will process up to {total_planned} chunk(s).")

    processed = 0
    t_global_start = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as f_out, \
         error_path.open("a", encoding="utf-8") as f_err:

        for idx, chunk_file in enumerate(iter_chunk_files(chunks_root), start=1):
            if max_chunks is not None and processed >= max_chunks:
                break

            print(f"\n=== Processing chunk {idx}: {chunk_file} ===")
            t_chunk_start = time.perf_counter()

            with chunk_file.open("r", encoding="utf-8") as f_in:
                chunk_text = f_in.read().strip()

            if not chunk_text:
                print("  Chunk is empty, skipping.")
                continue

            examples = generate_examples_for_chunk(
                chunk_text,
                model=model,
                error_fh=f_err,
                source_path=str(chunk_file),
            )

            if not examples:
                print("  No valid examples parsed for this chunk.")
            else:
                print(f"  Writing {len(examples)} examples to {out_path}...")
                for ex in examples:
                    # Optional: include source file metadata
                    # ex["_source_file"] = str(chunk_file)
                    line = json.dumps(ex, ensure_ascii=False)
                    f_out.write(line + "\n")
                f_out.flush()
                print("  Write + flush complete for this chunk.")


            t_chunk_end = time.perf_counter()
            print(f"=== Finished chunk {idx} in {t_chunk_end - t_chunk_start:.3f} s ===")

            processed += 1

    t_global_end = time.perf_counter()
    print(f"\nDone. Processed {processed} chunk file(s) into {output_jsonl}")
    print(f"Total wall-clock time: {t_global_end - t_global_start:.3f} s")
    print(f"Skipped / malformed lines logged to {error_path}")




if __name__ == "__main__":
    main(
        chunks_root="books/Therapy and the Postpartum Women - PARTS/chunks",
        output_jsonl="data/datasets/therapy_and_the_postpartum_women.jsonl",
        model="gpt-5-mini",
        max_chunks=None,  # set an integer to limit for testing
    )