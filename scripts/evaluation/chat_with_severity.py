# scripts/chat_with_severity.py

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

#########################################
# 0. Prompt builder (same as evaluation)
#########################################

def build_prompt(instr: str, inp: str | None) -> str:
    """
    Must match training / evaluation format:

      Instruction: ...
      Input: ...
      Response:

    or without Input if empty.
    """
    inp = inp or ""
    if inp:
        prompt = f"Instruction: {instr}\nInput: {inp}\nResponse:"
    else:
        prompt = f"Instruction: {instr}\nResponse:"
    return prompt


#########################################
# 1. Severity + rules
#########################################

LABELS = ["green", "yellow", "red"]
IDX_RED = LABELS.index("red")

# threshold for treating a conversation as "red" based on p(red)
CONV_RED_THRESH = 0.20

MENTAL_HEALTH_PATTERNS = [
    "thoughts of hurting myself",
    "thought of hurting myself",
    "hurting myself",
    "harm myself",
    "ending my life",
    "end my life",
    "kill myself",
    "killing myself",
    "better off without me",
    "better off without us",
    "don't want to live",
    "do not want to live",
    "want to die",
    "want to disappear",
    "can't keep myself safe",
    "cannot keep myself safe",
    "thoughts of hurting my baby",
    "thought of hurting my baby",
    "hurting my baby",
    "harm my baby",
]


def rule_based_red(text: str) -> bool:
    """Hard-rule detection for clear self-harm / harm-to-baby language."""
    t = text.lower()
    for p in MENTAL_HEALTH_PATTERNS:
        if p in t:
            return True
    return False


def build_conversation_text(user_msg: str, draft: str) -> str:
    return f"User: {user_msg}\nDraft: {draft}"


def load_severity_model(path: Path, device: str):
    print(f"Loading severity model from: {path}")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    return tok, model


def predict_label_and_probs(tokenizer, model, text: str, device: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

    pred_id = int(torch.argmax(torch.tensor(probs)).item())
    label = LABELS[pred_id]
    return label, probs


#########################################
# 2. Main LM (fine-tuned causal model)
#########################################

def load_main_model(model_dir: Path, device: str):
    print(f"Loading main model from: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return tok, model


def generate_answer(
    lm_tokenizer,
    lm_model,
    instr: str,
    inp: str | None,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    prompt = build_prompt(instr, inp)
    inputs = lm_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_ids = lm_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy, matches your eval script
            pad_token_id=lm_tokenizer.eos_token_id,
        )

    full_text = lm_tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # Remove prompt echo if present
    if full_text.startswith(prompt):
        pred = full_text[len(prompt):]
    else:
        pred = full_text
    return pred.strip()


#########################################
# 3. Single-turn pipeline with overrides
#########################################

CRISIS_MESSAGE = (
    "This sounds potentially very serious. I’m not a replacement for urgent in-person care. "
    "Please contact your clinician right away or call your local emergency number. "
    "If you feel you might act on these thoughts or cannot stay safe, seek emergency help immediately."
)


def chat_turn(
    user_msg: str,
    lm_tokenizer,
    lm_model,
    tok_conv,
    sev_model_conv,
    device: str,
    max_new_tokens: int = 256,
):
    instr = user_msg
    inp = ""

    # 1) Hard rules on user message — block before generating
    if rule_based_red(instr):
        return {
            "conv_severity": "red",
            "conv_p_red": 1.0,
            "answer": CRISIS_MESSAGE,
            "blocked_for_red_input": True,
            "overridden_for_red_output": False,
        }

    # 2) Generate draft response
    answer = generate_answer(
        lm_tokenizer=lm_tokenizer,
        lm_model=lm_model,
        instr=instr,
        inp=inp,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    # 3) Classify the full conversation (user message + draft)
    if rule_based_red(answer):
        conv_severity = "red"
        conv_p_red = 1.0
    else:
        conv_text = build_conversation_text(instr, answer)
        conv_label, conv_probs = predict_label_and_probs(
            tok_conv, sev_model_conv, conv_text, device
        )
        conv_p_red = conv_probs[IDX_RED]
        if conv_p_red >= CONV_RED_THRESH or conv_label == "red":
            conv_severity = "red"
        else:
            conv_severity = conv_label

    overridden_for_red_output = conv_severity == "red"
    if overridden_for_red_output:
        answer = CRISIS_MESSAGE

    return {
        "conv_severity": conv_severity,
        "conv_p_red": conv_p_red,
        "answer": answer,
        "blocked_for_red_input": False,
        "overridden_for_red_output": overridden_for_red_output,
    }


#########################################
# 4. Interactive loop
#########################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent.parent

    llm_dir = repo_root / "scripts" / "outputs" / "llama31_8b_postpartum_qlora"
    sev_conv_dir = repo_root / "models" / "classifiers" / "severity_conversation"

    lm_tokenizer, lm_model = load_main_model(llm_dir, device)
    tok_conv, sev_model_conv = load_severity_model(sev_conv_dir, device)

    print("Interactive postpartum assistant with severity guard")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_msg.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not user_msg:
            continue

        result = chat_turn(
            user_msg=user_msg,
            lm_tokenizer=lm_tokenizer,
            lm_model=lm_model,
            tok_conv=tok_conv,
            sev_model_conv=sev_model_conv,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        prefix = (
            f"[severity={result['conv_severity']} (p_red={result['conv_p_red']:.2f}), "
            f"blocked_in={result['blocked_for_red_input']}, "
            f"overridden={result['overridden_for_red_output']}]"
        )
        print(f"{prefix}\nAssistant: {result['answer']}\n")
