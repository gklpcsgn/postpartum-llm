import argparse
import math
import random

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

SYSTEM_PROMPT = (
    "You are a concise, factual postpartum and newborn-care assistant. "
    "Use only the provided information from the books. "
    "Answer clearly in a few sentences, without repeating the same sentence "
    "or adding extra filler."
)

MAX_LEN = 1024
IGNORE_INDEX = -100

# Fixed QLoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
TRAIN_BATCH_SIZE = 1
GRAD_ACC_STEPS = 4
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Tokenizer setup
# ---------------------------------------------------------------------------

def configure_tokenizer(tokenizer, model_name: str) -> None:
    family = model_name.lower()
    if "gemma" in family:
        # Gemma-2 has no default pad token
        tokenizer.pad_token = tokenizer.eos_token
    elif "phi" in family:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif "qwen" in family:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # llama and generic fallback
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def make_format_fn(tokenizer, use_chat_template: bool, supports_system_role: bool = True):
    """Return a single-example format function closed over tokenizer.

    supports_system_role=False (e.g. Gemma-2): system prompt is prepended to
    the user turn instead of using a separate system message, since Gemma's
    chat template raises an error on the system role.
    """

    def format_chat(example):
        instr = example.get("instruction", "") or ""
        inp = example.get("input", "") or ""
        out = example.get("output", "") or ""
        user_text = f"Instruction: {instr}\nInput: {inp}" if inp else f"Instruction: {instr}"
        if supports_system_role:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"},
            ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer = out + tokenizer.eos_token
        return {"prompt": prompt, "answer": answer}

    def format_plain(example):
        instr = example.get("instruction", "") or ""
        inp = example.get("input", "") or ""
        out = example.get("output", "") or ""
        if inp:
            user_block = f"Instruction: {instr}\nInput: {inp}\nResponse:"
        else:
            user_block = f"Instruction: {instr}\nResponse:"
        prompt = SYSTEM_PROMPT + "\n\n" + user_block
        answer = " " + out + tokenizer.eos_token
        return {"prompt": prompt, "answer": answer}

    return format_chat if use_chat_template else format_plain


def make_tokenize_fn(tokenizer, use_chat_template: bool, max_len: int = MAX_LEN):
    """Return a batched tokenize function.

    Chat-template prompts already contain the BOS token via apply_chat_template,
    so we skip adding special tokens when tokenizing the full text.
    Prompt length for masking is always computed without special tokens so the
    count stays consistent with what was prepended by the template.
    """
    add_special_tokens_full = not use_chat_template

    def tokenize_function(batch):
        prompts = batch["prompt"]
        answers = batch["answer"]
        texts = [p + a for p, a in zip(prompts, answers)]

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            add_special_tokens=add_special_tokens_full,
        )

        prompt_lens = []
        for p in prompts:
            enc_p = tokenizer(
                p,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            )
            prompt_lens.append(len(enc_p["input_ids"]))

        enc["prompt_len"] = prompt_lens
        return enc

    return tokenize_function


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class SupervisedCollator:
    """Mask prompt tokens in labels so loss is only on the answer + EOS."""

    def __init__(self, tokenizer, ignore_index: int = IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        prompt_lens = [ex["prompt_len"] for ex in batch]

        labels = input_ids.clone()
        for i, p_len in enumerate(prompt_lens):
            labels[i, :p_len] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with QLoRA.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_file", default="data/original/merged_splits/train.jsonl")
    parser.add_argument("--val_file", default="data/original/merged_splits/val.jsonl")
    parser.add_argument("--max_steps", type=int, default=None)  # for dry runs only
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_vram", action="store_true", default=False,
                        help="Enable memory-saving overrides for large models (70B+): "
                             "paged_adamw_8bit optimizer, max_grad_norm=0.3, seq_len=512. "
                             "Seq len 512 vs 1024 tradeoff: halves activation memory but "
                             "truncates ~15%% of postpartum QA examples.")
    args = parser.parse_args()

    is_dry_run = args.max_steps is not None

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    # Sequence length: 512 under --low_vram to halve activation/KV-cache memory.
    # Tradeoff: ~15% of training examples contain responses longer than 512 tokens
    # and will be truncated, losing tail content but preserving most signal.
    seq_len = 512 if args.low_vram else MAX_LEN

    # --- Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    configure_tokenizer(tokenizer, args.model_name)

    use_chat_template = tokenizer.chat_template is not None
    # Gemma-2's chat template raises on system role — fold system prompt into user turn
    supports_system_role = "gemma" not in args.model_name.lower()
    print(f"Chat template: {'yes' if use_chat_template else 'no — using plain ### format'}"
          f"{' (no system role — folded into user turn)' if not supports_system_role else ''}")

    # --- Model ---
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    print("Adding LoRA adapters...")
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False

    # --- Dataset ---
    print("Loading dataset...")
    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.val_file},
    )

    print("Formatting dataset...")
    format_fn = make_format_fn(tokenizer, use_chat_template, supports_system_role)
    for split in ["train", "validation"]:
        ds[split] = ds[split].map(format_fn)

    print(f"Tokenizing dataset (seq_len={seq_len})...")
    tokenize_fn = make_tokenize_fn(tokenizer, use_chat_template, max_len=seq_len)
    for split in ["train", "validation"]:
        ds[split] = ds[split].map(
            tokenize_fn,
            batched=True,
            remove_columns=ds[split].column_names,
        )

    data_collator = SupervisedCollator(tokenizer=tokenizer)

    # prepare_model_for_kbit_training enables gradient checkpointing by default;
    # --low_vram makes it explicit in TrainingArguments as well.
    if args.low_vram:
        print("low_vram mode: paged_adamw_8bit optimizer, max_grad_norm=0.3, seq_len=512")
        print(f"  batch_size={TRAIN_BATCH_SIZE} ✓  grad_acc={GRAD_ACC_STEPS} ✓  gradient_checkpointing=True ✓")
    low_vram_kwargs = dict(
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
    ) if args.low_vram else {}

    # --- Training args ---
    if is_dry_run:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=min(WARMUP_STEPS, args.max_steps),
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=args.max_steps,
            save_strategy="no",
            load_best_model_at_end=False,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
            **low_vram_kwargs,
        )
        callbacks = []
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            seed=args.seed,
            **low_vram_kwargs,
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]

    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    val_loss = eval_metrics["eval_loss"]
    val_ppl = math.exp(val_loss)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation perplexity: {val_ppl:.2f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. Saved adapter to", args.output_dir)


if __name__ == "__main__":
    main()
