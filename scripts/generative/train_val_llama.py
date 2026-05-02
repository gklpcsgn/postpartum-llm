import math
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

DATA_DIR = "/home/gokalp/research/postpartum-llm/data/datasets/merged_splits"
TRAIN_FILE = f"{DATA_DIR}/train.jsonl"
VAL_FILE = f"{DATA_DIR}/val.jsonl"

OUT = "outputs/llama31_8b_postpartum_qlora"

MAX_LEN = 1024
IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a concise, factual postpartum and newborn-care assistant. "
    "Use only the provided information from the books. "
    "Answer clearly in a few sentences, without repeating the same sentence "
    "or adding extra filler."
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Ensure EOS exists
if tokenizer.eos_token is None:
    tokenizer.eos_token = "</s>"

print("Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading base model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Adding LoRA adapters...")
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# Disable cache during training
model.config.use_cache = False

print("Loading dataset (train + val)...")
ds = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)


def format_example(example):
    instr = example.get("instruction", "") or ""
    inp = example.get("input", "") or ""
    out = example.get("output", "") or ""

    if inp:
        user_block = f"Instruction: {instr}\nInput: {inp}\nResponse:"
    else:
        user_block = f"Instruction: {instr}\nResponse:"

    prompt = SYSTEM_PROMPT + "\n\n" + user_block
    answer = " " + out + tokenizer.eos_token

    return {
        "prompt": prompt,
        "answer": answer,
    }


print("Formatting dataset...")
for split in ["train", "validation"]:
    ds[split] = ds[split].map(format_example)


def tokenize_function(batch):
    prompts = batch["prompt"]
    answers = batch["answer"]

    # Full text = prompt + answer
    texts = [p + a for p, a in zip(prompts, answers)]

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

    # Token-level prompt lengths for masking
    prompt_lens = []
    for p in prompts:
        enc_p = tokenizer(
            p,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=False,
        )
        prompt_lens.append(len(enc_p["input_ids"]))

    enc["prompt_len"] = prompt_lens
    return enc


print("Tokenizing dataset...")
for split in ["train", "validation"]:
    ds[split] = ds[split].map(
        tokenize_function,
        batched=True,
        remove_columns=ds[split].column_names,  # drop original fields; keep enc outputs
    )


class SupervisedCollator:
    """
    Mask prompt tokens in labels so loss is only on answer + EOS.
    """
    def __init__(self, tokenizer, ignore_index=IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        prompt_lens = [ex["prompt_len"] for ex in batch]

        labels = input_ids.clone()
        # mask prompt tokens
        for i, p_len in enumerate(prompt_lens):
            labels[i, :p_len] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


data_collator = SupervisedCollator(tokenizer=tokenizer, ignore_index=IGNORE_INDEX)

training_args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

print("Evaluating on validation set...")
eval_metrics = trainer.evaluate()
val_loss = eval_metrics["eval_loss"]
val_ppl = math.exp(val_loss)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation perplexity: {val_ppl:.2f}")

trainer.save_model(OUT)
print("Training done. Saved to", OUT)
