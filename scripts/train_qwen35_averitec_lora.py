import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = os.getenv("FT_MODEL_NAME", "Qwen/Qwen3.5-4B")
DATA_PATH = os.getenv("FT_DATA_PATH", "/home/airs/homework/llm_fp/data/averitec/train_sft.jsonl")
OUTPUT_DIR = os.getenv("FT_OUTPUT_DIR", "/home/airs/homework/llm_fp/outputs/qwen35-averitec-lora")
MAX_LENGTH = int(os.getenv("FT_MAX_LENGTH", "1024"))


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def preprocess(tokenizer):
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def _tok(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    dataset = dataset.map(_tok, remove_columns=dataset.column_names)
    return dataset


def main():
    model, tokenizer = load_model_and_tokenizer()
    dataset = preprocess(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
