import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from lora import inject_lora


MODEL_NAME = "Qwen/Qwen2.5-7B"

MAX_LENGTH = 1024


def format_example(example):
    question = example["question"]
    answer = example["answer"]

    text = (
        "Question:\n"
        f"{question}\n\n"
        "Please solve the problem step by step. "
        "Put the final numerical answer at the end in the format #### number.\n\n"
        "Answer:\n"
        f"{answer}"
    )

    return {"text": text}


def tokenize(example, tokenizer):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    inject_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        r=16,
        alpha=32,
        dropout=0.05,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable} / {total}")

    dataset = load_dataset("openai/gsm8k", "main")
    train_dataset = dataset["train"].map(format_example)

    train_dataset = train_dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir="./qwen_lora_gsm8k",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./qwen_lora_gsm8k/final")
    tokenizer.save_pretrained("./qwen_lora_gsm8k/final")


if __name__ == "__main__":
    main()