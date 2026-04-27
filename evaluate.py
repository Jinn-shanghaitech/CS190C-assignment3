import json
import re
import glob
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

from lora import inject_lora


BASE_MODEL = "Qwen/Qwen2.5-7B"
LORA_DIR = "./qwen_lora_gsm8k/final"
VAL_PATH = "./gsm8k_val.jsonl"
OUT_PATH = "./results.jsonl"


def parse_answer(text):
    if text is None:
        return None

    text = str(text).replace(",", "")

    # Prefer the official GSM8K format: #### number
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)

    # Fallback: use the last number in the output
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]

    return None


def normalize_answer(ans):
    if ans is None:
        return None

    ans = str(ans).strip().replace(",", "")

    match = re.search(r"-?\d+(?:\.\d+)?", ans)
    if match:
        return match.group(0)

    return ans


def is_correct(pred, gt):
    pred = normalize_answer(pred)
    gt = normalize_answer(gt)

    if pred is None or gt is None:
        return False

    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except Exception:
        return pred == gt


def get_ground_truth(example):
    if "answer" in example:
        return parse_answer(example["answer"])

    if "ground_truth" in example:
        return normalize_answer(example["ground_truth"])

    if "target" in example:
        return normalize_answer(example["target"])

    return None


def build_prompt(question):
    return (
        "Question:\n"
        f"{question}\n\n"
        "Please solve the problem step by step. "
        "Put the final numerical answer at the end in the format #### number.\n\n"
        "Answer:\n"
    )


def load_lora_from_safetensors(model, lora_dir):
    lora_state = {}

    shard_paths = glob.glob(f"{lora_dir}/*.safetensors")

    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No safetensors files found in {lora_dir}")

    for path in shard_paths:
        shard = load_file(path)
        for key, value in shard.items():
            if "lora_A" in key or "lora_B" in key:
                lora_state[key] = value

    if len(lora_state) == 0:
        raise ValueError("No LoRA weights found. Check whether training saved LoRA parameters.")

    model.load_state_dict(lora_state, strict=False)
    print(f"Loaded LoRA tensors: {len(lora_state)}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        LORA_DIR,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    inject_lora(
        model,
        target_modules=["q_proj", "v_proj"],
        r=16,
        alpha=32,
        dropout=0.05,
    )

    load_lora_from_safetensors(model, LORA_DIR)

    model.eval()

    with open(VAL_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    correct = 0
    total = 0

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for example in tqdm(data):
            question = example["question"]
            ground_truth = get_ground_truth(example)

            prompt = build_prompt(question)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_output = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            if full_output.startswith(prompt):
                model_output = full_output[len(prompt):].strip()
            else:
                model_output = full_output.strip()

            parsed_answer = parse_answer(model_output)
            correct_flag = is_correct(parsed_answer, ground_truth)

            if correct_flag:
                correct += 1
            total += 1

            record = {
                "question": question,
                "ground_truth": str(ground_truth),
                "model_output": model_output,
                "parsed_answer": parsed_answer,
                "is_correct": correct_flag,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = correct / total if total > 0 else 0.0

    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Saved results to {OUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()