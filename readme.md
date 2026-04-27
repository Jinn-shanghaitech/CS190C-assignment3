# Assignment 3: LoRA Fine-tuning Qwen2.5-7B on GSM8K

## Final Accuracy

Accuracy: 77.00% (77/100)

## Base Model

Qwen/Qwen2.5-7B

## Dataset

Training dataset: openai/gsm8k train split  
Validation dataset: gsm8k_val.jsonl

## LoRA Hyperparameters

- r: 16
- alpha: 32
- dropout: 0.05
- target modules: q_proj, v_proj

## Training Details

- epochs: 2
- per-device batch size: 1
- gradient accumulation steps: 16
- learning rate: 2e-4
- max sequence length: 1024
- dtype: bfloat16
- optimizer/scheduler: Hugging Face Trainer default
- hardware: GPU on guoair517cgu-0

## Output

The evaluation results are saved in `results.jsonl`.