from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("text", data_files="prompts.txt", split="train")
dataset = dataset.rename_column("text", "prompt")


def reward(completions: list[str], **kwargs):
    return [
        (
            (sum(1 for word in completion.split() if len(word) == 4)) / len(completion.split())
            if len(completion.split()) > 0
            else 0
        )
        for completion in completions
    ]


model = "Qwen/Qwen2-0.5B-Instruct"
model = "gpt2"

training_args = GRPOConfig(
    output_dir=f"{model}-GRPO",
    logging_steps=1,
    log_completions=True,
    # --- Parameters
    max_completion_length=32,
    gradient_accumulation_steps=1,
    learning_rate=1.1e-4,
    beta=0.075,
    epsilon=0.2,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset,
)

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer)

with open("prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

print("\n=== Before Training ===")
for prompt in prompts[:3]:
    print(completion := pipe(prompt)[0]["generated_text"])
    print(f"% = {sum(1 for word in completion.split() if len(word) == 4) / len(completion.split()):.2%}")
    print("=" * 80)

trainer.train()

tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer)

print("\n=== After Training ===")
for prompt in prompts:
    print(completion := pipe(prompt)[0]["generated_text"])
    print(f"% = {sum(1 for word in completion.split() if len(word) == 4) / len(completion.split()):.2%}")
    print("=" * 80)
