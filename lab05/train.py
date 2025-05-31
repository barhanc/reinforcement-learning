from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


dataset = load_dataset("text", data_files="prompts.txt", split="train")
dataset = dataset.rename_column("text", "prompt")


def reward(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        rewards.append(sum(len(word) == 4 for word in completion.split()))
    return rewards


MODEL = "gpt2"
training_args = GRPOConfig(
    output_dir=f"{MODEL}-GRPO",
    logging_steps=1,
    log_completions=True,
    report_to=["tensorboard"],
    learning_rate=1e-6,  # default: 1e-6
    gradient_accumulation_steps=2,  # default: 1
    max_completion_length=32,  # default: 256
    beta=0.2,  # default: 0.04
)

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
