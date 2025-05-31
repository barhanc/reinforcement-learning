import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

with open("prompts.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

dataset = Dataset.from_dict({"prompt": lines})

MODEL = "gpt2"
model_name = f"{MODEL}"
model_name = f"./{MODEL}-GRPO/checkpoint-336/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

for prompt in random.sample(lines, k=3):
    print(generator(prompt, max_new_tokens=64)[0]["generated_text"])
    print("-" * 80)
