from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading Seed-0.5B...")

tokenizer = AutoTokenizer.from_pretrained("merterbak/Seed-0.5B")

model = AutoModelForCausalLM.from_pretrained(
    "merterbak/Seed-0.5B",
    trust_remote_code=True,
    dtype="auto"
)

prompt = "Climate change can affect"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\n Prompt: '{prompt}'")
outputs = model.generate(
    **inputs,
    temperature=0.3,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated: {generated_text}")

