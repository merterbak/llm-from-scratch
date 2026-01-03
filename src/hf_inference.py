from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading Seed-0.4B...")

tokenizer = AutoTokenizer.from_pretrained("merterbak/Seed-0.4B")

model = AutoModelForCausalLM.from_pretrained(
    "merterbak/Seed-0.4B",
    trust_remote_code=True,
    torch_dtype="auto"
)

prompt = "Climate change can affect"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\n Prompt: '{prompt}'")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    do_sample=True,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated: {generated_text}")
