import torch
from model import Config, Transformer
from prepare import load_tokenizer


checkpoint_path = "out/checkpoint.pt"
tokenizer_path = "tokenizer.model"
prompt = "Once upon a time"
max_tokens = 100
temperature = 1.0
top_k = 50


device = "cuda" if torch.cuda.is_available() else "cpu"

sp = load_tokenizer(tokenizer_path)

print(f"Loading model: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

config = Config(**checkpoint["model_args"])

model = Transformer(config)
model = model.to(device)

# It is for handling torch.compile prefix in state_dict keys
state_dict = checkpoint["model"]
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

@torch.no_grad()
def generate(prompt_text):
    prompt_ids = sp.encode(prompt_text, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    output_ids = model.generate(
        input_ids, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=sp.eos_id()
    )
    
    return sp.decode(output_ids[0].tolist())


print(f"\nPrompt: {prompt}")
print("-" * 50)
output = generate(prompt)
print(output)
print("-" * 50)
