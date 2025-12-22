import torch
import torch.nn.functional as F
from model import Transformer
from utils import DynamicCache
from prepare import load_tokenizer


checkpoint_path = "out/final.pt"
tokenizer_path = "spm.model"
prompt = "Once upon a time"
max_tokens = 100
temperature = 1.0
top_k = 50


device = "cuda" if torch.cuda.is_available() else "cpu"

sp = load_tokenizer(tokenizer_path)

print(f"Loading model: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint["config"]

model = Transformer(config)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()

@torch.no_grad()
def generate(prompt_text):
    prompt_ids = sp.encode(prompt_text, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    cache_len = config.sliding_window if getattr(config, "sliding_window", None) is not None else config.block_size
    cache = DynamicCache(max_cache_len=cache_len)
    
    for i in range(max_tokens):
        if cache.get_seq_length() > 0:
            x = input_ids[:, -1:]
        else:
            x = input_ids
        
        logits, _, cache = model(x, past_key_values=cache, use_cache=True)
        
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        top_probs, top_idx = torch.topk(probs, min(top_k, probs.size(-1)))
        top_probs = top_probs / top_probs.sum()
        next_token = top_idx.gather(-1, torch.multinomial(top_probs, 1))
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == sp.eos_id():
            break
    
    return sp.decode(input_ids[0].tolist())


print(f"\nPrompt: {prompt}")
print("-" * 30)
output = generate(prompt)
print(output)
print("-" * 30)
