import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
from utils import KVCache

# because of limited computing and data resources vocab_size is set to 32000(English only) and context(block) size is set to 4096

@dataclass
class Config:
    block_size: int = 4096
    vocab_size: int = 32000 
    n_layer: int = 28
    n_head: int = 16
    n_kv_head: int = 8
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = False 
    rope_theta: float = 10000.0
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    use_sdpa: bool = True
    max_cache: Optional[int] = None 
    kv_cache_offload: bool = False # it stores KV cache on CPU (slow but saves GPU VRAM)
    
class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon) * self.weight
        return x

class RoPEEmbedding(nn.Module):

    def __init__(self, config: Config, device=None):
        
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.rope_scaling_type = str(getattr(config, "rope_scaling_type", "none"))
        self.rope_scaling_factor = float(getattr(config, "rope_scaling_factor", 1.0))

        base = float(config.rope_theta)
        self.position_scale = 1.0
        self.attention_scaling = 1.0

        if self.rope_scaling_type == "none" or self.rope_scaling_factor == 1.0:
            pass
        elif self.rope_scaling_type == "yarn":
            self.position_scale = 1.0 / self.rope_scaling_factor
            self.attention_scaling = 0.1 * math.log(self.rope_scaling_factor) + 1.0
        elif self.rope_scaling_type in {"ntk"}:
            if self.head_dim > 2:
                base = base * (self.rope_scaling_factor ** (self.head_dim / (self.head_dim - 2.0)))
        else:
            raise ValueError(f"Unknown rope_scaling_type={self.rope_scaling_type!r}")

        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / float(self.head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        device = x.device
        dtype = x.dtype
        
        pos = position_ids.float().unsqueeze(-1) * self.position_scale
        inv_freq = self.inv_freq.to(device).unsqueeze(0).unsqueeze(0)
        freqs = pos * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = (emb.cos() * self.attention_scaling).to(dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype)
        return cos, sin


def rotate_half(x):
    
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):

    cos = cos.unsqueeze(unsqueeze_dim) 
    sin = sin.unsqueeze(unsqueeze_dim)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class GQA(nn.Module):

    def __init__(self, config: Config, layer_idx):

        super().__init__()
        self.layer_idx = int(layer_idx)
        self.n_head = config.n_head
        self.n_kv_head = int(getattr(config, "n_kv_head", config.n_head))
        self.n_embd = config.n_embd
        self.use_sdpa = bool(getattr(config, "use_sdpa", True))
        self.block_size = int(config.block_size)
        assert self.n_embd % self.n_head == 0
        assert 1 <= self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0

        head_dim = self.n_embd // self.n_head
        kv_dim = self.n_kv_head * head_dim

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.q_proj = nn.Linear(config.n_embd, self.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin, past_key_values=None, use_cache=False):

        B, T, C = x.shape
        head_dim = C // self.n_head
        q = self.q_proj(x).view(B, T, self.n_head, head_dim).transpose(1, 2)          
        k = self.k_proj(x).view(B, T, self.n_kv_head, head_dim).transpose(1, 2)       
        v = self.v_proj(x).view(B, T, self.n_kv_head, head_dim).transpose(1, 2)       
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length(self.layer_idx)
            k, v = past_key_values.update(k, v, self.layer_idx)
            past_len = int(k.size(2) - T)

        if self.n_kv_head != self.n_head:
            repeat_factor = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat_factor, dim=1)  
            v = v.repeat_interleave(repeat_factor, dim=1)  

        has_sdpa = hasattr(F, "scaled_dot_product_attention")
        if self.use_sdpa and has_sdpa:
            dropout_p = float(self.attn_dropout.p) if self.training else 0.0
            if past_key_values is None or past_len == 0:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
            else:
                Tk = int(k.size(2))
                i = torch.arange(T, device=x.device).unsqueeze(1)         
                j = torch.arange(Tk, device=x.device).unsqueeze(0)           
                upper = past_len + i
                allowed = j <= upper
                attn_mask = torch.zeros((1, 1, T, Tk), device=x.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(~allowed.view(1, 1, T, Tk), torch.finfo(q.dtype).min)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

            Tk = int(k.size(2))
            i = torch.arange(T, device=x.device).unsqueeze(1)               
            j = torch.arange(Tk, device=x.device).unsqueeze(0)             
            upper = past_len + i
            allowed = j <= upper
            att = att.masked_fill(~allowed.view(1, 1, T, Tk), torch.finfo(att.dtype).min)

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_dropout(y)
        return y

class SwiGLU(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.n_embd
        hidden_dim = int(4 * self.n_embd * 2 / 3) 
        hidden_dim = (hidden_dim + 255) // 256 * 256  

        self.gate_proj = nn.Linear(self.n_embd, hidden_dim, bias=config.bias)
        self.up_proj   = nn.Linear(self.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, self.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = self.down_proj(F.silu(gate) * up)
        x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config: Config, layer_idx):
        super().__init__()
        self.input_norm = RMSNorm(config.n_embd, eps=1e-6)
        self.post_attn_norm = RMSNorm(config.n_embd, eps=1e-6)
        self.attn = GQA(config, layer_idx=layer_idx)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, past_key_values=None, use_cache=False):

        residual = x
        x = self.input_norm(x)
        x = self.attn(x, cos, sin, past_key_values=past_key_values, use_cache=use_cache)
        x = residual + x

        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([DecoderLayer(config, layer_idx=i) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=1e-6)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rope = RoPEEmbedding(config)
        self.init_model_weights()

    def init_model_weights(self):
        std = 0.02
        
        nn.init.normal_(self.wte.weight, mean=0.0, std=std)
        
        num_layers = float(self.config.n_layer)
        for layer_idx, layer in enumerate(self.layers):
            nn.init.normal_(layer.attn.q_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.attn.k_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.attn.v_proj.weight, mean=0.0, std=std)
            
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * num_layers))
            
            nn.init.normal_(layer.mlp.gate_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.mlp.up_proj.weight, mean=0.0, std=std)
            
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * num_layers))
        
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, position_ids=None, labels=None, past_key_values=None, use_cache=False):
        B, T = input_ids.shape
        x = self.wte(input_ids) 

        if position_ids is None:
            past_seen = past_key_values.seen_tokens if past_key_values is not None else 0
            position_ids = torch.arange(past_seen, past_seen + T, device=input_ids.device).unsqueeze(0).expand(B, T)

        cos, sin = self.rope(x, position_ids)

        if use_cache and past_key_values is None:
            past_key_values = KVCache(
                max_cache=self.config.max_cache or self.config.block_size,
                offload_to_cpu=self.config.kv_cache_offload,
            )

        for layer in self.layers:
            x = layer(x, cos, sin, past_key_values=past_key_values, use_cache=use_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)), labels[:, 1:].contiguous().view(-1))

        if use_cache:
            return logits, loss, past_key_values
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, eos_token_id=3):
        cache = KVCache(
            max_cache=self.config.max_cache or self.config.block_size,
            offload_to_cpu=self.config.kv_cache_offload,
        )
        
        for _ in range(max_new_tokens):
            idx_cond = input_ids if cache.get_seq_length() == 0 else input_ids[:, -1:]
            if idx_cond.size(1) > self.config.block_size:
                idx_cond = idx_cond[:, -self.config.block_size:]
            
            logits, _, cache = self(idx_cond, past_key_values=cache, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if (next_token == eos_token_id).all():
                break
        
        return input_ids