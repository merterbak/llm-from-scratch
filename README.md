# LLM From Scratch


- A small sized 0.5B decoder only dense model with RoPE, GQA, KV cache. 
- Model config and structure inspired by one of best small sized model today, Qwen3-0.6B.
- Built for educational purposes and fast experimentation with modern LLM techniques.
- Improvements are on the way...

## Features 

### Architecture

- Decoder only dense Transformer with:
  - RoPE rotary embeddings (also YaRN and NTK)
  - GQA (separate KV heads) for better inference/memory tradeoffs
  - RMSNorm + SwiGLU
  - PyTorch manual attention (or FlashAttention)

### Data + Tokenization

- Download data via Hugging Face `datasets`, then write tokenized shards with custom data class
- SentencePiece BPE training

### Training

- Mixed precision (bfloat16/float16), gradient accumulation, and DDP
- Warmup Stable Decay + warmup, weight decay, and gradient clipping
- Eval + checkpoints


### Generation

- KV cache  (with max cache length + CPU offload)
- Sampling utilities


## Quick Start 

Note: For information about model checkpoint and tokenizer file check [here](model_checkpoint/README.md).

```bash
pip install -r requirements.txt
```

### 1) Prepare data + tokenizer

`prepare.py` does three things:
1) streams docs into `train_text.txt`
2) trains tokenizer model
3) tokenizes data into shards

```bash
python prepare.py
```

### 2) Train

```bash
python train.py --batch_size 16 --gradient_accumulation_steps 2
```

Supported commands (other parameters can change inside of file):
- `--batch_size`
- `--gradient_accumulation_steps`
- `--no-compile` (disables `torch.compile`
- `--resume_path PATH`
- `--init_from scratch`(or `resume` if path isn't changed)

### 3) Inference

 `inference.py` expects:
- `out/checkpoint.pt`
- `tokenizer.model`

Edit the top of `inference.py` to change `prompt`, sampling params, or paths.

```bash
python inference.py
```










