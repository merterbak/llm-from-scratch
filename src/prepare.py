import os
import glob
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import sentencepiece as spm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class DataConfig:
    text_file: str = "data/train_text.txt"
    tokenizer_path: str = "model_checkpoint/tokenizer.model"
    output_dir: str = "data/fineweb"
    vocab_size: int = 32000
    max_docs: Optional[int] = None
    shard_size: int = 1_000_000_000  
    max_shards: int = 10
    chunksize: int = 256 



def train_tokenizer(input_file, model_prefix, vocab_size=32000):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


def load_tokenizer(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def prepare_text(output_file, max_docs):

    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    with open(output_file, "w", encoding="utf-8") as f:
        total = None
        if max_docs is not None:
            total = max_docs
        else:
            try:
                total = dataset.info.splits["train"].num_examples
            except Exception:
                total = None

        for i, example in enumerate(tqdm(dataset, total=total, desc="Downloading", unit="docs")):
            text = example.get("text", "")
            if text.strip():
                f.write(text + "\n")
            if max_docs is not None and i >= max_docs - 1:
                break
    if max_docs is None:
        print(f"Saved {i + 1} documents to {output_file}")
    else:
        print(f"Saved {max_docs} documents to {output_file}")


tokenizer = None

def init_worker(model_path):
    global tokenizer
    tokenizer = load_tokenizer(model_path)


def tokenize_doc(doc):
    text = doc.get("text", "")
    if not text.strip():
        return np.array([], dtype=np.uint16)
    tokens = tokenizer.encode(text, out_type=int, add_eos=True)
    return np.array(tokens, dtype=np.uint16)


def tokenize_data(tokenizer_path, output_dir, shard_size, max_shards, chunksize=256):
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    nprocs = max(1, os.cpu_count() // 2)
    ctx = mp.get_context()
    print(f"Tokenizing with {nprocs} worker processes (start_method={ctx.get_start_method()})")

    def open_shard(path, length):
        return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(length,))
    
    with ctx.Pool(nprocs, initializer=init_worker, initargs=(tokenizer_path,)) as pool:
        shard_idx = 0
        buffer_pos = 0
        pbar = None
        shard = None

        def start_new_shard():
            nonlocal shard, buffer_pos, pbar, shard_idx
            split = "val" if shard_idx == 0 else "train"
            shard_path = os.path.join(output_dir, f"{split}_{shard_idx:05d}.npy")
            shard = open_shard(shard_path, shard_size)
            buffer_pos = 0
            if pbar:
                pbar.close()
            pbar = tqdm(total=shard_size, desc=f"Shard {shard_idx}", unit="tok")
            return shard_path

        shard_path = start_new_shard()

        for tokens in pool.imap(tokenize_doc, dataset, chunksize=chunksize):
            if len(tokens) == 0:
                continue

            tok_pos = 0
            while tok_pos < len(tokens):
                space = shard_size - buffer_pos
                take = min(space, len(tokens) - tok_pos)
                shard[buffer_pos : buffer_pos + take] = tokens[tok_pos : tok_pos + take]
                buffer_pos += take
                tok_pos += take
                if pbar:
                    pbar.update(take)

                if buffer_pos >= shard_size:
                    del shard 
                    if pbar:
                        pbar.close()
                        pbar = None
                    print(f"Wrote {shard_path}")

                    shard_idx += 1
                    if max_shards and shard_idx >= max_shards:
                        return
                    shard_path = start_new_shard()

        if buffer_pos > 0:
            split = "val" if shard_idx == 0 else "train"
            shard_path = os.path.join(output_dir, f"{split}_{shard_idx:05d}.npy")
            final = open_shard(shard_path, buffer_pos)
            final[:] = shard[:buffer_pos]
            del final
            if pbar:
                pbar.close()
            print(f"Wrote {shard_path} ({buffer_pos:,} tokens)")



class ShardedDataset(Dataset):

    def __init__(self, data_dir, split="train", block_size=1024, rank=0, world_size=1):
        self.block_size = block_size
        pattern = os.path.join(data_dir, f"{split}_*.npy")
        shard_paths = sorted(glob.glob(pattern))
        
        if not shard_paths:
            raise ValueError(f"No shards found for split '{split}' in {data_dir}")

        self.shard_paths = shard_paths[rank::world_size]

        self.shards = [np.load(p, mmap_mode="r") for p in self.shard_paths]
        
        self.shard_lengths = [len(s) // block_size for s in self.shards]
        self.total_length = sum(self.shard_lengths)
        print(f"[Rank {rank}] Loaded {len(self.shards)} shards, {self.total_length:,} blocks (non-overlapping)")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        for shard_idx, length in enumerate(self.shard_lengths):
            if idx < length:
                break
            idx -= length
        
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        
        shard = self.shards[shard_idx]
        chunk = torch.from_numpy(shard[start_idx:end_idx].astype(np.int64))
        x = chunk
        y = chunk
        
        return x, y


if __name__ == "__main__":
    config = DataConfig()
    
    if not os.path.exists(config.tokenizer_path):
        print("Step 1: Downloading text")
        if not os.path.exists(config.text_file):
            prepare_text(config.text_file, config.max_docs)
        else:
            print(f"Text file exists {config.text_file}")

        print("Step 2: Training tokenizer")
        model_prefix = config.tokenizer_path.replace(".model", "")
        train_tokenizer(config.text_file, model_prefix, config.vocab_size)
    else:
        print(f"Tokenizer exists {config.tokenizer_path}")

    print("Step 3: Tokenizing data")
    tokenize_data(config.tokenizer_path, config.output_dir, config.shard_size, config.max_shards, config.chunksize)
    print("Data preparation completed!")

