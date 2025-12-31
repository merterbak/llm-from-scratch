import os
import math
import time
from dataclasses import dataclass, asdict
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from model import Transformer, Config as ModelConfig
from prepare import ShardedDataset


@dataclass
class TrainingArguments:
    out_dir: str = "out"
    data_dir: str = "data/fineweb"
    seed: int = 42
    
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 6e-5
    max_iters: int = 100000
    eval_interval: int = 2000
    eval_iters: int = 200
    log_interval: int = 10
    
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    num_workers: int = 0
    shuffle: bool = True
    
    init_from: str = "scratch"
    resume_path: str = ""


def cosine_decay_lr(iteration, config):
    if not config.decay_lr:
        return config.learning_rate
    if iteration < config.warmup_iters:
        return config.learning_rate * (iteration + 1) / (config.warmup_iters + 1)
    if iteration >= config.lr_decay_iters:
        return config.min_lr
    ratio = (iteration - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


class Trainer:
    def __init__(self, train_config, model_config):
        self.config = train_config
        self.model_config = model_config
        
        self.ddp = "RANK" in os.environ
        if self.ddp:
            dist.init_process_group(backend="nccl")
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.master = self.rank == 0
        
        if train_config.device == "cuda":
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device(train_config.device)
        
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        self.dtype = dtype_map[train_config.dtype]
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
        
        if self.master:
            os.makedirs(train_config.out_dir, exist_ok=True)
        
        torch.manual_seed(train_config.seed + self.rank)
        
        self.iter_num = 0
        self.best_val_loss = float("inf")
        self.ema_loss = None
        
        self.raw_model = Transformer(model_config)
        
        if train_config.init_from == "resume":
            checkpoint_path = train_config.resume_path or os.path.join(train_config.out_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
                self.raw_model.load_state_dict(state)
                self.iter_num = checkpoint.get("iter_num", 0)
                self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                self._checkpoint_optim = checkpoint.get("optimizer", None)
                if self.master:
                    print(f"Resumed from {checkpoint_path} at iter {self.iter_num}")
            else:
                self._checkpoint_optim = None
        else:
            self._checkpoint_optim = None
        
        self.raw_model = self.raw_model.to(self.device)
        
        if train_config.compile:
            self.raw_model = torch.compile(self.raw_model)
        
        if self.ddp:
            self.model = DDP(self.raw_model, device_ids=[self.local_rank])
        else:
            self.model = self.raw_model
        
        decay, no_decay = [], []
        for p in self.raw_model.parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        
        if self.master:
            print(f"Params: {sum(p.numel() for p in decay):,} decay, {sum(p.numel() for p in no_decay):,} no-decay")
        
        use_fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames and self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": train_config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=train_config.learning_rate,
            betas=(train_config.beta1, train_config.beta2),
            fused=use_fused,
        )
        
        if self._checkpoint_optim:
            self.optimizer.load_state_dict(self._checkpoint_optim)
        
        self.scaler = torch.amp.GradScaler("cuda", enabled=(train_config.dtype == "float16"))
        
        train_dataset = ShardedDataset(
            train_config.data_dir, "train", model_config.block_size,
            rank=self.rank, world_size=self.world_size
        )
 
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=train_config.shuffle,
            num_workers=train_config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        val_dataset = ShardedDataset(
            train_config.data_dir, "val", model_config.block_size,
            rank=self.rank, world_size=self.world_size
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        def cycle(loader):
            while True:
                for batch in loader:
                    yield batch
        
        self.train_iter = cycle(train_loader)
        self.val_iter = cycle(val_loader)
        
        self.grad_accum = train_config.gradient_accumulation_steps // self.world_size
    
    def get_batch(self, split):
        x, y = next(self.train_iter if split == "train" else self.val_iter)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(self.config.eval_iters):
                x, y = self.get_batch(split)
                if self.device.type == "cpu":
                    _, loss = self.model(x, labels=y)
                else:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                        _, loss = self.model(x, labels=y)
                losses.append(loss.detach().float())
            loss_mean = torch.stack(losses).mean()
            if self.ddp:
                dist.all_reduce(loss_mean)
                loss_mean /= self.world_size
            out[split] = loss_mean.item()
        self.model.train()
        return out
    
    def save(self, val_loss):
        if not self.master:
            return
        checkpoint = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": val_loss,
            "model_args": asdict(self.model_config),
        }
        path = os.path.join(self.config.out_dir, "checkpoint.pt")
        tmp = path + ".tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, path)
        print(f"Saved checkpoint to {path}")
    
    def train(self):
        if self.master:
            print(f"Training {self.config.max_iters} iters (bs={self.config.batch_size}, accum={self.config.gradient_accumulation_steps}, world={self.world_size})")
            x, y = self.get_batch("train")
            print(f"Batch: x={x.shape}, y={y.shape}")
            if self.config.compile:
                print("Compiling model...")
        else:
            x, y = self.get_batch("train")
        
        t0 = time.time()
        
        while self.iter_num < self.config.max_iters:
            if self.iter_num > 0 and self.iter_num % self.config.eval_interval == 0:
                if self.master:
                    print(f"Eval at iter {self.iter_num}...")
                losses = self.evaluate()
                if self.master:
                    print(f"iter {self.iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
                self.save(losses["val"])
            
            lr = cosine_decay_lr(self.iter_num, self.config)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            
            loss_accum = 0.0
            for micro in range(self.grad_accum):
                x, y = self.get_batch("train")
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro == self.grad_accum - 1)
                if self.device.type == "cpu":
                    _, loss = self.model(x, labels=y)
                    loss = loss / self.grad_accum
                else:
                    with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                        _, loss = self.model(x, labels=y)
                        loss = loss / self.grad_accum
                loss_accum += loss.detach()
                self.scaler.scale(loss).backward()
            
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.iter_num % self.config.log_interval == 0 and self.master:
                dt = time.time() - t0
                t0 = time.time()
                
                tok_per_iter = self.config.batch_size * self.model_config.block_size * self.grad_accum * self.world_size
                tok_per_sec = (tok_per_iter * self.config.log_interval) / max(dt, 1e-8)
                loss_val = loss_accum.item()
                
                if self.ema_loss is None:
                    self.ema_loss = loss_val
                else:
                    self.ema_loss = 0.95 * self.ema_loss + 0.05 * loss_val
                
                print(f"iter {self.iter_num}: loss {loss_val:.4f} (ema {self.ema_loss:.4f}), {tok_per_sec:.0f} tok/s, lr {lr:.6e}")
            
            self.iter_num += 1
        
        if self.ddp:
            dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="Train model")
    defaults = TrainingArguments()
    
    p.add_argument("--batch_size", type=int, default=defaults.batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=defaults.gradient_accumulation_steps)
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    p.add_argument("--init_from", type=str, default=defaults.init_from, choices=["scratch", "resume"])
    p.add_argument("--resume_path", type=str, default=defaults.resume_path)
    
    args = p.parse_args()
    
    return TrainingArguments(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compile=not args.no_compile,
        init_from=args.init_from,
        resume_path=args.resume_path,
    ), ModelConfig()


if __name__ == "__main__":
    train_config, model_config = parse_args()
    trainer = Trainer(train_config, model_config)
    trainer.train()
