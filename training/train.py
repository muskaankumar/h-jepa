"""
Training Loop
==============
Self-supervised JEPA training with:
  - Automatic checkpoint/resume (for HPC spot instance preemption)
  - W&B experiment tracking
  - Mixed precision (bfloat16 on A100)
  - Cosine LR schedule with warmup
  - Gradient clipping
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import math
import os
import time
import json
import argparse
from pathlib import Path
from typing import Optional

import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import PISTJEPA
from data.dataset import build_dataloaders


# ──────────────────────────────────────────────────────────
# LR schedule: linear warmup + cosine decay
# ──────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────

def save_checkpoint(state: dict, checkpoint_dir: Path, tag: str = "latest"):
    path = checkpoint_dir / f"ckpt_{tag}.pt"
    torch.save(state, path)
    # Also save a "best" copy when relevant
    print(f"  [ckpt] Saved {path}")


def load_checkpoint(checkpoint_dir: Path, device: torch.device) -> Optional[dict]:
    """Load the most recent checkpoint if it exists."""
    candidates = sorted(checkpoint_dir.glob("ckpt_*.pt"))
    if not candidates:
        return None
    path = candidates[-1]
    print(f"  [resume] Loading checkpoint from {path}")
    return torch.load(path, map_location=device)


# ──────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────

def train(cfg: argparse.Namespace):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if (device.type == "cuda" and cfg.bf16) else torch.float32

    # ── Build model ──────────────────────────────────────────────────
    model = PISTJEPA(
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        num_heads=cfg.num_heads,
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        use_delta=cfg.use_delta,
        min_mask_frames=cfg.min_mask_frames,
        max_mask_frames=cfg.max_mask_frames,
        use_ema=cfg.use_ema,
        ema_momentum_start=cfg.ema_momentum_start,
        ema_momentum_end=cfg.ema_momentum_end,
        use_grad_ckpt=cfg.grad_ckpt,
        pred_dim=cfg.pred_dim,
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Parameter counts: {param_counts}")
    assert param_counts["total"] < 100_000_000, \
        f"Model exceeds 100M param limit: {param_counts['total']:,}"

    # ── Optimizer ────────────────────────────────────────────────────
    # Weight decay only on non-bias, non-norm parameters
    decay_params  = [p for n, p in model.named_parameters()
                     if p.requires_grad and p.ndim >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95))

    scaler = GradScaler(enabled=(dtype == torch.float32 and device.type == "cuda"))

    # ── Data ─────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        cfg.data_root, cfg.batch_size, cfg.num_workers, cfg.stats_path
    )
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    if cfg.use_ema:
        model.ema.total_steps = total_steps

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.resume == "auto":
        ckpt = load_checkpoint(ckpt_dir, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"  [resume] Resuming from epoch {start_epoch}, step {global_step}")

    # ── W&B ──────────────────────────────────────────────────────────
    if cfg.wandb_project and not cfg.no_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config=vars(cfg),
            resume="allow",
            id=cfg.run_name,
        )

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            x = batch["frames"].to(device, dtype=dtype, non_blocking=True)  # (B,T,11,H,W)

            # LR schedule
            lr = get_lr(global_step, warmup_steps, total_steps, cfg.lr, cfg.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + loss
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                loss, metrics = model.forward_train(x)

            if dtype == torch.float32:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            # EMA update
            if cfg.use_ema:
                model.ema.update()

            epoch_loss += loss.item()
            global_step += 1

            # Logging
            if global_step % cfg.log_every == 0:
                log_dict = {
                    "train/loss": metrics["loss"],
                    "train/lr": lr,
                    "train/mask_frames": metrics["mask_frames"],
                    **{f"train/{k}": v for k, v in metrics.items()
                       if k.startswith("repr/")},
                }
                if cfg.wandb_project:
                    wandb.log(log_dict, step=global_step)
                print(
                    f"Epoch {epoch:3d} | Step {global_step:6d} | "
                    f"Loss {metrics['loss']:.4f} | LR {lr:.2e} | "
                    f"cos_sim {metrics.get('repr/mean_cos_sim', 0):.3f}"
                )

        # ── Validation ───────────────────────────────────────────────
        if epoch % cfg.val_every == 0:
            val_loss = evaluate(model, val_loader, device, dtype)
            print(f"  [val] Epoch {epoch} | Val loss: {val_loss:.4f}")
            if cfg.wandb_project:
                wandb.log({"val/loss": val_loss}, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {"model": model.state_dict(), "epoch": epoch,
                     "global_step": global_step,
                     "best_val_loss": best_val_loss,
                     "optimizer": optimizer.state_dict()},
                    ckpt_dir, tag="best"
                )

        # ── Periodic checkpoint (every epoch for spot safety) ────────
        save_checkpoint(
            {"model": model.state_dict(), "epoch": epoch,
             "global_step": global_step, "best_val_loss": best_val_loss,
             "optimizer": optimizer.state_dict()},
            ckpt_dir, tag="latest"
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch} done | avg loss {epoch_loss/len(train_loader):.4f} | {elapsed:.1f}s")

    if cfg.wandb_project:
        wandb.finish()


@torch.no_grad()
def evaluate(model, loader, device, dtype):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["frames"].to(device, dtype=dtype, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            loss, _ = model.forward_train(x)
        total_loss += loss.item()
        n += 1
        if n >= 50:  # limit val to 50 batches for speed
            break
    model.train()
    return total_loss / max(n, 1)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_root",       default="/scratch/$USER/data/active_matter")
    p.add_argument("--stats_path",      default=None)
    p.add_argument("--checkpoint_dir",  default="./checkpoints")
    p.add_argument("--resume",          default="auto")
    # Model
    p.add_argument("--patch_size",      type=int,   default=16)
    p.add_argument("--embed_dim",       type=int,   default=384)
    p.add_argument("--encoder_depth",   type=int,   default=6)
    p.add_argument("--num_heads",       type=int,   default=6)
    p.add_argument("--pred_dim",        type=int,   default=192)
    p.add_argument("--num_frames",      type=int,   default=16)
    p.add_argument("--img_size",        type=int,   default=224)
    p.add_argument("--use_delta",       action="store_true", default=True)
    p.add_argument("--no_delta",        dest="use_delta", action="store_false")
    p.add_argument("--min_mask_frames", type=int,   default=4)
    p.add_argument("--max_mask_frames", type=int,   default=8)
    p.add_argument("--use_ema",         action="store_true", default=True)
    p.add_argument("--ema_momentum_start", type=float, default=0.996)
    p.add_argument("--ema_momentum_end",   type=float, default=0.9999)
    p.add_argument("--grad_ckpt",       action="store_true", default=True)
    # Training
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1.5e-4)
    p.add_argument("--min_lr",          type=float, default=1e-6)
    p.add_argument("--weight_decay",    type=float, default=0.04)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--num_workers",     type=int,   default=8)
    p.add_argument("--bf16",            action="store_true", default=True)
    p.add_argument("--seed",            type=int,   default=42)
    # Logging
    p.add_argument("--wandb_project",   default="pist_jepa_nyu")
    p.add_argument("--no_wandb", action="store_true", default=False)
    p.add_argument("--run_name",        default="pist_jepa_v1")
    p.add_argument("--log_every",       type=int,   default=50)
    p.add_argument("--val_every",       type=int,   default=5)

    cfg = p.parse_args()
    train(cfg)
