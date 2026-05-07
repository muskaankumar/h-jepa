"""
Spatiotemporal ViT Encoder
===========================
A lightweight Vision Transformer encoder adapted for spatiotemporal token sequences.

Design decisions:
  - No [CLS] token: we pool spatiotemporally for the downstream probe,
    so a class token would be wasted capacity.
  - Gradient checkpointing: essential for 3136-length sequences on A100 40GB.
  - Pre-norm (LayerNorm before attention): more stable for self-supervised training.
  - RoPE-style relative position bias is omitted for simplicity; the tokenizer
    already injects absolute 3D positional embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import math


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, attn_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, head_dim)
        q = q.transpose(1, 2)        # (B, H, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (uses flash attention if available)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        return self.proj(attn_out)


class FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SpatiotemporalEncoder(nn.Module):
    """
    ViT-S style encoder for spatiotemporal token sequences.

    Args:
        embed_dim    : token dimension (default 384 for ViT-S)
        depth        : number of transformer blocks (default 6)
        num_heads    : attention heads (default 6)
        mlp_ratio    : FFN hidden dim multiplier
        drop_rate    : path/ffn dropout
        use_grad_ckpt: gradient checkpointing (saves ~40% VRAM, ~20% slower)
    """
    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        use_grad_ckpt: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_grad_ckpt = use_grad_ckpt

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N_total, D)  — full token sequence (context + any delta tokens)
        returns : (B, N_total, D)  — encoded tokens, layer-normed
        """
        for blk in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.norm(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Alias kept for compatibility."""
        return self.forward(x)
