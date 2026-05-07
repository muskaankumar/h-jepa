"""
Channel-Group Factorized Tokenizer
===================================
Splits the 11 physical channels into 4 groups based on physical role,
applies separate patch embedders per group, then fuses via cross-group attention.

Channel layout (active_matter, 11 total):
  Group A — Concentration scalar : channels [0]        (1 ch)
  Group B — Velocity vector      : channels [1, 2]     (2 ch)
  Group C — Orientation tensor   : channels [3,4,5,6]  (4 ch)  (symmetric 2x2)
  Group D — Strain-rate tensor   : channels [7,8,9,10] (4 ch)  (symmetric 2x2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Physical channel groups
CHANNEL_GROUPS = {
    "concentration": [0],
    "velocity":      [1, 2],
    "orientation":   [3, 4, 5, 6],
    "strain_rate":   [7, 8, 9, 10],
}


class PatchEmbedGroup(nn.Module):
    """
    Patch embedder for a single channel group.
    Uses a Conv2d patch stem so each group learns its own spatial statistics.
    """
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True
        )
        # Per-group layer norm before fusion
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C_group, H, W)
        returns : (B, N_patches, embed_dim)  where N = (H/P)*(W/P)
        """
        x = self.proj(x)                          # (B, D, H/P, W/P)
        x = rearrange(x, "b d h w -> b (h w) d")  # (B, N, D)
        return self.norm(x)


class CrossGroupFusion(nn.Module):
    """
    Fuses embeddings from the 4 channel groups via lightweight multi-head attention.
    Each group token attends to all other group tokens, then outputs are averaged.
    This replaces a naive concat+linear (which ignores physical grouping structure).

    Input:  list of 4 tensors, each (B, N, D)
    Output: (B, N, D) fused token sequence
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_groups: int = 4):
        super().__init__()
        self.num_groups = num_groups
        # Group-type positional embeddings (learned, one per group)
        self.group_type_embed = nn.Parameter(
            torch.zeros(num_groups, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.group_type_embed, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, group_tokens: list) -> torch.Tensor:
        """
        group_tokens: list of G tensors, each (B, N, D)
        returns: (B, N, D)
        """
        # Add group-type embeddings so the fusion knows which physical field is which
        tagged = [
            g + self.group_type_embed[i]
            for i, g in enumerate(group_tokens)
        ]
        # Stack along group axis: (B, G*N, D)
        stacked = torch.cat(tagged, dim=1)

        # Each group queries all others (cross-group attention)
        outputs = []
        N = group_tokens[0].shape[1]
        for i in range(self.num_groups):
            q = self.norm_q(tagged[i])
            kv = self.norm_kv(stacked)
            out, _ = self.attn(q, kv, kv)
            outputs.append(out)

        # Weighted mean fusion (equal weight — can be learned later)
        fused = torch.stack(outputs, dim=0).mean(dim=0)  # (B, N, D)
        fused = fused + self.ffn(fused)
        return fused


class ChannelGroupTokenizer(nn.Module):
    """
    Full tokenizer pipeline:
      1. Split 11 channels into 4 physical groups
      2. Per-group patch embedding
      3. Cross-group attention fusion
      4. Optional: inject temporal difference tokens

    Args:
        patch_size   : spatial patch size (default 16, giving 14x14=196 patches for 224x224)
        embed_dim    : output token dimension
        num_frames   : T (temporal dimension)
        img_size     : spatial size H=W
        use_delta    : if True, prepend temporal difference tokens (xt+1 - xt)
    """
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_frames: int = 16,
        img_size: int = 224,
        use_delta: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.use_delta = use_delta
        self.embed_dim = embed_dim

        num_patches_spatial = (img_size // patch_size) ** 2  # 196
        self.num_patches_spatial = num_patches_spatial

        # One patch embedder per group
        self.group_embedders = nn.ModuleDict({
            name: PatchEmbedGroup(len(chs), patch_size, embed_dim)
            for name, chs in CHANNEL_GROUPS.items()
        })

        # Cross-group fusion
        self.fusion = CrossGroupFusion(embed_dim, num_heads=4)

        # Learnable 3D positional embeddings: temporal × spatial
        # Shape: (1, T, N_spatial, D) — broadcast over batch
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, num_patches_spatial, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Separate pos embed for delta tokens (T-1 frames)
        if use_delta:
            self.delta_pos_embed = nn.Parameter(
                torch.zeros(1, num_frames - 1, num_patches_spatial, embed_dim)
            )
            nn.init.trunc_normal_(self.delta_pos_embed, std=0.02)
            # Delta tokens use same group tokenizers but on difference frames
            self.delta_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm = nn.LayerNorm(embed_dim)

    def _tokenize_frame_batch(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Tokenizes a batch of frames using per-group embedders + fusion.
        x : (B*T, 11, H, W)  — all frames flattened into batch dim
        returns : (B*T, N_spatial, D)
        """
        group_tokens = []
        for name, ch_indices in CHANNEL_GROUPS.items():
            group_x = x[:, ch_indices]                         # (B*T, C_g, H, W)
            tokens = self.group_embedders[name](group_x)       # (B*T, N, D)
            group_tokens.append(tokens)

        fused = self.fusion(group_tokens)  # (B*T, N, D)
        return fused

    def forward(self, x: torch.Tensor):
        """
        x : (B, T, C, H, W)  with C=11
        returns:
            tokens     : (B, T*N, D)  main token sequence (with pos embed)
            delta_tokens: (B, (T-1)*N, D) or None
        """
        B, T, C, H, W = x.shape
        assert C == 11, f"Expected 11 channels, got {C}"
        N = self.num_patches_spatial

        # ── Main tokens ──────────────────────────────────────────────
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        tokens = self._tokenize_frame_batch(x_flat)          # (B*T, N, D)
        tokens = rearrange(tokens, "(b t) n d -> b t n d", b=B, t=T)
        tokens = tokens + self.pos_embed[:, :T]              # broadcast pos embed
        tokens = rearrange(tokens, "b t n d -> b (t n) d")  # (B, T*N, D)

        delta_tokens = None
        # ── Temporal difference tokens ────────────────────────────────
        if self.use_delta:
            # Compute frame differences: (B, T-1, C, H, W)
            delta_x = x[:, 1:] - x[:, :-1]
            delta_flat = rearrange(delta_x, "b t c h w -> (b t) c h w")
            dt = self._tokenize_frame_batch(delta_flat)      # (B*(T-1), N, D)
            dt = rearrange(dt, "(b t) n d -> b t n d", b=B, t=T - 1)
            dt = dt + self.delta_pos_embed[:, : T - 1]
            dt = self.delta_proj(dt)
            delta_tokens = rearrange(dt, "b t n d -> b (t n) d")  # (B, (T-1)*N, D)

        return self.norm(tokens), (self.norm(delta_tokens) if delta_tokens is not None else None)
