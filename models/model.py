"""
PIST-JEPA: Full Model
======================
Assembles all components into a single trainable module.

Forward pass (training):
  1. Tokenize with channel-group tokenizer (+ delta tokens)
  2. Concatenate main + delta tokens
  3. Apply causal mask: split into context / target
  4. Online encoder processes context tokens
  5. EMA encoder processes all tokens (no grad)
  6. Predictor maps context repr → predicted target repr
  7. JEPA loss: L2(predicted_target, ema_target)

Forward pass (evaluation / probing):
  1. Tokenize (no masking)
  2. Encoder processes ALL tokens
  3. Hierarchical pool: global mean + local spatial variance
  4. Return pooled representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict

from .tokenizer import ChannelGroupTokenizer
from .encoder import SpatiotemporalEncoder
from .jepa import (
    JEPAPredictor, EMAEncoder,
    make_causal_mask, sample_mask_frames,
    jepa_loss, check_collapse,
)


class PISTJEPA(nn.Module):
    """
    Physically-Informed Spatiotemporal Tokenizer with Hierarchical JEPA.

    Args:
        patch_size      : spatial patch size (default 16)
        embed_dim       : token/encoder dimension (default 384)
        encoder_depth   : number of transformer blocks (default 6)
        num_heads       : attention heads (default 6)
        num_frames      : temporal sequence length (default 16)
        img_size        : spatial resolution (default 224)
        use_delta       : prepend temporal difference tokens (default True)
        min_mask_frames : minimum frames to mask for JEPA (default 4)
        max_mask_frames : maximum frames to mask for JEPA (default 8)
        use_ema         : use EMA target encoder (recommended True)
        ema_momentum    : EMA starting momentum
        use_grad_ckpt   : gradient checkpointing to save VRAM
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_frames: int = 16,
        img_size: int = 224,
        use_delta: bool = True,
        min_mask_frames: int = 4,
        max_mask_frames: int = 8,
        use_ema: bool = True,
        ema_momentum_start: float = 0.996,
        ema_momentum_end: float = 0.9999,
        ema_total_steps: int = 100_000,
        use_grad_ckpt: bool = True,
        pred_dim: int = 192,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.min_mask_frames = min_mask_frames
        self.max_mask_frames = max_mask_frames
        self.use_ema = use_ema
        self.use_delta = use_delta
        self.num_patches_spatial = (img_size // patch_size) ** 2

        # ── Tokenizer ─────────────────────────────────────────────────
        self.tokenizer = ChannelGroupTokenizer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_frames=num_frames,
            img_size=img_size,
            use_delta=use_delta,
        )

        # ── Online encoder ────────────────────────────────────────────
        self.encoder = SpatiotemporalEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_grad_ckpt=use_grad_ckpt,
        )

        # ── Predictor ─────────────────────────────────────────────────
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            pred_dim=pred_dim,
        )

        # ── EMA target encoder ────────────────────────────────────────
        if use_ema:
            self.ema = EMAEncoder(
                self.encoder,
                momentum_start=ema_momentum_start,
                momentum_end=ema_momentum_end,
                total_steps=ema_total_steps,
            )
            self.ema.set_online(self.encoder)

        # Projection for aligning predictor output dim back to embed_dim
        # (no-op here since pred outputs embed_dim, but useful for ablations)
        self.pred_head = nn.Identity()

    # ─────────────────────────────────────────────────────────────────
    # Training forward
    # ─────────────────────────────────────────────────────────────────
    def forward_train(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        x : (B, T, C, H, W)  C=11
        Returns (loss, metrics_dict)
        """
        B, T, C, H, W = x.shape
        device = x.device
        N_sp = self.num_patches_spatial

        # 1. Tokenize
        tokens, delta_tokens = self.tokenizer(x)  # (B, T*N, D), (B,(T-1)*N,D)|None

        # 2. Concatenate delta tokens if present
        if delta_tokens is not None:
            all_tokens = torch.cat([tokens, delta_tokens], dim=1)
        else:
            all_tokens = tokens

        # Sequence length for main tokens only (we mask on main tokens)
        N_main = T * N_sp

        # 3. Sample causal mask
        mask_frames = sample_mask_frames(self.min_mask_frames, self.max_mask_frames)
        ctx_idx, tgt_idx = make_causal_mask(T, N_sp, mask_frames, device)

        # Context tokens: first (T - mask_frames) frames from main token stream
        # Delta tokens appended in full to context (they don't require prediction)
        ctx_main = all_tokens[:, ctx_idx]         # (B, N_ctx, D)
        if delta_tokens is not None:
            ctx_tokens = torch.cat([ctx_main, all_tokens[:, N_main:]], dim=1)
        else:
            ctx_tokens = ctx_main

        # Target tokens: last mask_frames from main token stream
        tgt_tokens_main = all_tokens[:, tgt_idx]  # (B, N_tgt, D)

        # 4. Online encoder on context
        ctx_repr = self.encoder(ctx_tokens)        # (B, N_ctx+delta, D)
        ctx_repr_main = ctx_repr[:, :ctx_main.shape[1]]  # only main ctx part

        # 5. EMA encoder on all tokens (no grad, stable targets)
        if self.use_ema:
            with torch.no_grad():
                all_repr_ema = self.ema(all_tokens)  # (B, N_all, D)
                tgt_repr_ema = all_repr_ema[:, tgt_idx]  # (B, N_tgt, D)
        else:
            # Fallback: use online encoder with stop-gradient
            with torch.no_grad():
                all_repr_sg = self.encoder(all_tokens)
                tgt_repr_ema = all_repr_sg[:, tgt_idx].detach()

        # 6. Predictor on context representations
        pred_out = self.predictor(ctx_repr_main)   # (B, N_ctx_main, D)

        # We predict the LAST N_tgt tokens from LAST N_tgt context positions
        # (predictor is causal: last context token predicts first target)
        N_tgt = tgt_idx.shape[0]
        pred_tgt = pred_out[:, -N_tgt:]            # (B, N_tgt, D)

        # 7. Loss
        loss = jepa_loss(pred_tgt, tgt_repr_ema, normalize=True)

        # Collapse diagnostics
        collapse_metrics = check_collapse(ctx_repr_main)

        metrics = {
            "loss": loss.item(),
            "mask_frames": mask_frames,
            **collapse_metrics,
        }

        return loss, metrics

    # ─────────────────────────────────────────────────────────────────
    # Representation extraction (frozen, for probing)
    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass without masking. Returns hierarchically pooled repr.
        x : (B, T, C, H, W)
        returns : (B, D_repr)  where D_repr = embed_dim * 2 (mean + var)
        """
        tokens, delta_tokens = self.tokenizer(x)

        if delta_tokens is not None:
            all_tokens = torch.cat([tokens, delta_tokens], dim=1)
        else:
            all_tokens = tokens

        repr_tokens = self.encoder(all_tokens)  # (B, N_total, D)

        # Hierarchical pooling: global mean + local spatial variance
        return self.hierarchical_pool(repr_tokens, x.shape[1])

    def hierarchical_pool(
        self,
        tokens: torch.Tensor,    # (B, N_total, D)
        T: int,
    ) -> torch.Tensor:
        """
        Two-level pooling that captures both global mean and local variance.

        Level 1 (global mean): average all tokens → captures mean state
        Level 2 (spatial variance): compute per-frame spatial variance,
                then average over frames → captures spatial heterogeneity
                (critical for detecting isotropic vs nematic transition)

        Returns: (B, 2*D) concatenation
        """
        B = tokens.shape[0]
        N_sp = self.num_patches_spatial
        N_main = T * N_sp

        main_tokens = tokens[:, :N_main]  # (B, T*N_sp, D)

        # Global mean pool
        global_mean = main_tokens.mean(dim=1)  # (B, D)

        # Per-frame spatial variance, then mean over frames
        frames = rearrange(main_tokens, "b (t n) d -> b t n d", t=T, n=N_sp)
        spatial_var = frames.var(dim=2)      # (B, T, D) — variance over spatial patches
        mean_var = spatial_var.mean(dim=1)   # (B, D) — mean over time

        return torch.cat([global_mean, mean_var], dim=-1)  # (B, 2*D)

    @property
    def repr_dim(self) -> int:
        """Dimension of the representation vector output by encode()."""
        return self.embed_dim * 2

    # ─────────────────────────────────────────────────────────────────
    # Standard forward — dispatches based on training mode
    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        if self.training:
            return self.forward_train(x)
        else:
            return self.encode(x)

    def count_parameters(self) -> dict:
        """Returns parameter counts per submodule."""
        def count(m): return sum(p.numel() for p in m.parameters())
        d = {
            "tokenizer":  count(self.tokenizer),
            "encoder":    count(self.encoder),
            "predictor":  count(self.predictor),
        }
        d["total"] = sum(d.values())
        return d
