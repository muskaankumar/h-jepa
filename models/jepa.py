"""
JEPA Predictor and Causal Masking
===================================
Implements the predictive objective for PIST-JEPA.

Key design: Shifted Causal Masking
  - Always mask the LAST K frames (not random)
  - Forces the model to encode temporal dynamics, not just spatial statistics
  - K is drawn uniformly from [min_mask_frames, max_mask_frames] each iteration
    to provide curriculum variety

The predictor is a shallow MLP that maps context tokens → predicted target latents.
Operating in latent space (not pixel space) is critical for avoiding representation
collapse onto low-level features.

EMA Target Encoder (optional but recommended):
  - A momentum-updated copy of the encoder provides stable targets
  - Prevents the trivial collapsed solution
  - Momentum schedule: starts low (0.996), ramps to (0.9999) over training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────
# Masking utilities
# ──────────────────────────────────────────────────────────

def make_causal_mask(
    num_frames: int,
    num_patches_spatial: int,
    mask_frames: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns boolean index tensors for context and target tokens.

    Context: first (num_frames - mask_frames) frames
    Target:  last  mask_frames frames

    Returns:
        context_idx : (N_context,) long tensor of token indices
        target_idx  : (N_target,)  long tensor of token indices
    """
    N = num_frames * num_patches_spatial
    N_ctx = (num_frames - mask_frames) * num_patches_spatial
    ctx_idx = torch.arange(N_ctx, device=device)
    tgt_idx = torch.arange(N_ctx, N, device=device)
    return ctx_idx, tgt_idx


def sample_mask_frames(
    min_frames: int = 4,
    max_frames: int = 8,
) -> int:
    """Uniform random number of frames to mask."""
    return torch.randint(min_frames, max_frames + 1, (1,)).item()


# ──────────────────────────────────────────────────────────
# Predictor
# ──────────────────────────────────────────────────────────

class JEPAPredictor(nn.Module):
    """
    Shallow MLP predictor that maps context token representations to
    predicted target token latents.

    Architecture: LayerNorm → Linear → GELU → Linear → LayerNorm
    Kept intentionally shallow (2 layers) so the encoder is forced to
    learn rich representations rather than offloading work to the predictor.

    Args:
        embed_dim    : token dimension (must match encoder output)
        pred_dim     : predictor hidden dim (default: embed_dim // 2)
        num_targets  : number of target tokens to predict (T_mask * N_spatial)
    """
    def __init__(
        self,
        embed_dim: int = 384,
        pred_dim: int = 192,
    ):
        super().__init__()
        self.norm_in = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, pred_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(pred_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

        # Learnable query tokens for each target position
        # (added to context mean to condition on temporal position)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,   # (B, N_ctx, D)
    ) -> torch.Tensor:                  # (B, N_ctx, D)
        """
        Predicts representations for all context positions.
        The loss is computed only on the target subset (via index selection outside).
        """
        x = self.norm_in(context_tokens)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm_out(x)
        return x


# ──────────────────────────────────────────────────────────
# EMA target encoder
# ──────────────────────────────────────────────────────────

class EMAEncoder(nn.Module):
    """
    Exponential Moving Average copy of the online encoder.
    Provides stable, slowly-updating targets for the JEPA loss.

    Momentum schedule: cosine ramp from momentum_start → momentum_end
    over total_steps steps.
    """
    def __init__(
        self,
        online_encoder: nn.Module,
        momentum_start: float = 0.996,
        momentum_end: float = 0.9999,
        total_steps: int = 100_000,
    ):
        super().__init__()
        import copy
        self.target_encoder = copy.deepcopy(online_encoder)
        # Freeze target encoder — updated only via EMA, not gradients
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.total_steps = total_steps
        self._step = 0

    @torch.no_grad()
    def update(self):
        """Call after each optimizer step to update EMA weights."""
        # Cosine momentum schedule
        progress = min(self._step / self.total_steps, 1.0)
        m = self.momentum_end - (self.momentum_end - self.momentum_start) * (
            math.cos(math.pi * progress) + 1
        ) / 2

        online_params = dict(self._online_ref.named_parameters())
        for name, param in self.target_encoder.named_parameters():
            if name in online_params:
                param.data.mul_(m).add_(online_params[name].data, alpha=1.0 - m)

        self._step += 1

    def set_online(self, online_encoder: nn.Module):
        """Must be called once to link the online encoder."""
        self._online_ref = online_encoder

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)


# ──────────────────────────────────────────────────────────
# JEPA Loss
# ──────────────────────────────────────────────────────────

def jepa_loss(
    predicted: torch.Tensor,    # (B, N_tgt, D) — predictor output at target positions
    target: torch.Tensor,        # (B, N_tgt, D) — EMA encoder output at target positions
    normalize: bool = True,
) -> torch.Tensor:
    """
    Smooth L2 loss in normalized latent space.
    Normalizing both sides prevents the scale of the latent space from dominating.
    """
    if normalize:
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target, dim=-1)

    loss = F.mse_loss(predicted, target)
    return loss


def check_collapse(
    tokens: torch.Tensor,
    eps: float = 1e-4,
) -> dict:
    """
    Diagnostic: checks for representation collapse.
    Returns a dict of scalar metrics to log to W&B.

    Collapse indicators:
      - std_mean < eps          : all tokens collapsed to a point
      - mean_cos_sim > 0.99     : all tokens nearly identical direction
    """
    with torch.no_grad():
        # Normalize and compute pairwise cosim on a small random subset
        B, N, D = tokens.shape
        n_sample = min(N, 64)
        idx = torch.randperm(N, device=tokens.device)[:n_sample]
        t = F.normalize(tokens[:, idx], dim=-1)          # (B, n_sample, D)
        cosim = torch.bmm(t, t.transpose(1, 2))          # (B, n_sample, n_sample)
        # Exclude diagonal
        mask = ~torch.eye(n_sample, dtype=torch.bool, device=tokens.device)
        mean_cos_sim = cosim[:, mask].mean().item()

        std_mean = tokens.std(dim=1).mean().item()        # average token variance

    return {
        "repr/std_mean": std_mean,
        "repr/mean_cos_sim": mean_cos_sim,
        "repr/collapsed": float(std_mean < eps or mean_cos_sim > 0.99),
    }
