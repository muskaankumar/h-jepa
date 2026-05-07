# PIST-JEPA
## Physically-Informed Spatiotemporal Tokenizer with Hierarchical JEPA

Self-supervised representation learning for the `active_matter` physical simulation dataset.

## Architecture Overview

```
Input (B, T=16, C=11, 224, 224)
        │
        ├─── Temporal diff tokens (xt+1 - xt) ─────┐
        │                                            │
        ▼                                            │
Channel-Group Factorized Tokenizer                   │
  ├── Group A: Concentration   [ch 0]               │
  ├── Group B: Velocity        [ch 1,2]             │
  ├── Group C: Orientation tensor [ch 3-6]          │
  └── Group D: Strain-rate tensor [ch 7-10]         │
        │                                            │
        ▼                                            │
Cross-Group Attention Fusion ◄───────────────────────┘
        │
        ▼
Spatiotemporal Encoder (ViT-S, 6 layers, 384 dim)
        │
    ┌───┴───┐
    │       │
Context   Target (last K frames, causal mask)
    │       │
    ▼       ▼
Predictor  EMA Encoder (momentum-updated)
    │       │
    └───┬───┘
        │
    JEPA Loss (L2 in normalized latent space)

── Evaluation ──────────────────────────
Frozen Encoder → Hierarchical Pool (mean + spatial var)
        │
  ┌─────┴──────┐
  │            │
Linear Probe  kNN Regression
  └─────┬──────┘
        │
    MSE(α, ζ)
```

## Key Novel Contributions

1. **Channel-Group Factorized Tokenization** — 4 separate patch embedders per physical field group, fused via cross-group attention. Respects physics instead of flattening all channels.

2. **Temporal Difference Tokens** — `xt+1 - xt` prepended to the token sequence. Provides explicit signal for the nematic transition dynamics.

3. **Shifted Causal Masking** — Always masks the LAST K frames (not random), forcing temporal dynamics to be encoded in context representations.

4. **Hierarchical Pooling for Probing** — Global mean + local spatial variance concatenated. Captures both mean state and spatial heterogeneity.

## Parameter Budget

| Component           | Parameters |
|---------------------|-----------|
| Channel tokenizer   | ~0.5M     |
| Cross-group fusion  | ~0.3M     |
| ViT-S encoder       | ~5.5M     |
| Predictor           | ~0.5M     |
| **Total**           | **~7M**   |

Well under the 100M limit. VRAM usage ~25-35 GB at batch size 8 on A100 40GB.

## Files

```
pist_jepa/
├── models/
│   ├── tokenizer.py    # Channel-group tokenizer + cross-group fusion
│   ├── encoder.py      # Spatiotemporal ViT encoder
│   ├── jepa.py         # Predictor, EMA encoder, loss, collapse diagnostics
│   └── model.py        # Full PIST-JEPA model (assembles all components)
├── data/
│   ├── dataset.py      # ActiveMatterDataset + dataloaders
│   └── compute_stats.py # Per-channel normalization stats
├── training/
│   └── train.py        # Training loop (checkpoint, W&B, cosine LR)
├── eval/
│   └── eval_probe.py   # Linear probe + kNN evaluation
├── scripts/
│   ├── train_a100.sh   # Slurm script for training
│   └── eval.sh         # Slurm script for evaluation
├── ENV.md              # Full HPC environment setup
└── requirements.txt
```

## Quick Start

```bash
# 1. Setup environment (see ENV.md)
# 2. Download data
huggingface-cli download polymathic-ai/active_matter \
  --repo-type dataset --local-dir /scratch/$USER/data/active_matter

# 3. Compute normalization stats
python data/compute_stats.py \
  --data_root /scratch/$USER/data/active_matter \
  --output    /scratch/$USER/data/active_matter/stats.json

# 4. Train
sbatch scripts/train_a100.sh

# 5. Evaluate
sbatch scripts/eval.sh
```

## Rules Compliance Checklist

- [x] Training from scratch (no pretrained weights)
- [x] Model < 100M parameters (~7M)
- [x] Only active_matter dataset
- [x] No training on val/test splits
- [x] Linear probe evaluation (single linear layer)
- [x] kNN regression evaluation
- [x] MSE loss (regression, not classification)
- [x] No complex regression heads
- [x] Checkpoint/requeue for spot instances (#SBATCH --requeue)
