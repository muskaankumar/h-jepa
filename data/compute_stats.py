"""
Compute per-channel mean and std from the training set.
Run this BEFORE training to get correct normalization stats.

Usage:
    python compute_stats.py --data_root /scratch/$USER/data/active_matter \
                            --output stats.json
"""

import torch
import numpy as np
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import ActiveMatterDataset


def compute_stats(data_root: str, output_path: str, max_samples: int = 500):
    """
    Welford's online algorithm for numerically stable mean/std.
    Processes max_samples training examples (500 is sufficient for 11 channels).
    """
    ds = ActiveMatterDataset(data_root, split="train", normalize=False, augment=False)

    n = min(max_samples, len(ds))
    print(f"Computing stats over {n} training samples...")

    # Accumulate per-channel statistics
    ch_sum   = np.zeros(11, dtype=np.float64)
    ch_sq    = np.zeros(11, dtype=np.float64)
    count    = 0

    alpha_vals = []
    zeta_vals  = []

    for i in range(n):
        sample = ds[i]
        frames = sample["frames"].numpy()     # (T, 11, H, W)
        alpha_vals.append(sample["alpha_raw"])
        zeta_vals.append(sample["zeta_raw"])

        # Sum over T, H, W for each channel
        ch_sum += frames.sum(axis=(0, 2, 3))   # (11,)
        ch_sq  += (frames ** 2).sum(axis=(0, 2, 3))
        count  += frames.shape[0] * frames.shape[2] * frames.shape[3]

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n}")

    mean = ch_sum / count
    std  = np.sqrt(ch_sq / count - mean ** 2)
    std  = np.maximum(std, 1e-6)   # prevent division by zero

    alpha_arr = np.array(alpha_vals)
    zeta_arr  = np.array(zeta_vals)

    stats = {
        "mean": mean.tolist(),
        "std":  std.tolist(),
        "alpha_mean": float(alpha_arr.mean()),
        "alpha_std":  float(alpha_arr.std()),
        "zeta_mean":  float(zeta_arr.mean()),
        "zeta_std":   float(zeta_arr.std()),
        "n_samples":  n,
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {output_path}")
    print(f"Channel means: {[f'{v:.4f}' for v in mean]}")
    print(f"Channel stds:  {[f'{v:.4f}' for v in std]}")
    print(f"Alpha: mean={stats['alpha_mean']:.4f}, std={stats['alpha_std']:.4f}")
    print(f"Zeta:  mean={stats['zeta_mean']:.4f},  std={stats['zeta_std']:.4f}")

    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    required=True)
    p.add_argument("--output",       default="stats.json")
    p.add_argument("--max_samples",  type=int, default=500)
    cfg = p.parse_args()
    compute_stats(cfg.data_root, cfg.output, cfg.max_samples)
