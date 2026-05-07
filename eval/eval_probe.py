"""
Evaluation: Linear Probe + kNN Regression
==========================================
Evaluates frozen encoder representations on the task of predicting
physical parameters α (alpha) and ζ (zeta) as a regression problem.

Strictly follows the project rules:
  - Frozen encoder (no backbone fine-tuning)
  - Single linear layer OR kNN — no MLP heads
  - MSE loss on z-score normalized targets
  - Reports both linear probe and kNN MSE
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model import PISTJEPA
from data.dataset import build_dataloaders


# ──────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device, max_batches=None):
    """
    Run the frozen encoder over a dataloader and collect:
      - features : (N, D_repr) numpy array
      - labels   : (N, 2) numpy array  [alpha_normalized, zeta_normalized]
    """
    model.eval()
    all_feats  = []
    all_labels = []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x      = batch["frames"].to(device)
        labels = batch["labels"]           # (B, 2) — already normalized

        feats = model.encode(x)            # (B, D_repr)

        all_feats.append(feats.cpu().float().numpy())
        all_labels.append(labels.numpy())

        if (i + 1) % 50 == 0:
            print(f"  Extracted {(i+1)*x.shape[0]} samples...")

    return np.concatenate(all_feats, axis=0), np.concatenate(all_labels, axis=0)


# ──────────────────────────────────────────────────────────
# Linear Probe
# ──────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    """Single linear layer: D_repr → 2 (predicts normalized alpha, zeta)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.linear(x)


def train_linear_probe(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> dict:
    """
    Trains a linear probe on extracted features.
    Returns dict with train/val MSE for alpha and zeta separately.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.from_numpy(train_feats).float().to(device)
    Y_tr = torch.from_numpy(train_labels).float().to(device)
    X_val = torch.from_numpy(val_feats).float().to(device)
    Y_val = torch.from_numpy(val_labels).float().to(device)

    probe = LinearProbe(X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    N_tr = X_tr.shape[0]
    best_val_mse = float("inf")
    best_state = None

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(N_tr, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N_tr, batch_size):
            idx = perm[i:i+batch_size]
            pred = probe(X_tr[idx])
            loss = nn.functional.mse_loss(pred, Y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation MSE
        probe.eval()
        with torch.no_grad():
            val_pred  = probe(X_val)
            val_mse   = nn.functional.mse_loss(val_pred, Y_val).item()
            val_mse_a = nn.functional.mse_loss(val_pred[:, 0], Y_val[:, 0]).item()
            val_mse_z = nn.functional.mse_loss(val_pred[:, 1], Y_val[:, 1]).item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  [linear probe] Epoch {epoch+1:3d} | "
                  f"val MSE: {val_mse:.4f} (α={val_mse_a:.4f}, ζ={val_mse_z:.4f})")

    # Load best
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        val_pred  = probe(X_val)
        final_mse   = nn.functional.mse_loss(val_pred, Y_val).item()
        final_mse_a = nn.functional.mse_loss(val_pred[:, 0], Y_val[:, 0]).item()
        final_mse_z = nn.functional.mse_loss(val_pred[:, 1], Y_val[:, 1]).item()

    return {
        "linear_probe/mse_total":  final_mse,
        "linear_probe/mse_alpha":  final_mse_a,
        "linear_probe/mse_zeta":   final_mse_z,
        "linear_probe/best_val_mse": best_val_mse,
    }


# ──────────────────────────────────────────────────────────
# kNN Regression
# ──────────────────────────────────────────────────────────

def knn_regression(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    val_feats: np.ndarray,
    val_labels: np.ndarray,
    k_values: list = [1, 5, 10, 20],
) -> dict:
    """
    kNN regression with L2-normalized features.
    Tries multiple k values and reports the best.
    Uses GPU matrix multiplication for efficient nearest-neighbor search.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # L2 normalize features
    def l2_norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    X_tr_n  = l2_norm(train_feats)
    X_val_n = l2_norm(val_feats)

    X_tr_t  = torch.from_numpy(X_tr_n).float().to(device)
    X_val_t = torch.from_numpy(X_val_n).float().to(device)
    Y_tr_t  = torch.from_numpy(train_labels).float().to(device)
    Y_val_t = torch.from_numpy(val_labels).float().to(device)

    results = {}
    best_mse = float("inf")
    best_k = k_values[0]

    print(f"  Running kNN regression (N_train={len(train_feats)}, k={k_values})...")

    # Batch over val set to avoid OOM
    batch_size = 512
    for k in k_values:
        all_preds = []
        for i in range(0, X_val_t.shape[0], batch_size):
            xb = X_val_t[i:i+batch_size]
            # Cosine similarity (features are L2-normed)
            sims = xb @ X_tr_t.T                 # (B_val, N_train)
            topk = sims.topk(k, dim=1).indices   # (B_val, k)
            neighbor_labels = Y_tr_t[topk]       # (B_val, k, 2)
            pred = neighbor_labels.mean(dim=1)   # (B_val, 2)
            all_preds.append(pred)

        preds = torch.cat(all_preds, dim=0)

        mse_total = nn.functional.mse_loss(preds, Y_val_t).item()
        mse_alpha = nn.functional.mse_loss(preds[:, 0], Y_val_t[:, 0]).item()
        mse_zeta  = nn.functional.mse_loss(preds[:, 1], Y_val_t[:, 1]).item()

        print(f"  k={k:2d} | MSE: {mse_total:.4f} (α={mse_alpha:.4f}, ζ={mse_zeta:.4f})")
        results[f"knn_k{k}/mse_total"] = mse_total
        results[f"knn_k{k}/mse_alpha"] = mse_alpha
        results[f"knn_k{k}/mse_zeta"]  = mse_zeta

        if mse_total < best_mse:
            best_mse = mse_total
            best_k = k

    results["knn_best/k"]         = best_k
    results["knn_best/mse_total"] = results[f"knn_k{best_k}/mse_total"]
    results["knn_best/mse_alpha"] = results[f"knn_k{best_k}/mse_alpha"]
    results["knn_best/mse_zeta"]  = results[f"knn_k{best_k}/mse_zeta"]

    return results


# ──────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────

def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────
    print("Loading model...")
    model = PISTJEPA(
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        encoder_depth=cfg.encoder_depth,
        num_heads=cfg.num_heads,
        num_frames=cfg.num_frames,
        img_size=cfg.img_size,
        use_delta=cfg.use_delta,
        use_grad_ckpt=False,  # no grad ckpt needed for eval
    ).to(device)

    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Repr dim: {model.repr_dim}")

    # ── Build dataloaders ─────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg.data_root, batch_size=64, num_workers=8, stats_path=cfg.stats_path
    )

    # ── Extract features ──────────────────────────────────────────────
    print("Extracting train features...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    print(f"  Train: {train_feats.shape}")

    print("Extracting val features...")
    val_feats, val_labels = extract_features(model, val_loader, device)
    print(f"  Val: {val_feats.shape}")

    print("Extracting test features...")
    test_feats, test_labels = extract_features(model, test_loader, device)
    print(f"  Test: {test_feats.shape}")

    # ── Linear probe ─────────────────────────────────────────────────
    print("\n=== Linear Probe ===")
    linear_results = train_linear_probe(
        train_feats, train_labels, val_feats, val_labels,
        epochs=cfg.probe_epochs, lr=cfg.probe_lr,
    )

    # Also evaluate on test set
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = LinearProbe(model.repr_dim).to(device2)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=cfg.probe_lr, weight_decay=1e-4)
    X_tr = torch.from_numpy(train_feats).float().to(device2)
    Y_tr = torch.from_numpy(train_labels).float().to(device2)
    for _ in range(cfg.probe_epochs):
        perm = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], 256):
            idx = perm[i:i+256]
            loss = nn.functional.mse_loss(probe(X_tr[idx]), Y_tr[idx])
            probe_opt.zero_grad(); loss.backward(); probe_opt.step()
    probe.eval()
    with torch.no_grad():
        X_test = torch.from_numpy(test_feats).float().to(device2)
        Y_test = torch.from_numpy(test_labels).float().to(device2)
        test_pred = probe(X_test)
        linear_results["linear_probe/test_mse_total"] = nn.functional.mse_loss(test_pred, Y_test).item()
        linear_results["linear_probe/test_mse_alpha"] = nn.functional.mse_loss(test_pred[:,0], Y_test[:,0]).item()
        linear_results["linear_probe/test_mse_zeta"]  = nn.functional.mse_loss(test_pred[:,1], Y_test[:,1]).item()

    # ── kNN Regression ────────────────────────────────────────────────
    print("\n=== kNN Regression ===")
    knn_results = knn_regression(
        train_feats, train_labels, val_feats, val_labels,
        k_values=[1, 5, 10, 20],
    )

    # ── Print & save results ──────────────────────────────────────────
    all_results = {**linear_results, **knn_results}

    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    for k, v in sorted(all_results.items()):
        print(f"  {k:<45} {v:.6f}" if isinstance(v, float) else f"  {k:<45} {v}")

    output_path = Path(cfg.output_dir) / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--data_root",     required=True)
    p.add_argument("--stats_path",    default=None)
    p.add_argument("--output_dir",    default="./eval_results")
    # Model config (must match training)
    p.add_argument("--patch_size",    type=int, default=16)
    p.add_argument("--embed_dim",     type=int, default=384)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--num_heads",     type=int, default=6)
    p.add_argument("--num_frames",    type=int, default=16)
    p.add_argument("--img_size",      type=int, default=224)
    p.add_argument("--use_delta",     action="store_true", default=True)
    # Probe config
    p.add_argument("--probe_epochs",  type=int,   default=100)
    p.add_argument("--probe_lr",      type=float, default=1e-3)

    cfg = p.parse_args()
    evaluate(cfg)
