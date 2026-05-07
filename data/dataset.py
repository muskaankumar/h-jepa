"""
Active Matter Dataset
======================
HDF5 structure (per file):
  - t0_fields/concentration : (3, 81, 256, 256)        — 1 scalar channel
  - t1_fields/velocity      : (3, 81, 256, 256, 2)     — 2 vector channels
  - t2_fields/D             : (3, 81, 256, 256, 2, 2)  — 4 orientation tensor channels
  - t2_fields/E             : (3, 81, 256, 256, 2, 2)  — 4 strain-rate tensor channels
  - attrs: alpha, zeta

Each file has 3 trajectories × 81 frames.
With num_frames=16 and stride=4, each trajectory yields ~16 windows.
45 train files × 3 traj × 16 windows = ~2160 effective training samples.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Optional, Tuple, List

CHANNEL_MEAN = np.zeros(11, dtype=np.float32)
CHANNEL_STD  = np.ones(11,  dtype=np.float32)


class ActiveMatterDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 16,
        img_size: int = 224,
        normalize: bool = True,
        augment: bool = False,
        stats_path: Optional[str] = None,
        window_stride: int = 1,   # stride between temporal windows
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.normalize = normalize
        self.augment = augment and (split == "train")
        self.window_stride = window_stride if split == "train" else num_frames

        if stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                stats = json.load(f)
            self.ch_mean    = np.array(stats["mean"], dtype=np.float32)
            self.ch_std     = np.array(stats["std"],  dtype=np.float32)
            self.alpha_mean = stats.get("alpha_mean", 0.0)
            self.alpha_std  = stats.get("alpha_std",  1.0)
            self.zeta_mean  = stats.get("zeta_mean",  0.0)
            self.zeta_std   = stats.get("zeta_std",   1.0)
        else:
            self.ch_mean    = CHANNEL_MEAN.copy()
            self.ch_std     = CHANNEL_STD.copy()
            self.alpha_mean = 0.0
            self.alpha_std  = 1.0
            self.zeta_mean  = 0.0
            self.zeta_std   = 1.0

        # Build index: list of (filepath, traj_idx, start_frame)
        self.samples = self._build_index()
        print(f"[{split}] {len(self.samples)} samples "
              f"(stride={self.window_stride}) from "
              f"{len(set(str(p) for p,_,_ in self.samples))} files")

    def _build_index(self) -> List[Tuple[Path, int, int]]:
        split_name = "valid" if self.split == "val" else self.split
        split_dir = self.data_root / split_name
        if not split_dir.exists():
            split_dir = self.data_root / "data" / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        files = sorted(split_dir.glob("*.hdf5")) + sorted(split_dir.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No HDF5 files in {split_dir}")

        index = []
        for path in files:
            try:
                with h5py.File(path, "r") as f:
                    n_traj = f["t0_fields"]["concentration"].shape[0]
                    T_raw  = f["t0_fields"]["concentration"].shape[1]
                # For each trajectory, enumerate all valid start frames
                starts = range(0, T_raw - self.num_frames + 1, self.window_stride)
                for traj_idx in range(n_traj):
                    for start in starts:
                        index.append((path, traj_idx, start))
            except Exception as e:
                print(f"Warning: could not open {path}: {e}")
        return index

    def _load_sample(
        self, path: Path, traj_idx: int, start: int
    ) -> Tuple[np.ndarray, float, float]:
        end = start + self.num_frames
        with h5py.File(path, "r") as f:
            alpha = float(f.attrs["alpha"])
            zeta  = float(f.attrs["zeta"])
            conc  = f["t0_fields"]["concentration"][traj_idx, start:end]   # (T, 256, 256)
            vel   = f["t1_fields"]["velocity"][traj_idx, start:end]        # (T, 256, 256, 2)
            D     = f["t2_fields"]["D"][traj_idx, start:end]               # (T, 256, 256, 2, 2)
            E     = f["t2_fields"]["E"][traj_idx, start:end]               # (T, 256, 256, 2, 2)

        T, H, W = conc.shape
        D_flat = D.reshape(T, H, W, 4)
        E_flat = E.reshape(T, H, W, 4)

        frames = np.concatenate([
            conc[..., np.newaxis],
            vel,
            D_flat,
            E_flat,
        ], axis=-1)                                    # (T, H, W, 11)
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32)  # (T, 11, H, W)
        return frames, alpha, zeta

    def _spatial_crop(self, frames: np.ndarray) -> np.ndarray:
        T, C, H, W = frames.shape
        if H == self.img_size and W == self.img_size:
            return frames
        if self.augment:
            top  = np.random.randint(0, H - self.img_size + 1)
            left = np.random.randint(0, W - self.img_size + 1)
        else:
            top  = (H - self.img_size) // 2
            left = (W - self.img_size) // 2
        return frames[:, :, top:top + self.img_size, left:left + self.img_size]

    def _augment_spatial(self, frames: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            frames = np.flip(frames, axis=3).copy()
            frames[:, 1] *= -1
        if np.random.rand() < 0.5:
            frames = np.flip(frames, axis=2).copy()
            frames[:, 2] *= -1
        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, traj_idx, start = self.samples[idx]
        frames, alpha, zeta = self._load_sample(path, traj_idx, start)
        frames = self._spatial_crop(frames)
        if self.augment:
            frames = self._augment_spatial(frames)
        if self.normalize:
            frames = (frames - self.ch_mean[None, :, None, None]) / (
                self.ch_std[None, :, None, None] + 1e-6
            )
        x = torch.from_numpy(frames)
        alpha_norm = (alpha - self.alpha_mean) / (self.alpha_std + 1e-6)
        zeta_norm  = (zeta  - self.zeta_mean)  / (self.zeta_std  + 1e-6)
        labels = torch.tensor([alpha_norm, zeta_norm], dtype=torch.float32)
        return {
            "frames":    x,
            "labels":    labels,
            "alpha_raw": alpha,
            "zeta_raw":  zeta,
        }


def build_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 8,
    stats_path: Optional[str] = None,
    window_stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ActiveMatterDataset(
        data_root, "train", normalize=True, augment=True,
        stats_path=stats_path, window_stride=window_stride
    )
    val_ds = ActiveMatterDataset(
        data_root, "val", normalize=True, augment=False,
        stats_path=stats_path, window_stride=16
    )
    test_ds = ActiveMatterDataset(
        data_root, "test", normalize=True, augment=False,
        stats_path=stats_path, window_stride=16
    )
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, persistent_workers=(num_workers > 0))
    return (DataLoader(train_ds, shuffle=True,  drop_last=True,  **kwargs),
            DataLoader(val_ds,   shuffle=False, drop_last=False, **kwargs),
            DataLoader(test_ds,  shuffle=False, drop_last=False, **kwargs)) 