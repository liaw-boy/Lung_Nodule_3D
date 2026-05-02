"""PyTorch Dataset for the 3D volumes built by build_3d_volumes.py.

Each item:
  - volume: (1, D, H, W) float32 normalized to [-1, 1]
  - label : 0 / 1
  - aux   : (3,) lobulation / spiculation / margin in [0, 1]
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class LIDC3DDataset(Dataset):
    def __init__(self, rows: list[dict], augment: bool = False):
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        vol = np.load(r["volume_path"])  # (D, H, W) uint8

        if self.augment:
            if random.random() > 0.5:
                vol = vol[:, :, ::-1]
            if random.random() > 0.5:
                vol = vol[:, ::-1, :]
            if random.random() > 0.5:
                vol = vol[::-1, :, :]
            k = random.randint(0, 3)
            vol = np.rot90(vol, k, axes=(1, 2))

        vol = np.ascontiguousarray(vol)
        vol_t = (torch.from_numpy(vol).float().unsqueeze(0) / 255.0 - 0.5) / 0.5
        # (1, D, H, W)

        label = int(r["label"])
        aux = torch.tensor([
            (float(r["lobulation"]) - 1.0) / 4.0,
            (float(r["spiculation"]) - 1.0) / 4.0,
            (float(r["margin"]) - 1.0) / 4.0,
        ], dtype=torch.float32).clamp(0.0, 1.0)
        return vol_t, label, aux


def patient_split(index_csv: str, train_r: float = 0.70, val_r: float = 0.15,
                  seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Same split rule as the upstream 2D project (seed=42, patient-level)."""
    df = pd.read_csv(index_csv)
    pids = sorted(df["patient_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_tr = int(n * train_r)
    n_va = int(n * val_r)
    tr_p = set(pids[:n_tr])
    va_p = set(pids[n_tr:n_tr + n_va])
    te_p = set(pids[n_tr + n_va:])
    print(f"Patients — Train:{len(tr_p)} Val:{len(va_p)} Test:{len(te_p)}")
    tr = df[df.patient_id.isin(tr_p)].to_dict("records")
    va = df[df.patient_id.isin(va_p)].to_dict("records")
    te = df[df.patient_id.isin(te_p)].to_dict("records")
    print(f"Volumes  — Train:{len(tr)} Val:{len(va)} Test:{len(te)}")
    return tr, va, te


def create_loaders(index_csv: str, batch_size: int = 8, num_workers: int = 4):
    tr_rows, va_rows, te_rows = patient_split(index_csv)
    tr_ds = LIDC3DDataset(tr_rows, augment=True)
    va_ds = LIDC3DDataset(va_rows, augment=False)
    te_ds = LIDC3DDataset(te_rows, augment=False)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                    num_workers=max(1, num_workers // 2))
    te = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                    num_workers=max(1, num_workers // 2))
    return tr, va, te
