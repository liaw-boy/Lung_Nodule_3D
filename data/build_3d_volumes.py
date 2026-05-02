"""Build 3D volumes from upstream LIDC PNG slices.

For each (patient_id, nodule_id):
  - Collect all CTX slices, sort by index
  - Interpolate Z dimension to a fixed depth (default 16)
  - Save (D, H, W) uint8 volume as .npy
  - Append a row to volumes_index.csv with patient/nodule/label/path

Usage:
    python data/build_3d_volumes.py \
        --csv /home/lbw/project/LIDC-IDRI/nodules_hires/labels_multitask.csv \
        --out_dir data/volumes \
        --depth 16 --hw 128
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_SLICE_RE = re.compile(r"slice-(\d+)\.png$")


def _slice_idx(path: str) -> int:
    m = _SLICE_RE.search(path)
    return int(m.group(1)) if m else -1


def interpolate_depth(stack: np.ndarray, target_d: int) -> np.ndarray:
    """Linearly interpolate (N, H, W) → (target_d, H, W) along Z."""
    n, h, w = stack.shape
    if n == target_d:
        return stack
    if n == 1:
        return np.repeat(stack, target_d, axis=0)
    # Use OpenCV resize on (H*W, N) treating Z as a width dimension
    flat = stack.reshape(n, h * w).T.astype(np.float32)  # (H*W, N)
    resized = cv2.resize(flat, (target_d, h * w), interpolation=cv2.INTER_LINEAR)
    return resized.T.reshape(target_d, h, w).astype(np.uint8)


def build(csv_path: str, out_dir: str, depth: int, hw: int, kind: str) -> None:
    df = pd.read_csv(csv_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    path_col = "ctx_path" if kind == "ctx" else "roi_path"
    nodule_groups = df.groupby(["patient_id", "nodule_id"])
    print(f"Processing {len(nodule_groups)} nodules → {depth}×{hw}×{hw} volumes ({kind})")

    rows = []
    for (pid, nid), grp in nodule_groups:
        sorted_rows = grp.sort_values(path_col, key=lambda s: s.map(_slice_idx))
        slices = []
        for _, r in sorted_rows.iterrows():
            img = cv2.imread(r[path_col], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (hw, hw):
                img = cv2.resize(img, (hw, hw))
            slices.append(img)
        if not slices:
            print(f"  [skip] {pid} nodule-{nid}: no readable slices")
            continue
        stack = np.stack(slices, axis=0)  # (N, hw, hw)
        volume = interpolate_depth(stack, depth)  # (depth, hw, hw)

        rel = f"{pid}_n{nid}_{kind}.npy"
        out_path = out_root / rel
        np.save(out_path, volume)

        first = sorted_rows.iloc[0]
        rows.append({
            "patient_id": pid,
            "nodule_id": int(nid),
            "label": int(first["label"]),
            "lobulation": float(first["lobulation"]),
            "spiculation": float(first["spiculation"]),
            "margin": float(first["margin"]),
            "n_slices_orig": len(slices),
            "volume_path": str(out_path),
        })

    out_csv = out_root.parent / f"volumes_index_{kind}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✅ Wrote {len(rows)} volumes to {out_root}")
    print(f"✅ Index CSV: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/lbw/project/LIDC-IDRI/nodules_hires/labels_multitask.csv")
    ap.add_argument("--out_dir", default="data/volumes")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--hw", type=int, default=128)
    ap.add_argument("--kind", choices=["ctx", "roi"], default="ctx")
    args = ap.parse_args()
    build(args.csv, args.out_dir, args.depth, args.hw, args.kind)


if __name__ == "__main__":
    main()
