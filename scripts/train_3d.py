"""3D ResNet18 + AttFB training on LIDC volumes.

Mirrors upstream train_attfeedback.py methodology:
  - patient-level split (seed=42, identical to 2D project)
  - cross-entropy + weighted aux MSE
  - cosine annealing LR + AMP
  - early stop on val F1 plateau
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.lidc_3d_dataset import create_loaders
from models.nodule_classifier_3d import NoduleClassifier3D


# Same correlation-based aux weights as upstream AttFB
AUX_WEIGHTS = torch.tensor([0.655, 0.631, 0.597])


def weighted_aux_loss(preds: torch.Tensor, targets: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(preds.device)
    return (((preds - targets) ** 2) * w).mean()


def train_epoch(model, loader, optimizer, scaler, device, aux_weight=0.3):
    model.train()
    total_loss = correct = n = 0
    ce = nn.CrossEntropyLoss()
    for vol, labels, aux in loader:
        vol, labels, aux = vol.to(device), labels.to(device), aux.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            logits, aux_pred = model(vol)
            loss = ce(logits, labels) + aux_weight * weighted_aux_loss(aux_pred, aux, AUX_WEIGHTS)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_l, all_p, all_pr = [], [], []
    for vol, labels, _ in loader:
        vol, labels = vol.to(device), labels.to(device)
        logits, _ = model(vol)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.4).long()
        all_l.extend(labels.cpu().tolist())
        all_p.extend(preds.cpu().tolist())
        all_pr.extend(probs.cpu().tolist())
    acc = sum(p == l for p, l in zip(all_p, all_l)) / max(len(all_l), 1)
    auc = roc_auc_score(all_l, all_pr) if len(set(all_l)) > 1 else 0.0
    rec = recall_score(all_l, all_p, zero_division=0)
    prec = precision_score(all_l, all_p, zero_division=0)
    f1 = f1_score(all_l, all_p, zero_division=0)
    cm = confusion_matrix(all_l, all_p)
    return acc, auc, rec, prec, f1, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", default=str(ROOT / "data" / "volumes_index_ctx.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Index : {args.index_csv}")

    tr_loader, va_loader, te_loader = create_loaders(args.index_csv,
                                                     batch_size=args.batch,
                                                     num_workers=4)

    model = NoduleClassifier3D(in_channels=1, n_aux=3,
                                use_attribute_feedback=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"3D ResNet18+CBAM+AttFB params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_auc = best_f1 = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, scaler, device)
        scheduler.step()
        acc, auc, rec, prec, f1, cm = evaluate(model, va_loader, device)
        flag = ""
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), out_dir / "resnet3d_best_auc.pth")
            flag += " [AUC]"
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "resnet3d_best_f1.pth")
            flag += " [F1]"
            no_improve = 0
        else:
            no_improve += 1
        print(f"Ep{epoch:3d} | loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val acc={acc:.3f} AUC={auc:.4f} Rec={rec:.3f} F1={f1:.3f}{flag}")
        if no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    print("\n=== TEST RESULTS ===")
    for tag, path in [("AUC-best", out_dir / "resnet3d_best_auc.pth"),
                      ("F1-best",  out_dir / "resnet3d_best_f1.pth")]:
        m = NoduleClassifier3D(in_channels=1, n_aux=3,
                                use_attribute_feedback=True).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        acc, auc, rec, prec, f1, cm = evaluate(m, te_loader, device)
        print(f"{tag}: Acc={acc:.4f} AUC={auc:.4f} Rec={rec:.4f} Prec={prec:.4f} F1={f1:.4f}")
        print(f"  CM: {cm.tolist()}")


if __name__ == "__main__":
    main()
