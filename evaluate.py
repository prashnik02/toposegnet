"""
Evaluation script — loads best saved model and evaluates on test set.
Run: python evaluate.py
"""

import torch
import numpy as np
import os
from dataset import build_dataloaders
from model import build_baseline_unet

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_DIR   = r"C:\Users\prash\toposegnet\data\high-fixed"
CKPT_PATH  = r"C:\Users\prash\toposegnet\checkpoints\best_unet.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_CLASSES = 4
CLASS_NAMES = ["clear", "thick cloud", "thin cloud", "shadow"]

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def compute_metrics(preds, labels, num_classes=4):
    """
    Compute per-class IoU, F1, precision, recall and mean IoU.
    preds:  (N, H, W)
    labels: (N, H, W)
    """
    ious, f1s, precisions, recalls = [], [], [], []

    for cls in range(num_classes):
        pred_cls  = (preds == cls)
        label_cls = (labels == cls)

        tp = (pred_cls & label_cls).sum().item()
        fp = (pred_cls & ~label_cls).sum().item()
        fn = (~pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()

        iou       = tp / union if union > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else float('nan'))

        ious.append(iou)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    return ious, f1s, precisions, recalls


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
def evaluate_test(model, loader, device):
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)           # (B, 4, 256, 256)
            preds   = outputs.argmax(dim=1)   # (B, 256, 256)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds, dim=0)   # (N, 256, 256)
    all_labels = torch.cat(all_labels, dim=0)  # (N, 256, 256)

    ious, f1s, precisions, recalls = compute_metrics(
        all_preds, all_labels, NUM_CLASSES
    )

    return ious, f1s, precisions, recalls


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CKPT_PATH}\n")

    # Data — test set only
    _, _, test_loader = build_dataloaders(
        data_dir=DATA_DIR,
        patch_size=256,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # Model
    model = build_baseline_unet(in_channels=13, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    print("Model loaded successfully.\n")

    # Evaluate
    ious, f1s, precisions, recalls = evaluate_test(model, test_loader, DEVICE)

    # Report
    print("=" * 60)
    print(f"{'Class':<15} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    print("=" * 60)
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name:<15} "
            f"{ious[i]:>8.4f} "
            f"{f1s[i]:>8.4f} "
            f"{precisions[i]:>8.4f} "
            f"{recalls[i]:>8.4f}"
        )
    print("=" * 60)
    miou = np.nanmean(ious)
    mf1  = np.nanmean(f1s)
    print(f"{'mIoU':<15} {miou:>8.4f}")
    print(f"{'mF1':<15} {mf1:>8.4f}")
    print("=" * 60)