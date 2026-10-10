# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:42:40 2026

@author: prash
"""


import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import json

from dataset import CloudSEN12Dataset

# ── input channels per terrain mode (must match train.py) ──────────────────
IN_CHANNELS = {
    "none"   : 13,
    "elev"   : 14,
    "optionA": 15,
    "optionB": 17,
    "optionC": 19,
    "optionA_v2": 15,
    "optionB_v2": 17,
    "optionC_v2": 19,
    "latlon_only": 15,  # Stage 5b(1): 13 bands + lat + lon, no other terrain
    "elev_pooled": 14,  # pooled-elevation control
    "elev_latlon": 16,  # 13 bands + per-pixel elev + lat + lon, for arch="gap_latlon"
    "rgbn": 4,          # spectral-ablation follow-up: B2,B3,B4,B8 only
    "rgbn_elev": 5,     # the same 4 bands + per-pixel elevation
}

# ── metrics (identical to train.py, for direct comparability) ──────────────
def compute_metrics(preds, labels, num_classes=4):
    iou_per_class = []
    f1_per_class  = []

    for c in range(num_classes):
        pred_c  = (preds  == c)
        label_c = (labels == c)

        tp = (pred_c & label_c).sum().item()
        fp = (pred_c & ~label_c).sum().item()
        fn = (~pred_c & label_c).sum().item()

        iou = tp / (tp + fp + fn + 1e-8)
        f1  = 2*tp / (2*tp + fp + fn + 1e-8)

        iou_per_class.append(iou)
        f1_per_class.append(f1)

    return {
        "iou"  : iou_per_class,
        "miou" : float(np.mean(iou_per_class)),
        "f1"   : f1_per_class,
        "mf1"  : float(np.mean(f1_per_class)),
    }

# ── checkpoint loader: handles old (bare state_dict) and new (dict wrapper) ─
def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", None)
        best_miou = ckpt.get("best_miou", None)
        print(f"Loaded new-format checkpoint (epoch={epoch}, best_miou={best_miou})")
    else:
        model.load_state_dict(ckpt)
        print("Loaded old-format checkpoint (bare state_dict)")

    return model

# ── test-set evaluation loop ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  test ", leave=False):
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        preds  = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return compute_metrics(all_preds, all_labels)

# ── main callable function — run this directly in a notebook cell ──────────
def run_test_eval(terrain_mode, checkpoint_path, save_dir=None, batch_size=2, num_workers=0,
                  arch="unet"):   # arch="film" for Stage 5b(2)
    """
    Evaluate a trained TopoSegNet checkpoint on the test set.

    Example (in a Jupyter cell):
        results = run_test_eval(
            terrain_mode="optionA",
            checkpoint_path=r"C:\\Users\\prash\\toposegnet\\experiments\\unet_optionA_augFalse\\best_model.pth"
        )
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Terrain mode : {terrain_mode}")
    print(f"Checkpoint   : {checkpoint_path}")

    save_dir = save_dir or os.path.dirname(checkpoint_path)
    os.makedirs(save_dir, exist_ok=True)

    # ── dataset ──────────────────────────────────────────────────────────────
    test_ds = CloudSEN12Dataset(
        split="Test",
        terrain_mode=terrain_mode,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── model ────────────────────────────────────────────────────────────────
    in_channels = IN_CHANNELS[terrain_mode]
    if arch == "tcm":
        # Stage 6: `elev` is 14ch, encoder takes 13; elevation goes to the
        # cross-attention terrain branch.
        from model import build_tcm_unet
        model = build_tcm_unet(
            in_channels=in_channels - 1, num_classes=4, pretrained=False
        ).to(device)
        print(f"Model        : TCM-UNet ResNet-34 | encoder_in={in_channels - 1} "
              f"(+elevation via cross-attention) | dataset_ch={in_channels}")
    elif arch == "gap":
        # gap_terrain ablation: TCM with attention deleted; same channel split.
        from model import build_gap_terrain_unet
        model = build_gap_terrain_unet(
            in_channels=in_channels - 1, num_classes=4, pretrained=False
        ).to(device)
        print(f"Model        : GAP-Terrain UNet ResNet-34 | encoder_in={in_channels - 1} "
              f"(+elevation via global-avg-pooled terrain) | dataset_ch={in_channels}")
    elif arch == "film_elev":
        # Pooled-elevation FiLM control: 14ch elev_pooled tensor, encoder takes
        # 13 bands, patch-mean elevation injected as gamma/beta at bottleneck.
        from model import build_film_elev_unet
        model = build_film_elev_unet(
            in_channels=in_channels - 1, num_classes=4, pretrained=False
        ).to(device)
        print(f"Model        : FiLM-Elev UNet ResNet-34 | encoder_in={in_channels - 1} "
              f"(+patch-mean elevation via bottleneck FiLM) | dataset_ch={in_channels}")
    elif arch == "gap_latlon":
        # gap + lat/lon FiLM: 16ch elev_latlon, encoder takes 13 bands.
        from model import build_gap_latlon_unet
        model = build_gap_latlon_unet(
            in_channels=in_channels - 3, num_classes=4, pretrained=False
        ).to(device)
        print(f"Model        : GAP-Terrain + lat/lon FiLM UNet ResNet-34 | encoder_in={in_channels - 3} "
              f"(+elevation via gap, +lat/lon via bottleneck FiLM) | dataset_ch={in_channels}")
    elif arch in ("film", "film_multi"):
        # Stage 5b(2): encoder sees bands only; lat/lon enter via bottleneck FiLM.
        # Stage 5b(3) "film_multi": same, but every encoder scale is conditioned.
        # (film_multi added 2026-07-28)
        from model import build_film_unet, build_multiscale_film_unet
        builder = build_multiscale_film_unet if arch == "film_multi" else build_film_unet
        model = builder(
            in_channels=in_channels - 2, num_classes=4, pretrained=False
        ).to(device)
        where = "all scales" if arch == "film_multi" else "bottleneck"
        print(f"Model        : FiLM-UNet ResNet-34 | encoder_in={in_channels - 2} "
              f"(+lat/lon via {where}) | dataset_ch={in_channels}")
    else:
        model = smp.Unet(
            encoder_name    = "resnet34",
            encoder_weights = None,
            in_channels     = in_channels,
            classes         = 4,
        ).to(device)
        print(f"Model        : U-Net ResNet-34 | in_channels={in_channels}")

    model = load_checkpoint(model, checkpoint_path, device)

    # ── evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate_test(model, test_loader, device)

    print("\n── Test Results ──────────────────────────────────────────────")
    print(f"  mIoU = {metrics['miou']:.4f} | mF1 = {metrics['mf1']:.4f}")
    print(f"  per-class IoU: clear={metrics['iou'][0]:.4f} "
          f"thick={metrics['iou'][1]:.4f} "
          f"thin={metrics['iou'][2]:.4f} "
          f"shadow={metrics['iou'][3]:.4f}")
    print(f"  per-class F1 : clear={metrics['f1'][0]:.4f} "
          f"thick={metrics['f1'][1]:.4f} "
          f"thin={metrics['f1'][2]:.4f} "
          f"shadow={metrics['f1'][3]:.4f}")

    # ── save results ─────────────────────────────────────────────────────────
    results = {
        "terrain_mode": terrain_mode,
        "checkpoint": checkpoint_path,
        "test_miou": metrics["miou"],
        "test_mf1": metrics["mf1"],
        "test_iou_clear": metrics["iou"][0],
        "test_iou_thick": metrics["iou"][1],
        "test_iou_thin": metrics["iou"][2],
        "test_iou_shadow": metrics["iou"][3],
        "test_f1_clear": metrics["f1"][0],
        "test_f1_thick": metrics["f1"][1],
        "test_f1_thin": metrics["f1"][2],
        "test_f1_shadow": metrics["f1"][3],
    }

    out_path = os.path.join(save_dir, "test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results