"""
Stage 1a terrain-stratified error analysis
==========================================

Question this answers: do the *baseline* model's errors correlate with terrain?

The Stage 1a model (terrain_mode="none") never sees elevation/slope/aspect.
If its test-set errors concentrate in high/steep/snowy terrain, that is direct
evidence that (a) terrain-correlated signal exists in the failure modes, and
(b) a terrain-aware architecture (TCM) has headroom to exploit it.
If errors do NOT correlate with terrain, the whole terrain-conditioning thesis
is at risk — better to know now than after building TCM.

What it does:
  1. Runs Stage 1a (unet_none_augFalse) inference over all 975 test patches.
  2. Per patch  → per-class TP/FP/FN, elevation/slope stats, snow fraction
                  (NDSI proxy), centroid lon/lat  →  per_patch.csv
  3. Per pixel  → 4x4 confusion matrices binned by elevation / slope / snow
                  (accumulated online, never holds all 256M pixels)
                  →  pixel_confusions.json
  4. Prints a stratified summary (error rates + key confusions per bin,
     Spearman correlations at patch level).

Run:  python analysis/stage1a_terrain_error_analysis.py
      (conda env `toposegnet`; needs GPU but ~4 GB is plenty at batch=4)

NOTE: model input is 13 spectral bands ONLY. Terrain memmaps are loaded
      separately and used purely for stratification — the model never sees them.
"""

import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import CloudSEN12Dataset

# ── config ────────────────────────────────────────────────────────────────────
CKPT_PATH   = r"C:\Users\prash\toposegnet\experiments\unet_none_augFalse\best_model.pth"
DATA_ROOT   = r"C:\Users\prash\toposegnet\data\high_509"
OUT_DIR     = r"C:\Users\prash\toposegnet\experiments\stage1a_error_analysis"
BATCH_SIZE  = 4
NUM_CLASSES = 4
CLASS_NAMES = ["clear", "thick", "thin", "shadow"]
N_TEST      = 975

# Stratification bins.
# Elevation (metres): 6 bins. Right-open [lo, hi).
ELEV_EDGES  = np.array([-np.inf, 0, 500, 1000, 2000, 3000, np.inf])
ELEV_LABELS = ["<0", "0-500", "500-1000", "1000-2000", "2000-3000", ">3000"]
# Slope (degrees): 5 bins.
SLOPE_EDGES  = np.array([-np.inf, 5, 15, 30, 45, np.inf])
SLOPE_LABELS = ["0-5", "5-15", "15-30", "30-45", ">45"]

# Snow proxy: NDSI = (B3 - B11) / (B3 + B11) > 0.4 (standard Dozier threshold)
# AND B8 (NIR) reflectance > 0.11 to exclude water — water is also high-NDSI
# (verified: 15 all-zero-elevation equatorial patches had "snow_frac" 0.62,
# i.e. ocean). Snow is NIR-bright, water is NIR-dark.
# Band indices in the 13-band stack: B3 = idx 2, B8 = idx 7, B11 = idx 11.
# Caveat: TOA-based; some ice cloud is also high-NDSI — a proxy, not truth.
NDSI_B3_IDX, NDSI_B8_IDX, NDSI_B11_IDX = 2, 7, 11
NDSI_THRESH = 0.4
NIR_THRESH  = 0.11
SURF_LABELS = ["other", "snow", "water_like"]   # mutually exclusive pixel classes

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    """Build 13-channel U-Net and load weights, handling BOTH checkpoint
    formats (stages 1-3: bare state_dict; stage 4+: dict wrapper)."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,      # weights come from the checkpoint
        in_channels=13,
        classes=NUM_CLASSES,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]     # new (stage 4+) format
    else:
        state = ckpt                          # old (stage 1-3) format
    model.load_state_dict(state)
    return model.to(device).eval()


def bin_index(values, edges):
    """Map float array → integer bin index 0..len(edges)-2.
    e.g. elevation 683.0 with ELEV_EDGES → bin 2 ("500-1000")."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1,
                   0, len(edges) - 2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = load_model(CKPT_PATH, device)
    print(f"Loaded : {CKPT_PATH}")

    # Model input: 13 bands only (terrain_mode='none'), no augmentation.
    ds = CloudSEN12Dataset(split="Test", terrain_mode="none", augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)   # HARD: num_workers=0

    # Terrain memmaps — for stratification only, model never sees these.
    shape = (N_TEST, 512, 512)
    elev  = np.memmap(f"{DATA_ROOT}\\Test\\EXTRA_elevation.dat",
                      dtype="int16",   mode="r", shape=shape)
    slope = np.memmap(f"{DATA_ROOT}\\Test\\DERIVED_slope.dat",
                      dtype="float32", mode="r", shape=shape)

    # Patch centroids (same parsing as dataset.py optionC).
    meta = pd.read_csv(f"{DATA_ROOT}\\Test\\metadata.csv")
    cent = meta["proj_centroid"].str.extract(r"POINT \(([0-9.\-]+) ([0-9.\-]+)\)")
    lons = cent[0].astype(float).values
    lats = cent[1].astype(float).values

    n_elev_bins, n_slope_bins = len(ELEV_LABELS), len(SLOPE_LABELS)

    # Pixel-level accumulators: confusion[bin, true, pred] as int64 counts.
    conf_elev  = np.zeros((n_elev_bins,  NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_slope = np.zeros((n_slope_bins, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_surf  = np.zeros((3,            NUM_CLASSES, NUM_CLASSES), dtype=np.int64)  # other/snow/water

    # persist raw predictions so future re-slicing needs no GPU (975*512*512 int8 = 256 MB)
    preds_mm = np.lib.format.open_memmap(
        os.path.join(OUT_DIR, "preds_stage1a.npy"),
        mode="w+", dtype=np.int8, shape=(N_TEST, 512, 512))

    patch_rows = []

    with torch.no_grad():
        for b, (images, labels) in enumerate(tqdm(loader, desc="inference")):
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()  # (B,512,512)
            labels_np = labels.numpy()
            i0 = b * BATCH_SIZE                       # shuffle=False → dataset idx = i0..i0+B-1

            # NDSI from the (already /10000-scaled) input bands — ratio, scale cancels.
            # B8 is /10000-scaled reflectance, so NIR_THRESH applies directly.
            b3  = images[:, NDSI_B3_IDX].numpy()
            b8  = images[:, NDSI_B8_IDX].numpy()
            b11 = images[:, NDSI_B11_IDX].numpy()
            ndsi = (b3 - b11) / (b3 + b11 + 1e-8)
            high_ndsi = ndsi > NDSI_THRESH
            snow  = high_ndsi & (b8 > NIR_THRESH)     # NIR-bright  → snow/ice
            water = high_ndsi & (b8 <= NIR_THRESH)    # NIR-dark    → water-like
            # surface class per pixel: 0=other, 1=snow, 2=water_like
            surf = snow.astype(np.int64) + 2 * water.astype(np.int64)

            for k in range(preds.shape[0]):
                idx  = i0 + k
                p, y = preds[k], labels_np[k]
                preds_mm[idx] = p.astype(np.int8)
                ev   = elev[idx].astype(np.float32)   # metres
                sl   = slope[idx]                     # degrees
                sn   = snow[k]
                wt   = water[k]

                # ── pixel-level: accumulate binned 4x4 confusions ─────────
                # combined index = bin*16 + true*4 + pred → one bincount pass
                yp = (y.ravel() * NUM_CLASSES + p.ravel()).astype(np.int64)
                eb = bin_index(ev.ravel(), ELEV_EDGES)
                conf_elev  += np.bincount(eb * 16 + yp, minlength=n_elev_bins * 16
                                          ).reshape(n_elev_bins, 4, 4)
                sb = bin_index(sl.ravel(), SLOPE_EDGES)
                conf_slope += np.bincount(sb * 16 + yp, minlength=n_slope_bins * 16
                                          ).reshape(n_slope_bins, 4, 4)
                conf_surf  += np.bincount(surf[k].ravel() * 16 + yp,
                                          minlength=3 * 16).reshape(3, 4, 4)

                # ── patch-level row ───────────────────────────────────────
                row = {
                    "idx": idx,
                    "err_rate": float((p != y).mean()),
                    "mean_elev": float(ev.mean()), "max_elev": float(ev.max()),
                    "std_elev": float(ev.std()),
                    "mean_slope": float(sl.mean()), "max_slope": float(sl.max()),
                    "snow_frac": float(sn.mean()),
                    "water_frac": float(wt.mean()),
                    "elev_zero_frac": float((ev == 0).mean()),
                    # snow among truly-clear pixels = cloud-free snowy ground,
                    # the exact surface hypothesized to fool the baseline
                    "clear_snow_frac": float((sn & (y == 0)).sum() / max((y == 0).sum(), 1)),
                    "lon": lons[idx], "lat": lats[idx],
                }
                for c, name in enumerate(CLASS_NAMES):
                    tp = int(((p == c) & (y == c)).sum())
                    fp = int(((p == c) & (y != c)).sum())
                    fn = int(((p != c) & (y == c)).sum())
                    union = tp + fp + fn
                    row[f"tp_{name}"] = tp
                    row[f"fp_{name}"] = fp
                    row[f"fn_{name}"] = fn
                    row[f"iou_{name}"] = tp / union if union > 0 else np.nan
                    row[f"frac_{name}"] = float((y == c).mean())
                patch_rows.append(row)

    # ── save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(patch_rows)
    df.to_csv(os.path.join(OUT_DIR, "per_patch.csv"), index=False)

    preds_mm.flush()

    with open(os.path.join(OUT_DIR, "pixel_confusions.json"), "w") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "elev_labels": ELEV_LABELS,   "conf_elev":  conf_elev.tolist(),
            "slope_labels": SLOPE_LABELS, "conf_slope": conf_slope.tolist(),
            "surf_labels": SURF_LABELS,   "conf_surf":  conf_surf.tolist(),
            "ndsi_thresh": NDSI_THRESH,   "nir_thresh": NIR_THRESH,
            "checkpoint": CKPT_PATH,
        }, f, indent=2)

    # sanity: every pixel counted exactly once per stratifier
    total_px = N_TEST * 512 * 512
    assert conf_elev.sum() == total_px, "elevation confusion lost pixels"
    assert conf_slope.sum() == total_px
    assert conf_surf.sum() == total_px

    # Summaries/correlations live in summarize_stage1a_errors.py (run it next).
    # NOTE: do not use np.corrcoef here — numpy's BLAS path hard-crashes in
    # this env (Windows delay-load failure 0xc06d007f).
    print(f"Saved: {OUT_DIR}\\per_patch.csv  ({len(df)} rows)")
    print(f"Saved: {OUT_DIR}\\pixel_confusions.json")
    print(f"Saved: {OUT_DIR}\\preds_stage1a.npy")


if __name__ == "__main__":
    main()
