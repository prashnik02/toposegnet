"""
Joint snow x elevation, and slope stratification — the two gaps found 2026-08-07.
================================================================================


Two questions the existing artifacts cannot answer:

Q1  IS THE TCM EFFECT A SNOW EFFECT OR AN ALTITUDE EFFECT?
    Snow is 7.3% of test pixels, >2000 m is 8.6%, and they overlap heavily.
    `stratified_model_comparison.py` stores confusions binned by surface AND
    binned by elevation, but never JOINTLY, so the two cannot be separated.
    The joint accumulator here gives snow-at-low-altitude vs snow-at-high-
    altitude vs non-snow-at-high-altitude, which does separate them.

Q2  DOES SLOPE MATTER, PER CLASS?
    Never measured. The only slope result in the project is Stage 1a's
    patch-level Spearman of slope vs overall error rate (|rho| <= 0.09), which
    is a much weaker statement than per-class IoU by slope bin. Shadow in
    particular is tied to illumination geometry, so slope is the variable it
    should respond to most — and the elevation analysis just showed shadow IoU
    varies by 0.162 across altitude, ~40x any model effect.

Uses the CLEAN v2 slope (`DERIVED_slope_v2.dat`); v1 is the corrupted staircase
raster documented in project notes and must not be used for anything.

Surface definitions are byte-identical to stage1a_terrain_error_analysis.py and
stratified_model_comparison.py, so all three analyses stratify the same pixels.

Writes incrementally after every model — the 2026-08-07 sleep incident destroyed
a 10-model run that only serialised at the end.

Run:  python analysis/joint_slope_elev_stratification.py    (~25 min on GPU)
"""

import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")
sys.path.append(r"C:\Users\prash\toposegnet\analysis")

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CloudSEN12Dataset
from stratified_model_comparison import (
    MODELS, EXP, DATA_ROOT, N_TEST, NUM_CLASSES, CLASS_NAMES,
    NDSI_B3_IDX, NDSI_B8_IDX, NDSI_B11_IDX, NDSI_THRESH, NIR_THRESH,
    ELEV_EDGES, ELEV_LABELS, SURF_LABELS,
    load_model, bin_index, metrics_from_conf,
    CKPT_FILE, OUT_SUFFIX,   # best-val vs final-epoch switch
    training_unfinished,     # skip runs still mid-training
    spectral_bands,          # strata from the 13-band memmaps, not the model input
)

OUT_DIR   = os.path.join(EXP, "joint_stratification")
BATCH     = 4
N_SURF    = len(SURF_LABELS)      # 3
N_ELEV    = len(ELEV_LABELS)      # 6

# v2 slope: global median ~1.7-1.8 deg, mountainous patches ~92% within 5-45 deg,
# frac >45 deg is 4-6% (real steep terrain + source-DEM void cliffs).
SLOPE_EDGES  = np.array([-np.inf, 2, 5, 10, 20, 30, np.inf])
SLOPE_LABELS = ["<2", "2-5", "5-10", "10-20", "20-30", ">30"]
N_SLOPE      = len(SLOPE_LABELS)


@torch.no_grad()
def evaluate_joint(spec, device, elev_mm, slope_mm):
    model, epoch = load_model(spec, device)
    ds = CloudSEN12Dataset(split="Test", terrain_mode=spec["terrain_mode"], augment=False)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=True)      # HARD: num_workers=0

    c_surf_elev  = np.zeros((N_SURF, N_ELEV, 4, 4),  dtype=np.int64)
    c_slope      = np.zeros((N_SLOPE, 4, 4),         dtype=np.int64)
    c_slope_surf = np.zeros((N_SLOPE, N_SURF, 4, 4), dtype=np.int64)

    for b, (images, labels) in enumerate(tqdm(loader, desc=f"  {spec['name'][:28]:28s}",
                                              leave=False)):
        preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
        y_np  = labels.numpy()
        i0    = b * BATCH

        b3, b8, b11 = spectral_bands(i0, preds.shape[0])   # 4-band models have no B11 in their input
        ndsi = (b3 - b11) / (b3 + b11 + 1e-8)
        hi   = ndsi > NDSI_THRESH
        surf = (hi & (b8 > NIR_THRESH)).astype(np.int64) \
             + 2 * (hi & (b8 <= NIR_THRESH)).astype(np.int64)

        for k in range(preds.shape[0]):
            idx = i0 + k
            yp  = (y_np[k].ravel() * NUM_CLASSES + preds[k].ravel()).astype(np.int64)
            sf  = surf[k].ravel()
            eb  = bin_index(elev_mm[idx].astype(np.float32).ravel(),  ELEV_EDGES)
            sb  = bin_index(slope_mm[idx].astype(np.float32).ravel(), SLOPE_EDGES)

            c_surf_elev += np.bincount((sf * N_ELEV + eb) * 16 + yp,
                                       minlength=N_SURF * N_ELEV * 16
                                       ).reshape(N_SURF, N_ELEV, 4, 4)
            c_slope     += np.bincount(sb * 16 + yp,
                                       minlength=N_SLOPE * 16).reshape(N_SLOPE, 4, 4)
            c_slope_surf += np.bincount((sb * N_SURF + sf) * 16 + yp,
                                        minlength=N_SLOPE * N_SURF * 16
                                        ).reshape(N_SLOPE, N_SURF, 4, 4)

    total = N_TEST * 512 * 512
    assert c_surf_elev.sum() == total,  "lost pixels in surf x elev"
    assert c_slope.sum() == total,      "lost pixels in slope"
    assert c_slope_surf.sum() == total, "lost pixels in slope x surf"

    del model
    torch.cuda.empty_cache()
    return dict(epoch=epoch,
                conf_surf_elev=c_surf_elev.tolist(),
                conf_slope=c_slope.tolist(),
                conf_slope_surf=c_slope_surf.tolist())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    elev_mm  = np.memmap(f"{DATA_ROOT}\\Test\\EXTRA_elevation.dat",
                         dtype="int16", mode="r", shape=(N_TEST, 512, 512))
    slope_mm = np.memmap(f"{DATA_ROOT}\\Test\\DERIVED_slope_v2.dat",
                         dtype="float32", mode="r", shape=(N_TEST, 512, 512))

    path = os.path.join(OUT_DIR, f"joint_stratification{OUT_SUFFIX}.json")   # separate file for final-epoch
    out = dict(class_names=CLASS_NAMES, surf_labels=SURF_LABELS,
               elev_labels=ELEV_LABELS, slope_labels=SLOPE_LABELS, models={})
    if os.path.exists(path):                      # resume a killed run
        out = json.load(open(path))
        print(f"resuming — {len(out['models'])} model(s) already done\n")

    for spec in MODELS:
        if spec["name"] in out["models"]:
            continue
        ck = os.path.join(EXP, spec["dir"], CKPT_FILE)   # see CKPT_FILE
        if not os.path.exists(ck):
            print(f"SKIP {spec['name']} — no checkpoint")
            continue
        if training_unfinished(spec):   # never score a model mid-training
            print(f"SKIP {spec['name']} — training not finished")
            continue
        out["models"][spec["name"]] = evaluate_joint(spec, device, elev_mm, slope_mm)
        with open(path, "w") as f:                # incremental: survives a kill
            json.dump(out, f)
        print(f"done {spec['name']:28s}  (written)")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
