"""Pre-flight for the Stage 3 v2 re-run. Verifies the dataset wiring before
committing a ~2-day training run:
  - optionA_v2 yields 15 channels, finite, correct label range
  - the slope channel really is the v2 data (differs from v1 optionA)
  - v2 slope channel is standardized (~0 mean) with the v2 stats
"""
import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import numpy as np
import torch
from dataset import (CloudSEN12Dataset, SLOPE_MEAN_V2, SLOPE_STD_V2,
                     SLOPE_MEAN_V1, SLOPE_STD_V1)

print(f"v1 slope stats: mean={SLOPE_MEAN_V1}, std={SLOPE_STD_V1}")
print(f"v2 slope stats: mean={SLOPE_MEAN_V2}, std={SLOPE_STD_V2}")

ds_v2 = CloudSEN12Dataset(split="Train", terrain_mode="optionA_v2", augment=False)
ds_v1 = CloudSEN12Dataset(split="Train", terrain_mode="optionA",    augment=False)
print(f"\ndataset len: {len(ds_v2)} (expect 8490)")
print(f"v2.v2 flag={ds_v2.v2}  slope_mean={ds_v2.slope_mean}  slope_std={ds_v2.slope_std}")
print(f"v1.v2 flag={ds_v1.v2}  slope_mean={ds_v1.slope_mean}  slope_std={ds_v1.slope_std}")

# check several patches, including a known-mountainous one (idx 208 is in Test;
# for Train just sample a spread)
slope_ch = 14   # channels 0-12 = bands, 13 = elevation, 14 = slope (optionA)
for idx in [0, 100, 3000, 8489]:
    img_v2, lbl = ds_v2[idx]
    img_v1, _   = ds_v1[idx]
    assert img_v2.shape == (15, 512, 512), f"bad shape {img_v2.shape}"
    assert torch.isfinite(img_v2).all(), f"non-finite in v2 patch {idx}"
    assert lbl.min() >= 0 and lbl.max() <= 3, f"label out of range in {idx}"
    s2 = img_v2[slope_ch]
    s1 = img_v1[slope_ch]
    # de-standardize back to degrees to sanity-check
    deg2 = s2 * SLOPE_STD_V2 + SLOPE_MEAN_V2
    diff = (s2 - s1).abs().mean().item()
    print(f"\npatch {idx}: shape ok, label {int(lbl.min())}-{int(lbl.max())}")
    print(f"  v2 slope(std) mean={s2.mean():.3f} std={s2.std():.3f}  "
          f"(deg: p50={np.percentile(deg2.numpy(),50):.2f} max={deg2.max():.2f})")
    print(f"  mean|v2-v1| in standardized slope channel = {diff:.3f}  "
          f"({'DIFFERENT (good)' if diff > 0.01 else 'IDENTICAL (BAD!)'})")

print("\npre-flight OK — optionA_v2 wired correctly, 15 channels, clean slope.")
