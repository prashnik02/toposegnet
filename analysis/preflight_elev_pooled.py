"""
Pre-flight for the `elev_pooled` control.

Why this control exists
-----------------------
gap_terrain beat concat, but it differs from concat in TWO ways at once:
  (a) elevation is POOLED to a scene-level summary rather than per-pixel, and
  (b) it arrives via a SEPARATE BRANCH rather than as an encoder input channel.
`elev_pooled` holds (b) fixed — still a concatenated 14th input channel, still a
plain U-Net, arch="unet" — and varies only (a). If it recovers gap's gain, the
finding is about POOLING. If it does not, the separate branch is doing the work.

Checks
  1. channel count is 14, identical to `elev`
  2. the 13 spectral bands are BYTE-IDENTICAL to `elev` (so nothing else moved)
  3. the elevation plane is CONSTANT within each patch (that's the whole point)
  4. that constant equals the patch mean of the `elev` plane, to float precision
  5. it VARIES across patches (a constant-everywhere channel would be inert)
  6. no NaN/Inf, and the value range is sane after standardisation
"""
import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import numpy as np
import torch
from dataset import CloudSEN12Dataset

print("Building Test datasets for `elev` and `elev_pooled` ...")
ds_e = CloudSEN12Dataset(split="Test", terrain_mode="elev",        augment=False)
ds_p = CloudSEN12Dataset(split="Test", terrain_mode="elev_pooled", augment=False)

IDX = [0, 1, 7, 42, 100, 500, 974]
means = []

for k, i in enumerate(IDX):
    xe, ye = ds_e[i]
    xp, yp = ds_p[i]

    if k == 0:
        assert xe.shape[0] == 14 and xp.shape[0] == 14, (xe.shape, xp.shape)
        print(f"1. channels          OK   elev {tuple(xe.shape)}  elev_pooled {tuple(xp.shape)}")

    # 2. spectral bands untouched
    d_bands = (xe[:13] - xp[:13]).abs().max().item()
    assert d_bands == 0.0, f"patch {i}: bands differ by {d_bands}"

    # labels must match too
    assert torch.equal(ye, yp), f"patch {i}: labels differ"

    ep, pp = xe[13], xp[13]

    # 3. pooled plane is constant within the patch
    spread = (pp.max() - pp.min()).item()
    assert spread == 0.0, f"patch {i}: pooled plane not constant (spread {spread:.3e})"

    # 4. that constant == the patch mean of the per-pixel plane
    d_mean = abs(pp[0, 0].item() - ep.mean().item())
    assert d_mean < 1e-4, f"patch {i}: pooled {pp[0,0].item():.6f} != mean {ep.mean().item():.6f}"

    # 6. finite + sane
    assert torch.isfinite(xp).all(), f"patch {i}: non-finite value"
    means.append(pp[0, 0].item())

print(f"2. bands byte-identical OK   max|diff| = 0.0 across {len(IDX)} patches")
print(f"3. plane constant     OK   within-patch spread = 0.0 everywhere")
print(f"4. equals patch mean  OK   max|pooled - mean(elev)| < 1e-4")

m = np.array(means)
assert m.std() > 1e-3, "pooled channel does not vary across patches — it would be inert"
print(f"5. varies across patches OK  std {m.std():.4f}, range [{m.min():.4f}, {m.max():.4f}] (standardised)")
print(f"6. finite + sane      OK   in metres: [{m.min()*1025.689+683.63:.0f}, {m.max()*1025.689+683.63:.0f}]")

print("\nALL PRE-FLIGHT CHECKS PASSED — elev_pooled is safe to launch.")
