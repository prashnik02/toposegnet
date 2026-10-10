"""
Pre-flight for the spectral-ablation follow-up (project notes ACTIVE PLAN, Phase 1).


Must pass before the 4-band runs are queued. CPU only, no training.

Checks
  (a) `rgbn` tensor == `none` tensor channels [1, 2, 3, 7] (B2, B3, B4, B8) exactly
  (b) `rgbn_elev` channel 4 == `elev` channel 13 (the elevation plane) exactly, and
      its channels 0-3 == `rgbn` exactly
  (c) shapes (4,512,512) / (5,512,512)
  (d) forward passes of unet/rgbn, unet/rgbn_elev and gap/rgbn_elev — built exactly as
      train.py builds them (ImageNet weights adapted to 4/5 inputs) — return (B,4,512,512)
  (e) the new memmap-based strata (`spectral_bands`) are BYTE-IDENTICAL to the old
      input-tensor slicing for 13-band models, so no existing stratification changes
"""
import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")
sys.path.append(r"C:\Users\prash\toposegnet\analysis")

import numpy as np
import torch
import segmentation_models_pytorch as smp
from dataset import CloudSEN12Dataset, RGBN_BAND_IDX
from model import build_gap_terrain_unet
from stratified_model_comparison import spectral_bands, NDSI_B3_IDX, NDSI_B8_IDX, NDSI_B11_IDX

rng = np.random.default_rng(0)
assert RGBN_BAND_IDX == [1, 2, 3, 7], RGBN_BAND_IDX

for split, n in [("Train", 8490), ("Test", 975)]:
    ds_none = CloudSEN12Dataset(split, "none", augment=False)
    ds_elev = CloudSEN12Dataset(split, "elev", augment=False)
    ds_rgbn = CloudSEN12Dataset(split, "rgbn", augment=False)
    ds_re   = CloudSEN12Dataset(split, "rgbn_elev", augment=False)
    for idx in rng.choice(n, 5, replace=False):
        x_none, y0 = ds_none[idx]; x_elev, _ = ds_elev[idx]
        x_rgbn, y1 = ds_rgbn[idx]; x_re, y2 = ds_re[idx]
        assert x_rgbn.shape == (4, 512, 512) and x_re.shape == (5, 512, 512), (x_rgbn.shape, x_re.shape)  # (c)
        assert torch.equal(x_rgbn, x_none[[1, 2, 3, 7]]), f"(a) {split}[{idx}]"                         # (a)
        assert torch.equal(x_re[4], x_elev[13]), f"(b) elevation {split}[{idx}]"                       # (b)
        assert torch.equal(x_re[:4], x_rgbn), f"(b) bands {split}[{idx}]"
        assert torch.equal(y0, y1) and torch.equal(y0, y2), f"labels {split}[{idx}]"
        assert torch.isfinite(x_re).all()
    print(f"(a)(b)(c) {split:5s} OK  5 random patches: rgbn == none[1,2,3,7], rgbn_elev[4] == elev[13], shapes 4/5x512x512")

# (d) forward passes, built the way train.py builds them
x5 = torch.stack([ds_re[i][0] for i in (0, 1)])            # (2,5,512,512) Test patches
with torch.no_grad():
    m_none = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=4, classes=4).eval()
    m_cat  = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=5, classes=4).eval()
    m_gap  = build_gap_terrain_unet(in_channels=4, num_classes=4, encoder_name="resnet34", pretrained=True).eval()
    outs = {"unet/rgbn": m_none(x5[:, :4]), "unet/rgbn_elev": m_cat(x5), "gap/rgbn_elev": m_gap(x5)}
for k, o in outs.items():
    assert o.shape == (2, 4, 512, 512) and torch.isfinite(o).all(), (k, o.shape)
print("(d) forward OK  " + ", ".join(f"{k} -> {tuple(o.shape)}" for k, o in outs.items()))
n_none = sum(p.numel() for p in m_none.parameters()); n_gap = sum(p.numel() for p in m_gap.parameters())
print(f"    params: unet/rgbn {n_none:,} | unet/rgbn_elev {sum(p.numel() for p in m_cat.parameters()):,} | gap/rgbn_elev {n_gap:,}")

# (e) memmap strata == old input-tensor strata for 13-band data
ds_t = CloudSEN12Dataset("Test", "none", augment=False)
for i0 in (0, 4, 488, 968):
    nb = min(4, 975 - i0)
    imgs = torch.stack([ds_t[i][0] for i in range(i0, i0 + nb)])
    b3, b8, b11 = spectral_bands(i0, nb)
    assert np.array_equal(b3, imgs[:, NDSI_B3_IDX].numpy()), i0
    assert np.array_equal(b8, imgs[:, NDSI_B8_IDX].numpy()), i0
    assert np.array_equal(b11, imgs[:, NDSI_B11_IDX].numpy()), i0
print("(e) strata OK  memmap B3/B8/B11 byte-identical to the old images[:, idx] path on 4 test batches")

print("\nALL PRE-FLIGHT CHECKS PASSED — the 4-band Phase 1 runs are safe to queue.")
