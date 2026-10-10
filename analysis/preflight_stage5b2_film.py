"""
Pre-flight for Stage 5b(2): FiLM lat/lon bottleneck conditioning.


Verifies, before burning ~26h of GPU:
  1. shapes are right end-to-end
  2. ZERO-INIT IDENTITY: at step 0 FiLMUNet == plain U-Net on the 13 bands
  3. the conditioning path is LIVE (not dead code that only *looks* identity)
  4. lat/lon scalars recovered from the constant planes match the dataset
  5. parameter overhead vs the 5b(1) concat model
"""
import sys
sys.path.insert(0, r"C:\Users\prash\toposegnet\src")

import torch
import numpy as np
import segmentation_models_pytorch as smp

from model import build_film_unet
from dataset import CloudSEN12Dataset

torch.manual_seed(0)
np.random.seed(0)

print("=" * 68)
print("STAGE 5b(2) PRE-FLIGHT — FiLM lat/lon bottleneck conditioning")
print("=" * 68)

# ── 1. shapes ────────────────────────────────────────────────────────────────
model = build_film_unet(in_channels=13, num_classes=4, pretrained=False).eval()
x = torch.randn(2, 15, 512, 512)      # 13 bands + 2 constant lat/lon planes
x[:, 13] = 0.4771                     # lat plane (constant, as dataset builds it)
x[:, 14] = -0.3890                    # lon plane

with torch.no_grad():
    y = model(x)
print(f"\n[1] input  {tuple(x.shape)} -> output {tuple(y.shape)}")
assert y.shape == (2, 4, 512, 512), "output shape wrong"
print("    OK: (B,4,512,512) as expected")

# ── 2. zero-init identity ────────────────────────────────────────────────────
# FiLMUNet at init must equal running its own inner U-Net on the bands alone.
bands = x[:, :13]
with torch.no_grad():
    feats = list(model.unet.encoder(bands))
    baseline_out = model.unet.segmentation_head(model.unet.decoder(feats))
    film_out = model(x)

max_diff = (baseline_out - film_out).abs().max().item()
print(f"\n[2] zero-init identity: max|FiLM - baseline| = {max_diff:.3e}")
assert max_diff == 0.0, f"NOT identity at init (diff={max_diff})"
print("    OK: EXACT identity — training starts as the untouched baseline")

# gamma/beta must literally be zero
with torch.no_grad():
    cond = x[:, 13:, 0, 0]
    g, b = model.film.mlp(cond).chunk(2, dim=1)
print(f"    gamma: max|g|={g.abs().max().item():.3e} | beta: max|b|={b.abs().max().item():.3e}")

# ── 3. the conditioning path is LIVE ─────────────────────────────────────────
# Perturb the zero-inited layer: output must now (a) differ from baseline and
# (b) RESPOND to cond. Without this, a dead path would pass test [2] silently.
import copy
probe = copy.deepcopy(model)
with torch.no_grad():
    torch.nn.init.normal_(probe.film.mlp[-1].weight, std=0.05)
    torch.nn.init.normal_(probe.film.mlp[-1].bias, std=0.05)

    out_A = probe(x)                       # lat/lon = (0.4771, -0.3890)
    x_B = x.clone()
    x_B[:, 13] = -0.8                      # different location
    x_B[:, 14] = 0.6
    out_B = probe(x_B)

d_base = (out_A - baseline_out).abs().max().item()
d_cond = (out_A - out_B).abs().max().item()
print(f"\n[3] path liveness (after perturbing FiLM weights):")
print(f"    max|perturbed - baseline| = {d_base:.4f}  (must be > 0)")
print(f"    max|cond_A   - cond_B   | = {d_cond:.4f}  (must be > 0)")
assert d_base > 0, "FiLM has no effect — path is dead"
assert d_cond > 0, "output does not respond to lat/lon — cond is not wired"
print("    OK: conditioning genuinely reaches the output and depends on lat/lon")

# ── 4. lat/lon recovery from the real dataset ────────────────────────────────
ds = CloudSEN12Dataset(split="Val", terrain_mode="latlon_only", augment=False)
img, _ = ds[0]
img = img.unsqueeze(0)
cond_recovered = img[:, 13:, 0, 0]
plane_lat_unique = torch.unique(img[0, 13]).tolist()
plane_lon_unique = torch.unique(img[0, 14]).tolist()
print(f"\n[4] real dataset patch (Val[0]):")
print(f"    dataset channels     : {img.shape[1]} (expect 15)")
print(f"    lat plane unique vals: {plane_lat_unique}  (must be a single value)")
print(f"    lon plane unique vals: {plane_lon_unique}")
print(f"    recovered cond       : {cond_recovered.tolist()}")
assert len(plane_lat_unique) == 1 and len(plane_lon_unique) == 1, \
    "lat/lon planes are NOT constant — corner-pixel extraction would be wrong"
print(f"    -> implied lat={plane_lat_unique[0]*90:.3f} deg, "
      f"lon={plane_lon_unique[0]*180:.3f} deg")
print("    OK: planes constant, scalar recovery exact")

# ── 5. parameter overhead ────────────────────────────────────────────────────
film_total = sum(p.numel() for p in model.parameters())
film_only  = sum(p.numel() for p in model.film.parameters())
concat_5b1 = sum(p.numel() for p in smp.Unet(
    encoder_name="resnet34", encoder_weights=None, in_channels=15, classes=4).parameters())
print(f"\n[5] parameters:")
print(f"    5b(1) concat U-Net (15ch): {concat_5b1:,}")
print(f"    5b(2) FiLM-UNet    (13ch): {film_total:,}  (FiLM MLP: {film_only:,})")
print(f"    delta                    : {film_total - concat_5b1:+,} "
      f"({100*(film_total-concat_5b1)/concat_5b1:+.3f}%)")

print("\n" + "=" * 68)
print("ALL PRE-FLIGHT CHECKS PASSED — safe to launch Stage 5b(2)")
print("=" * 68)
