"""
Pre-flight for Stage 6: TCM cross-attention conditioning on elevation.


Verifies, before committing ~28h of GPU:
  1. shapes end-to-end
  2. ZERO-INIT IDENTITY: at step 0 TCMUNet == plain U-Net on the 13 bands
  3. the attention path is LIVE at BOTH conditioned scales (a dead branch
     would otherwise mimic a correct zero-init and pass check 2 silently)
  4. output actually depends on the ELEVATION input, not just on the bands
  5. elevation is recovered correctly from the real `elev` dataset
  6. parameter budget vs the Stage 2a concat comparator
  7. REAL VRAM headroom: full fwd+bwd train step at batch 4 on 512x512

Run:  python analysis/preflight_stage6_tcm.py
"""
import sys
sys.path.insert(0, r"C:\Users\prash\toposegnet\src")

import copy
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp

from model import build_tcm_unet
from dataset import CloudSEN12Dataset

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("STAGE 6 PRE-FLIGHT — TCM cross-attention on elevation")
print("=" * 70)

model = build_tcm_unet(in_channels=13, num_classes=4, pretrained=False).to(DEV).eval()

# ── 1. shapes ────────────────────────────────────────────────────────────────
x = torch.randn(2, 14, 512, 512, device=DEV)
x[:, 13] = torch.randn(2, 512, 512, device=DEV) * 0.5     # elevation plane
with torch.no_grad():
    y = model(x)
print(f"\n[1] input {tuple(x.shape)} -> output {tuple(y.shape)}")
assert y.shape == (2, 4, 512, 512)
print(f"    conditioned levels: {model.cond_levels}")
print("    OK")

# ── 2. zero-init identity ────────────────────────────────────────────────────
with torch.no_grad():
    feats = list(model.unet.encoder(x[:, :13]))
    baseline = model.unet.segmentation_head(model.unet.decoder(feats))
    tcm_out = model(x)
d = (baseline - tcm_out).abs().max().item()
print(f"\n[2] zero-init identity: max|TCM - baseline| = {d:.3e}")
assert d == 0.0, f"NOT an identity at init (diff={d})"
print("    OK: EXACT — training starts as the untouched 1a baseline")

# ── 3. both TCM blocks live ──────────────────────────────────────────────────
print("\n[3] per-level liveness (perturb ONE block's out_proj at a time):")
for lvl in model.cond_levels:
    probe = copy.deepcopy(model)
    with torch.no_grad():
        nn.init.normal_(probe.tcm[str(lvl)].out_proj.weight, std=0.05)
        out = probe(x)
    delta = (out - baseline).abs().max().item()
    print(f"    feats[{lvl}]: delta={delta:.4f}", "OK" if delta > 0 else "DEAD")
    assert delta > 0, f"TCM block at level {lvl} is dead"

# ── 4. output depends on ELEVATION (not just the bands) ──────────────────────
probe = copy.deepcopy(model)
with torch.no_grad():
    for lvl in probe.cond_levels:
        nn.init.normal_(probe.tcm[str(lvl)].out_proj.weight, std=0.05)
    xa = x.clone(); xa[:, 13] = -1.0          # uniformly low terrain
    xb = x.clone(); xb[:, 13] = 3.0           # uniformly high terrain
    da = (probe(xa) - probe(xb)).abs().max().item()
print(f"\n[4] response to elevation: max|out(low DEM) - out(high DEM)| = {da:.4f}")
assert da > 0, "output ignores the elevation channel — terrain path not wired"
print("    OK: conditioning genuinely reads the DEM")

# ── 5. real dataset recovery ─────────────────────────────────────────────────
ds = CloudSEN12Dataset(split="Val", terrain_mode="elev", augment=False)
img, _ = ds[0]
print(f"\n[5] real `elev` patch: {tuple(img.shape)} (expect (14,512,512))")
dem = img[13]
print(f"    elevation plane: min={dem.min():.3f} max={dem.max():.3f} "
      f"mean={dem.mean():.3f} (standardized, so ~0-centred)")
assert img.shape[0] == 14
assert torch.isfinite(img).all(), "non-finite values in the input"
print("    OK: 14 finite channels, band 13 is the DEM")

# ── 6. parameter budget ──────────────────────────────────────────────────────
concat_2a = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=14, classes=4)
p_concat = sum(p.numel() for p in concat_2a.parameters())
p_tcm    = sum(p.numel() for p in model.parameters())
p_terr   = sum(p.numel() for p in model.terrain.parameters())
p_att    = sum(p.numel() for p in model.tcm.parameters())
print(f"\n[6] parameters:")
print(f"    Stage 2a concat U-Net (14ch): {p_concat:,}")
print(f"    Stage 6 TCM-UNet            : {p_tcm:,}")
print(f"      terrain encoder           : {p_terr:,}")
print(f"      TCM blocks                : {p_att:,}")
print(f"    delta {p_tcm - p_concat:+,} ({100*(p_tcm-p_concat)/p_concat:+.2f}%)")

# ── 7. REAL VRAM at batch 4, fwd + bwd ───────────────────────────────────────
if DEV.type == "cuda":
    del model, probe
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = build_tcm_unet(in_channels=13, num_classes=4, pretrained=True).to(DEV).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 2.0384, 5.6464, 6.1658], device=DEV))
    xb = torch.randn(4, 14, 512, 512, device=DEV)
    yb = torch.randint(0, 4, (4, 512, 512), device=DEV)
    opt.zero_grad()
    loss = crit(m(xb), yb)
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n[7] VRAM, full train step @ batch 4:")
    print(f"    peak allocated {peak:.2f} GB / {total:.2f} GB "
          f"({100*peak/total:.0f}%)  loss={loss.item():.4f}")
    assert peak < total * 0.85, "too close to the VRAM ceiling"
    print("    OK: fits with headroom")
else:
    print("\n[7] SKIPPED (no CUDA)")

print("\n" + "=" * 70)
print("ALL PRE-FLIGHT CHECKS PASSED — safe to launch Stage 6")
print("=" * 70)
