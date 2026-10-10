"""
Pre-flight for gap + lat/lon bottleneck FiLM (arch="gap_latlon").

Question: once terrain is already injected at depth (gap), does LOCATION add
anything? Exploratory single seed. Lat/lon goes through FiLM, not attention:
a per-patch constant added to the attention keys shifts every logit of a query
by the same amount and softmax cancels it, so attention cannot see it at all.

Checks
  1. dataset contract: 16 channels; encoder takes 13
  2. elevation plane == `elev` mode's plane, lat/lon planes == `latlon_only`'s
     (the features are byte-identical to the chain's existing inputs)
  3. cond peeled at (0,0) is exactly the constant lat/lon planes
  4. ZERO-INIT IDENTITY vs a gap model built under the SAME seed: every gap
     weight matches bit-for-bit and the outputs are EXACTLY equal at step 0
  5. the lat/lon path is LIVE (check 4 alone is satisfied by a dead path too)
  6. gradients reach the FiLM MLP
  7. parameter delta vs gap is exactly the FiLM MLP
"""
import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import torch
from dataset import CloudSEN12Dataset
from model import build_gap_latlon_unet, build_gap_terrain_unet

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {dev}\n")
IDX = [0, 1, 42, 500, 974]

# ── 1. dataset contract ──────────────────────────────────────────────────────
ds = CloudSEN12Dataset(split="Test", terrain_mode="elev_latlon", augment=False)
x0, _ = ds[0]
assert x0.shape[0] == 16, x0.shape
print(f"1. contract           OK   dataset {tuple(x0.shape)} -> encoder_in=13 + elev(gap) + lat/lon(FiLM)")

# ── 2. features byte-identical to existing modes ─────────────────────────────
ds_e  = CloudSEN12Dataset(split="Test", terrain_mode="elev",        augment=False)
ds_ll = CloudSEN12Dataset(split="Test", terrain_mode="latlon_only", augment=False)
for i in IDX:
    x, xe, xll = ds[i][0], ds_e[i][0], ds_ll[i][0]
    assert torch.equal(x[:14], xe),          f"patch {i}: bands+elev differ from `elev`"
    assert torch.equal(x[14:], xll[13:]),    f"patch {i}: lat/lon differ from `latlon_only`"
    assert torch.isfinite(x).all(),          f"patch {i}: non-finite values"
print(f"2. features identical OK   bands+elev == `elev`, lat/lon == `latlon_only` on {len(IDX)} patches (max diff 0.0)")

# ── 3. peeled cond == constant planes ────────────────────────────────────────
xs = torch.stack([ds[i][0] for i in IDX]).to(dev)
cond = xs[:, 14:16, 0, 0]
spread = (xs[:, 14:16].amax(dim=(2, 3)) - xs[:, 14:16].amin(dim=(2, 3))).abs().max().item()
assert spread == 0.0, spread
assert cond.shape == (len(IDX), 2)
print(f"3. cond peel          OK   planes constant (spread {spread}), cond (lat,lon) = "
      f"{[tuple(round(v, 3) for v in r) for r in cond.tolist()]}")

# ── 4. zero-init identity vs gap under the same seed ─────────────────────────
torch.manual_seed(0)
model = build_gap_latlon_unet(in_channels=13, num_classes=4, pretrained=False).to(dev).eval()
torch.manual_seed(0)
gap = build_gap_terrain_unet(in_channels=13, num_classes=4, pretrained=False).to(dev).eval()
sd_m, sd_g = model.state_dict(), gap.state_dict()
extra = sorted(set(sd_m) - set(sd_g))
assert all(k.startswith("film.") for k in extra), extra
assert set(sd_g) <= set(sd_m)
mism = [k for k in sd_g if not torch.equal(sd_m[k], sd_g[k])]
assert not mism, f"gap weights differ under the same seed: {mism[:5]}"
with torch.no_grad():
    y_m = model(xs)
    y_g = gap(xs[:, :14])
delta = (y_m - y_g).abs().max().item()
assert delta == 0.0, f"not an exact identity at init: {delta:.3e}"
print(f"4. zero-init identity OK   all {len(sd_g)} gap tensors equal under seed 0; "
      f"max|gap_latlon - gap| = {delta:.1e} (EXACT)")

# ── 5. lat/lon path is LIVE ──────────────────────────────────────────────────
with torch.no_grad():
    for p in model.film.mlp[-1].parameters():
        p.normal_(0.0, 0.05)
    y_a = model(xs)
    xs_b = xs.clone()
    xs_b[:, 14:16] += 0.2                   # move every patch ~18 deg lat / 36 deg lon
    y_b = model(xs_b)
resp = (y_a - y_b).abs().max().item()
assert resp > 1e-4, f"lat/lon path is DEAD: {resp:.3e}"
print(f"5. lat/lon path LIVE  OK   max|out(cond) - out(cond+0.2)| = {resp:.4f}")

# ── 6. gradients reach the FiLM MLP ──────────────────────────────────────────
model.train()
model(xs[:2]).mean().backward()
gnorms = {n: p.grad.abs().sum().item()
          for n, p in model.film.named_parameters() if p.grad is not None}
assert gnorms and max(gnorms.values()) > 0, gnorms
print(f"6. grads reach FiLM   OK   " + ", ".join(f"{n}={v:.3e}" for n, v in gnorms.items()))

# ── 7. parameter accounting ──────────────────────────────────────────────────
n_m = sum(p.numel() for p in model.parameters())
n_g = sum(p.numel() for p in gap.parameters())
n_film = sum(p.numel() for p in model.film.parameters())
assert n_m - n_g == n_film, (n_m, n_g, n_film)
print(f"7. params             OK   gap_latlon {n_m:,} vs gap {n_g:,} "
      f"(+{n_m - n_g:,} = FiLM MLP exactly, {100*(n_m-n_g)/n_g:+.2f}%)")

print("\nALL PRE-FLIGHT CHECKS PASSED — gap_latlon is safe to launch.")
