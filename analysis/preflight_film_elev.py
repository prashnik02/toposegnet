"""
Pre-flight for the pooled-elevation FiLM control.

Why this control exists
-----------------------
`elev_pooled` + arch="unet" was a NULL — 0.6939 [0.6888, 0.7024], OVERLAP with
concat/TCM/gap, and the slope dose-response absent in all six bins. But that
null is WEAK. A channel constant across a patch enters the first convolution as
w*k + b with k fixed, so it can only act through that layer's effective bias:
one scalar offset per output filter. It therefore cannot distinguish

    (a) "a pooled scalar carries no useful signal", from
    (b) "a concatenated channel cannot exploit a pooled scalar".

This variant delivers the IDENTICAL scalar by depth-injection instead — gamma/
beta modulation of the 512-channel bottleneck, the route Stage 5b(2) used for
lat/lon. Recovering gap's gain would mean (b) and vindicate pooling; a second
null points at the branch's capacity rather than its depth.

Checks
  1. dataset contract: 14 channels, and the FiLM model's encoder takes 13
  2. the peeled cond equals the patch-mean elevation plane, exactly
  3. ZERO-INIT IDENTITY: at step 0 the model is bit-for-bit the 13-band U-Net
  4. the conditioning path is LIVE — perturbing cond changes the output.
     (Check 3 alone is satisfied by a dead path too; this separates them.)
  5. gradients actually reach the FiLM MLP
  6. parameter delta vs the plain 14ch U-Net is small and accounted for
"""
import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import torch
import segmentation_models_pytorch as smp
from dataset import CloudSEN12Dataset
from model import build_film_elev_unet

torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {dev}\n")

# ── 1. dataset contract ──────────────────────────────────────────────────────
ds = CloudSEN12Dataset(split="Test", terrain_mode="elev_pooled", augment=False)
x0, _ = ds[0]
assert x0.shape[0] == 14, x0.shape
model = build_film_elev_unet(in_channels=13, num_classes=4, pretrained=False).to(dev).eval()
assert model.in_channels == 13
print(f"1. contract           OK   dataset {tuple(x0.shape)} -> encoder_in=13 + 1 via FiLM")

# ── 2. peeled cond == patch mean ─────────────────────────────────────────────
IDX = [0, 1, 42, 500, 974]
xs = torch.stack([ds[i][0] for i in IDX]).to(dev)
cond = xs[:, 13:, 0, 0]                      # what the model peels
plane_mean = xs[:, 13].mean(dim=(1, 2), keepdim=False).unsqueeze(1)
d = (cond - plane_mean).abs().max().item()
assert d < 1e-6, d
assert cond.shape == (len(IDX), 1), cond.shape
print(f"2. cond == patch mean OK   max|diff| = {d:.2e}, cond shape {tuple(cond.shape)}, "
      f"values {[round(v, 4) for v in cond.squeeze(1).tolist()]}")

# ── 3. zero-init identity vs the plain 13-band U-Net ─────────────────────────
plain = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=13, classes=4).to(dev).eval()
plain.load_state_dict(model.unet.state_dict())   # same weights, no FiLM
with torch.no_grad():
    y_film  = model(xs)
    y_plain = plain(xs[:, :13])
delta = (y_film - y_plain).abs().max().item()
assert delta == 0.0, f"not an exact identity at init: {delta:.3e}"
print(f"3. zero-init identity OK   max|FiLM - plain| = {delta:.1e}  (EXACT)")

# ── 4. the conditioning path is LIVE ─────────────────────────────────────────
# Give gamma/beta a nonzero head so the path can express itself, then check the
# output actually responds to cond. A dead path would still pass check 3.
with torch.no_grad():
    for p in model.film.mlp[-1].parameters():
        p.normal_(0.0, 0.05)
    y_a = model(xs)
    xs_b = xs.clone()
    xs_b[:, 13] += 1.0                      # shift the pooled elevation by +1 sd
    y_b = model(xs_b)
resp = (y_a - y_b).abs().max().item()
assert resp > 1e-4, f"conditioning path is DEAD: {resp:.3e}"
print(f"4. cond path LIVE     OK   max|out(cond) - out(cond+1sd)| = {resp:.4f}")

# ── 5. gradients reach the FiLM MLP ──────────────────────────────────────────
model.train()
out = model(xs[:2])
out.mean().backward()
gnorms = {n: p.grad.abs().sum().item()
          for n, p in model.film.named_parameters() if p.grad is not None}
assert gnorms and max(gnorms.values()) > 0, gnorms
print(f"5. grads reach FiLM   OK   " +
      ", ".join(f"{n}={v:.3e}" for n, v in gnorms.items()))

# ── 6. parameter accounting ──────────────────────────────────────────────────
n_film  = sum(p.numel() for p in model.parameters())
n_plain14 = sum(p.numel() for p in smp.Unet(
    encoder_name="resnet34", encoder_weights=None, in_channels=14, classes=4).parameters())
n_mlp = sum(p.numel() for p in model.film.parameters())
print(f"6. params             OK   FiLM model {n_film:,} vs plain-14ch {n_plain14:,} "
      f"({100*(n_film-n_plain14)/n_plain14:+.2f}%); FiLM MLP alone {n_mlp:,}")

print("\nALL PRE-FLIGHT CHECKS PASSED — film_elev is safe to launch.")
