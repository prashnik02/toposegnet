"""
Pre-flight for the `gap_terrain` ablation (arch="gap").


Checks, in order of what would silently ruin the experiment:

 1. SHAPES               forward works on the real 14ch `elev` tensor.
 2. ZERO-INIT IDENTITY   output == the 13-band baseline exactly at step 0.
                         Every conditioning variant in this chain starts as a
                         bit-for-bit copy of 1a; losing it reintroduces
                         init-shock as a confound.
 3. PATH IS LIVE         a dead terrain path would ALSO give a perfect identity,
                         so identity alone proves nothing. Perturb out_proj and
                         confirm the output actually moves, per level.
 4. ELEVATION-RESPONSIVE with trained-like weights, changing the DEM must change
                         the output. If it does not, the model cannot be using
                         terrain at all.
 5. UNIFORM-ATTENTION EQUIVALENCE
                         the whole scientific claim of this ablation is that GAP
                         == TCM with A[i,j] = 1/N. Verify numerically against a
                         TCM block whose attention is forced uniform.
 6. PARAM BUDGET         vs TCM and vs the 1a baseline.
 7. REAL VRAM            at the training batch size on this 6 GB card.
"""
import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from model import build_gap_terrain_unet, build_tcm_unet, GAPTerrainModule
from tcm import TerrainConditioningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")
torch.manual_seed(0)

# ── 1. shapes ───────────────────────────────────────────────────────────────
model = build_gap_terrain_unet(in_channels=13, num_classes=4, pretrained=True).to(device).eval()
x = torch.randn(2, 14, 512, 512, device=device)
with torch.no_grad():
    y = model(x)
assert y.shape == (2, 4, 512, 512), y.shape
print(f"1. shapes            OK   in {tuple(x.shape)} -> out {tuple(y.shape)}")

# ── 2. zero-init identity vs the plain 13-band U-Net ────────────────────────
base = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                in_channels=13, classes=4, activation=None).to(device).eval()
base.load_state_dict(model.unet.state_dict())
with torch.no_grad():
    d = (model(x) - base(x[:, :13])).abs().max().item()
print(f"2. zero-init identity {'OK' if d == 0.0 else 'FAIL'}   max|gap - baseline| = {d:.3e}")
assert d == 0.0, "zero-init identity broken — would confound the comparison"

# ── 3. each conditioning level is genuinely wired in ────────────────────────
with torch.no_grad():
    ref = model(x).clone()
    for lvl in model.cond_levels:
        w = model.gap[str(lvl)].out_proj.weight
        saved = w.clone()
        w.normal_(0, 0.05)
        delta = (model(x) - ref).abs().max().item()
        w.copy_(saved)
        print(f"3. level {lvl} live       {'OK' if delta > 1e-4 else 'FAIL'}   "
              f"perturbation moves output by {delta:.4f}")
        assert delta > 1e-4, f"level {lvl} is a dead path"

# ── 4. the output actually depends on the DEM ───────────────────────────────
with torch.no_grad():
    for lvl in model.cond_levels:                      # give it trained-like weights
        model.gap[str(lvl)].out_proj.weight.normal_(0, 0.05)
    x2 = x.clone()
    x2[:, 13] = torch.randn_like(x2[:, 13]) * 3.0      # different elevation field
    d_elev = (model(x2) - model(x)).abs().max().item()
print(f"4. elevation matters {'OK' if d_elev > 1e-4 else 'FAIL'}   "
      f"changing the DEM moves output by {d_elev:.4f}")
assert d_elev > 1e-4, "output ignores elevation"

# ── 5. GAP == TCM with uniform attention (the scientific premise) ───────────
torch.manual_seed(1)
img_ch, ter_ch, H = 256, 64, 32
img = torch.randn(2, img_ch, H, H, device=device)
ter = torch.randn(2, ter_ch, H, H, device=device)

gap = GAPTerrainModule(img_ch, ter_ch, d_model=128).to(device).eval()
tcm = TerrainConditioningModule(img_ch, ter_ch, d_model=128, num_heads=4,
                                spatial_size=(H, H)).to(device).eval()
# share the two weights the comparison depends on, and give out_proj a real value
with torch.no_grad():
    tcm.to_v.load_state_dict(gap.to_v.state_dict())
    gap.out_proj.weight.normal_(0, 0.05)
    tcm.out_proj.load_state_dict(gap.out_proj.state_dict())
    tcm.ffn.load_state_dict(gap.ffn.state_dict())
    tcm.norm2.load_state_dict(gap.norm2.state_dict())

    # force TCM's attention uniform: out_i = mean_j V_j
    V = tcm.to_v(ter)
    uniform_out = V.mean(dim=(2, 3), keepdim=True).expand_as(V)
    tcm_uniform = img + tcm.out_proj(uniform_out)
    tcm_uniform = tcm_uniform + tcm.ffn(tcm.norm2(tcm_uniform))

    gap_out = gap(img, ter)
    d_eq = (gap_out - tcm_uniform).abs().max().item()
print(f"5. GAP == uniform-A TCM {'OK' if d_eq < 1e-5 else 'FAIL'}   max diff {d_eq:.3e}")
assert d_eq < 1e-5, "GAP is not the uniform-attention limit — premise broken"

# ── 6. parameter budget ─────────────────────────────────────────────────────
n_gap  = sum(p.numel() for p in model.parameters())
n_tcm  = sum(p.numel() for p in build_tcm_unet(pretrained=False).parameters())
n_base = sum(p.numel() for p in base.parameters())
print(f"6. params            gap {n_gap:,}   tcm {n_tcm:,}   baseline(13ch) {n_base:,}")
print(f"                     gap vs baseline {100*(n_gap-n_base)/n_base:+.2f}%   "
      f"gap vs tcm {100*(n_gap-n_tcm)/n_tcm:+.2f}%")

# ── 7. real VRAM at training batch size ─────────────────────────────────────
if device.type == "cuda":
    del base
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    xb = torch.randn(4, 14, 512, 512, device=device)
    yb = torch.randint(0, 4, (4, 512, 512), device=device)
    loss = nn.CrossEntropyLoss()(model(xb), yb)
    loss.backward(); opt.step()
    print(f"7. VRAM peak         {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
          f"at batch=4 (card has 6 GB)")

print("\nALL PRE-FLIGHT CHECKS PASSED — gap_terrain is safe to launch.")
