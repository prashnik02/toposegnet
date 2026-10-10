"""
TCM attention-map analysis — WHY does cross-attention help thick cloud over snow?
================================================================================


Why this exists
---------------
The multi-seed campaign (2026-08-07) established ONE replicated effect: routing
elevation through cross-attention instead of concatenating it improves
thick-cloud detection over snow (recall +0.0587, IoU +0.0345, disjoint seed
ranges across n=3 per side). That is a measured effect with no mechanism.
A reviewer's first question is "why", and "attention helps" is not an answer.

The hypothesis on record (project notes, Stage 1a error analysis) is that elevation
acts as a SNOWLINE PRIOR: over snow, spectral bands alone confuse cloud with
snow, and elevation tells the network which bright pixels are plausibly snow.
If true, the attention should be (a) elevation-selective and (b) NON-LOCAL —
because a pixel needs the surrounding terrain context, not just its own height,
which is exactly what FiLM (strictly per-pixel) cannot express.

Four tests, in increasing order of how much they would prove
------------------------------------------------------------
T1  IS THE CONDITIONING USED AT ALL?
    out_proj and the last FFN conv are zero-init, so any weight norm is earned.
    Also measures ||residual|| / ||img_feat||: how much terrain actually moves
    the features. A near-zero ratio means the model learned to ignore terrain
    and the whole effect is something else.

T2  IS ATTENTION ELEVATION-SELECTIVE?
    Per key position j, attention mass received a_j = mean_i A[i,j]. Spearman
    rho(a_j, elevation_j) per patch. Null = 0. If TCM has learned "look at high
    terrain", rho > 0 over snow patches.

T3  IS IT NON-LOCAL?  <-- the test that separates TCM from FiLM
    Attended elevation E_att(i) = sum_j A[i,j] * elev_j.
    - If E_att(i) ~ elev(i), attention is degenerate/local: FiLM could do this
      and the architectural argument collapses.
    - If E_att(i) ~ mean(elev), attention is uniform: TCM reduces to injecting a
      global terrain summary — still not FiLM, but a much weaker claim.
    - If E_att(i) tracks neither, attention is genuinely selecting context.
    Reported as correlations of E_att against BOTH references.

T4  DOES IT DISCRIMINATE THE CLASS WHERE THE EFFECT LIVES?
    Restricted to snow patches: compare E_att for query cells that are truly
    THICK CLOUD vs truly CLEAR. The measured effect is thick-over-snow, so if
    the mechanism is real those two query populations should attend to
    different elevations. Same comparison on non-snow patches is the control:
    the effect is snow-specific, so the gap should be smaller there.

Notes
-----
* Attention is recomputed EXPLICITLY here (softmax(QK^T/sqrt(d))). The model
  uses F.scaled_dot_product_attention, which never materialises A and so cannot
  be inspected. Maths is identical; this just costs memory, which is fine at
  32x32 (1024 tokens) and 16x16 (256 tokens) with batch 1.
* All correlations are computed in TORCH, not numpy. np.corrcoef / np.dot hit
  the Windows BLAS delay-load fault documented in project notes for this env.
* Elevation is de-standardised back to metres (mean 683.63, std 1025.69) so the
  numbers are physically readable.
* Run over all 3 TCM seeds; a mechanism that only appears in one seed is not a
  mechanism, which is the lesson of the campaign this follows.

Run:  python analysis/tcm_attention_analysis.py
      (conda env `toposegnet`; ~6 min on GPU)
"""

import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from dataset import CloudSEN12Dataset
from model import build_tcm_unet

EXP = r"C:\Users\prash\toposegnet\experiments"
OUT = os.path.join(EXP, "tcm_attention_analysis")

SEEDS = [
    ("TCM seed0", "tcm_elev_augFalse"),
    ("TCM seed1", "tcm_elev_augFalse_seed1"),
    ("TCM seed2", "tcm_elev_augFalse_seed2"),
]

# must match dataset.py — de-standardise elevation to metres for readability
ELEV_MEAN, ELEV_STD = 683.63, 1025.69

# surface proxy — IDENTICAL to stage1a_terrain_error_analysis.py
NDSI_B3_IDX, NDSI_B8_IDX, NDSI_B11_IDX = 2, 7, 11
NDSI_THRESH, NIR_THRESH = 0.4, 0.11

CLASS_NAMES = ["clear", "thick", "thin", "shadow"]
N_PATCH_PER_GROUP = 100      # snow-heavy and snow-free patches to analyse
SNOW_HI, SNOW_LO  = 0.15, 0.01
CELL_PURITY       = 0.60     # a query cell must be >=60% one class to be counted


# ── correlation helpers (torch, to avoid the numpy/BLAS crash) ───────────────
def _pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = x.norm() * y.norm()
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _rank(x):
    """Average-rank transform (ties broken by order; fine for continuous data)."""
    idx = torch.argsort(x)
    r = torch.empty_like(x)
    r[idx] = torch.arange(len(x), dtype=x.dtype, device=x.device)
    return r


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


# ── explicit attention (the model uses SDPA, which hides the weights) ────────
@torch.no_grad()
def attention_weights(tcm, img_feat, terrain_feat):
    """Replicates TerrainConditioningModule's attention branch, returning A.

    Returns
      A        : (heads, HW_q, HW_k) attention weights, softmax over keys
      residual : (B,C,H,W) what the attention branch actually adds to img_feat
    """
    q_in = tcm.norm1(img_feat)
    Q = tcm.to_q(q_in)
    K = tcm.to_k(terrain_feat)
    V = tcm.to_v(terrain_feat)
    if tcm.pos_k is not None:
        K = K + tcm.pos_k

    B, _, H, W = img_feat.shape
    Qh = rearrange(Q, 'b (h d) x y -> b h (x y) d', h=tcm.num_heads)
    Kh = rearrange(K, 'b (h d) x y -> b h (x y) d', h=tcm.num_heads)
    Vh = rearrange(V, 'b (h d) x y -> b h (x y) d', h=tcm.num_heads)

    scale = 1.0 / (Qh.shape[-1] ** 0.5)
    A = torch.softmax((Qh @ Kh.transpose(-2, -1)) * scale, dim=-1)   # (B,h,HWq,HWk)

    out = rearrange(A @ Vh, 'b h (x y) d -> b (h d) x y', x=H, y=W)
    residual = tcm.out_proj(out)

    # sanity: explicit attention must reproduce the SDPA path the model uses
    ref = rearrange(F.scaled_dot_product_attention(Qh, Kh, Vh),
                    'b h (x y) d -> b (h d) x y', x=H, y=W)
    assert (out - ref).abs().max() < 1e-3, "explicit attention != SDPA path"

    return A[0], residual


def load_seed(exp_dir, device):
    model = build_tcm_unet(in_channels=13, num_classes=4, pretrained=False)
    ckpt = torch.load(os.path.join(EXP, exp_dir, "best_model.pth"),
                      map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if (isinstance(ckpt, dict)
                                         and "model_state_dict" in ckpt) else ckpt
    model.load_state_dict(state)
    return model.to(device).eval(), (ckpt.get("epoch") if isinstance(ckpt, dict) else None)


def pick_patches(ds, device):
    """Split test patches into snow-heavy and snow-free groups."""
    fracs = []
    for i in tqdm(range(len(ds)), desc="  scanning snow fraction", leave=False):
        img, _ = ds[i]
        b3, b8, b11 = img[NDSI_B3_IDX], img[NDSI_B8_IDX], img[NDSI_B11_IDX]
        ndsi = (b3 - b11) / (b3 + b11 + 1e-8)
        fracs.append(float(((ndsi > NDSI_THRESH) & (b8 > NIR_THRESH)).float().mean()))
    fracs = np.array(fracs)
    snow = np.argsort(-fracs)[:N_PATCH_PER_GROUP]
    snow = snow[fracs[snow] >= SNOW_HI]
    ctrl = np.where(fracs <= SNOW_LO)[0][:N_PATCH_PER_GROUP]
    return snow.tolist(), ctrl.tolist(), fracs


@torch.no_grad()
def analyse(model, ds, idxs, device, level):
    """Accumulate per-patch attention statistics at one conditioning level."""
    tcm = model.tcm[str(level)]
    stride = 2 ** level                      # feats[l] is 512/2^l
    rows = []

    for i in tqdm(idxs, desc=f"    level {level}", leave=False):
        img, lab = ds[i]
        x = img.unsqueeze(0).to(device)
        bands, dem = x[:, :13], x[:, 13:14]

        feats  = list(model.unet.encoder(bands))
        tfeats = model.terrain(dem)
        A, residual = attention_weights(tcm, feats[level], tfeats[level - 1])

        H = W = 512 // stride
        N = H * W
        Am = A.mean(0)                                    # (N,N) head-averaged

        # elevation on the key grid, in metres
        elev = F.avg_pool2d(dem, stride).reshape(-1) * ELEV_STD + ELEV_MEAN

        # ── T1 residual magnitude relative to the features it perturbs ──────
        rel = float(residual.norm() / feats[level].norm())

        # ── T2 elevation selectivity of received attention mass ────────────
        mass = Am.mean(0)                                 # (N,) per key
        rho_mass = _spearman(mass, elev)

        # ── T3 locality: does each query attend to its own elevation? ──────
        e_att = Am @ elev                                 # (N,) attended elevation
        rho_local  = _spearman(e_att, elev)               # vs own-cell elevation
        spread_att = float(e_att.std())                   # 0 => uniform attention
        spread_ter = float(elev.std())

        # entropy of each query's key distribution, normalised (1 = uniform)
        ent = float((-(Am * (Am + 1e-12).log()).sum(1) / np.log(N)).mean())

        # ── T4 class-conditional attended elevation on the query grid ──────
        lab_1h = F.one_hot(lab.long(), 4).permute(2, 0, 1).float().unsqueeze(0)
        cell = F.avg_pool2d(lab_1h.to(device), stride)[0].reshape(4, -1)  # (4,N)
        pure = cell.max(0)
        cls_of_cell = torch.where(pure.values >= CELL_PURITY, pure.indices,
                                  torch.full_like(pure.indices, -1))

        per_class = {}
        for c, name in enumerate(CLASS_NAMES):
            m = cls_of_cell == c
            if int(m.sum()) >= 4:
                per_class[name] = dict(n=int(m.sum()),
                                       e_att=float(e_att[m].mean()),
                                       e_own=float(elev[m].mean()))

        rows.append(dict(patch=int(i), rel_residual=rel, rho_mass=rho_mass,
                         rho_local=rho_local, entropy=ent,
                         spread_att=spread_att, spread_terrain=spread_ter,
                         elev_mean=float(elev.mean()), per_class=per_class))
    return rows


def summarise(rows, key):
    v = np.array([r[key] for r in rows if not np.isnan(r[key])], dtype=np.float64)
    return dict(mean=float(v.mean()), std=float(v.std()), n=int(len(v)))


def main():
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ds = CloudSEN12Dataset(split="Test", terrain_mode="elev", augment=False)
    snow_idx, ctrl_idx, fracs = pick_patches(ds, device)
    print(f"snow-heavy patches : {len(snow_idx)}  (snow frac >= {SNOW_HI})")
    print(f"snow-free  patches : {len(ctrl_idx)}  (snow frac <= {SNOW_LO})\n")

    results = {}
    for name, d in SEEDS:
        if not os.path.exists(os.path.join(EXP, d, "best_model.pth")):
            print(f"SKIP {name} — no checkpoint")
            continue
        model, epoch = load_seed(d, device)

        # T1 — learned weight norms (zero-init, so any norm is earned)
        wnorms = {}
        for lvl in model.cond_levels:
            t = model.tcm[str(lvl)]
            wnorms[f"L{lvl}"] = dict(
                out_proj = float(t.out_proj.weight.norm()),
                ffn_last = float(t.ffn[-1].weight.norm()),
                to_k     = float(t.to_k.weight.norm()),
                to_q     = float(t.to_q.weight.norm()),
                pos_k    = float(t.pos_k.norm()) if t.pos_k is not None else None,
            )

        per_level = {}
        for lvl in model.cond_levels:
            per_level[f"L{lvl}"] = dict(
                snow = analyse(model, ds, snow_idx, device, lvl),
                ctrl = analyse(model, ds, ctrl_idx, device, lvl),
            )
        results[name] = dict(dir=d, epoch=epoch, weight_norms=wnorms,
                             levels=per_level)
        print(f"done {name}")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, "attention_stats.json"), "w") as f:
        json.dump(dict(seeds=results, snow_idx=snow_idx, ctrl_idx=ctrl_idx,
                       snow_frac=fracs.tolist(), cell_purity=CELL_PURITY), f)

    # ── report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("T1  IS THE CONDITIONING USED?   (zero-init at step 0 => norms are earned)")
    print("=" * 78)
    for name, r in results.items():
        for lvl, w in r["weight_norms"].items():
            print(f"  {name:11s} {lvl}  out_proj {w['out_proj']:7.3f}   "
                  f"ffn_last {w['ffn_last']:7.3f}   pos_k {w['pos_k']:7.3f}")
    print("\n  residual magnitude ||attn_out|| / ||img_feat||")
    for name, r in results.items():
        for lvl, d_ in r["levels"].items():
            s = summarise(d_["snow"], "rel_residual")
            c = summarise(d_["ctrl"], "rel_residual")
            print(f"  {name:11s} {lvl}  snow {s['mean']:.4f}+-{s['std']:.4f}   "
                  f"ctrl {c['mean']:.4f}+-{c['std']:.4f}")

    print("\n" + "=" * 78)
    print("T2  ELEVATION SELECTIVITY   Spearman(attention mass received, elevation)")
    print("=" * 78)
    for name, r in results.items():
        for lvl, d_ in r["levels"].items():
            s = summarise(d_["snow"], "rho_mass")
            c = summarise(d_["ctrl"], "rho_mass")
            print(f"  {name:11s} {lvl}  snow rho {s['mean']:+.4f}+-{s['std']:.4f}   "
                  f"ctrl rho {c['mean']:+.4f}+-{c['std']:.4f}")

    print("\n" + "=" * 78)
    print("T3  LOCALITY   rho(attended elev, own elev);  entropy 1.0 = uniform")
    print("=" * 78)
    for name, r in results.items():
        for lvl, d_ in r["levels"].items():
            s_r = summarise(d_["snow"], "rho_local")
            s_e = summarise(d_["snow"], "entropy")
            s_a = summarise(d_["snow"], "spread_att")
            s_t = summarise(d_["snow"], "spread_terrain")
            print(f"  {name:11s} {lvl}  snow: rho_local {s_r['mean']:+.4f}   "
                  f"entropy {s_e['mean']:.4f}   "
                  f"sd(attended) {s_a['mean']:7.2f} m vs sd(terrain) {s_t['mean']:7.2f} m")

    print("\n" + "=" * 78)
    print("T4  CLASS-CONDITIONAL ATTENDED ELEVATION (metres)")
    print("     does thick-cloud attend differently from clear? effect lives here")
    print("=" * 78)
    for name, r in results.items():
        for lvl, d_ in r["levels"].items():
            for grp in ("snow", "ctrl"):
                acc = {}
                for row in d_[grp]:
                    for cname, v in row["per_class"].items():
                        acc.setdefault(cname, []).append(v["e_att"])
                parts = [f"{c} {np.mean(v):7.1f}(n={len(v):3d})"
                         for c, v in acc.items() if len(v) >= 5]
                gap = ""
                if "thick" in acc and "clear" in acc and \
                   len(acc["thick"]) >= 5 and len(acc["clear"]) >= 5:
                    gap = f"   thick-clear = {np.mean(acc['thick']) - np.mean(acc['clear']):+7.1f} m"
                print(f"  {name:11s} {lvl} {grp:4s}  " + "  ".join(parts) + gap)

    print(f"\nwrote {os.path.join(OUT, 'attention_stats.json')}")


if __name__ == "__main__":
    main()
