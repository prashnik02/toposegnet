"""
Elevation-stratified report — mining conf_elev, which has been accumulated since
July and never read.

`stratified_model_comparison.py` bins every test pixel by its raw elevation into
6 bands and stores a 4x4 confusion per band per model. Every analysis so far has
used `by_surface` (snow / water / other) and ignored `by_elevation` entirely.

Question this answers
---------------------
The campaign's one replicated effect is thick-cloud-over-snow. But snow and high
elevation are strongly correlated, so "snow effect" and "altitude effect" are
not yet distinguishable. If the concat-vs-TCM gap tracks ELEVATION rather than
snow, the mechanism story changes: it would be an altitude prior, not a
snow-context prior.

Also asks the thing nobody has checked: how does SHADOW behave with altitude?
Shadow is the class most tied to illumination geometry, so a monotone trend
would be a real finding independent of the TCM argument.

No GPU, no inference — pure re-read of the stored confusions.
"""
import json, os
import numpy as np

J = r"C:\Users\prash\toposegnet\experiments\stratified_comparison\stratified_comparison.json"
d = json.load(open(J))
CLASSES = d["class_names"]
BANDS   = d["elev_labels"]
M       = d["models"]

CONCAT = ["2a concat seed1", "2a concat seed2", "2a concat seed3"]
TCM    = ["6 elev CROSS-ATTN (TCM)", "6 TCM seed1", "6 TCM seed2"]
BASE   = "1a baseline (13ch)"


def iou_recall(conf, c):
    conf = np.asarray(conf, dtype=np.float64)
    tp = conf[c, c]
    fn = conf[c, :].sum() - tp
    fp = conf[:, c].sum() - tp
    u  = tp + fp + fn
    return (tp / u if u else np.nan,
            tp / (tp + fn) if (tp + fn) else np.nan,
            conf[c, :].sum())


def band_stat(model, bi, c, which=0):
    return iou_recall(M[model]["conf_elev"][bi], c)[which]


def band(vals, f="{:+.4f}"):
    v = np.array(vals, float)
    return f"{f.format(v.mean())} [{f.format(v.min())}, {f.format(v.max())}]"


# ── support per band ────────────────────────────────────────────────────────
print("=" * 100)
print("PIXEL SUPPORT BY ELEVATION BAND  (test set, 975 patches)")
print("=" * 100)
tot = sum(np.asarray(M[BASE]["conf_elev"][i]).sum() for i in range(len(BANDS)))
for i, b in enumerate(BANDS):
    n = np.asarray(M[BASE]["conf_elev"][i]).sum()
    sh = np.asarray(M[BASE]["conf_elev"][i])[3, :].sum()
    print(f"  {b:>12s}  {n/tot*100:6.2f}% of px   ({n/1e6:7.2f} M)   "
          f"shadow px {sh/1e6:6.3f} M ({sh/max(n,1)*100:5.2f}% of band)")

# ── per-class behaviour with altitude, baseline model ───────────────────────
print()
print("=" * 100)
print("1a BASELINE — per-class IoU by elevation band   (does altitude itself hurt?)")
print("=" * 100)
hdr = "  " + " " * 14 + "".join(f"{b:>13s}" for b in BANDS)
print(hdr)
for c, name in enumerate(CLASSES):
    row = "".join(f"{band_stat(BASE, i, c):>13.4f}" for i in range(len(BANDS)))
    print(f"  {name:12s}" + row)
print("  " + "-" * 92)
row = "".join(f"{np.nanmean([band_stat(BASE,i,c) for c in range(4)]):>13.4f}"
              for i in range(len(BANDS)))
print(f"  {'mIoU':12s}" + row)

print()
print("  shadow RECALL by band (the illumination-geometry question):")
row = "".join(f"{band_stat(BASE, i, 3, 1):>13.4f}" for i in range(len(BANDS)))
print("  " + " " * 12 + row)

# ── concat vs TCM by band ───────────────────────────────────────────────────
for c, name in enumerate(CLASSES):
    print()
    print("=" * 100)
    print(f"CONCAT vs CROSS-ATTENTION — {name.upper()} IoU by elevation band  (n=3 per side)")
    print("=" * 100)
    for i, b in enumerate(BANDS):
        cv = [band_stat(m, i, c) for m in CONCAT]
        tv = [band_stat(m, i, c) for m in TCM]
        if np.isnan(cv).any() or np.isnan(tv).any():
            print(f"  {b:>12s}   (no pixels)")
            continue
        sep = "SEPARATED" if (min(tv) > max(cv) or min(cv) > max(tv)) else "overlap"
        print(f"  {b:>12s}  concat {band(cv,'{:.4f}')}   TCM {band(tv,'{:.4f}')}   "
              f"d={np.mean(tv)-np.mean(cv):+.4f}  [1a {band_stat(BASE,i,c):.4f}]  {sep}")

# ── is the TCM gain an ALTITUDE effect or a SNOW effect? ────────────────────
print()
print("=" * 100)
print("KEY TEST — does the TCM thick-cloud gain track ALTITUDE or SNOW?")
print("  if altitude: delta should rise monotonically with band")
print("  if snow    : delta should be flat except where snow lives")
print("=" * 100)
print(f"  {'band':>12s}  {'thick d':>10s}  {'shadow d':>10s}  {'thin d':>10s}  {'clear d':>10s}  {'mIoU d':>10s}")
for i, b in enumerate(BANDS):
    cells = []
    for c in range(4):
        cv = [band_stat(m, i, c) for m in CONCAT]
        tv = [band_stat(m, i, c) for m in TCM]
        cells.append(np.mean(tv) - np.mean(cv))
    mi_c = np.mean([np.nanmean([band_stat(m, i, c) for c in range(4)]) for m in CONCAT])
    mi_t = np.mean([np.nanmean([band_stat(m, i, c) for c in range(4)]) for m in TCM])
    print(f"  {b:>12s}  {cells[1]:+10.4f}  {cells[3]:+10.4f}  {cells[2]:+10.4f}  "
          f"{cells[0]:+10.4f}  {mi_t-mi_c:+10.4f}")
