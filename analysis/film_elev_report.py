"""
Report on the pooled-elevation FiLM control (film_elev) vs the four existing
configurations, with the slope dose-response check that decided elev_pooled.


film_elev holds the FEATURE fixed (patch-mean elevation, identical to
elev_pooled) and varies only the DELIVERY ROUTE: gamma/beta conditioning at the
bottleneck instead of a concatenated 14th input channel.  Together with
elev_pooled it separates "pooling needs depth-injection" from "pooling needs the
gap branch's capacity" -- the gap elev_pooled could not close, because a
per-patch-constant channel can only act through the first conv's effective bias.
"""
import json, os
import numpy as np

J = os.path.join(r"C:\Users\prash\toposegnet\experiments\joint_stratification",
                 "joint_stratification.json")
d = json.load(open(J))
CLASSES, SURF = d["class_names"], d["surf_labels"]
ELEV, SLOPE = d["elev_labels"], d["slope_labels"]
M = d["models"]

CONFIGS = {
    "concat":      ["2a concat seed1", "2a concat seed2", "2a concat seed3"],
    "TCM":         ["6 elev CROSS-ATTN (TCM)", "6 TCM seed1", "6 TCM seed2"],
    "gap":         ["gap seed0", "gap seed1", "gap seed2"],
    "elev_pooled": ["elev_pooled seed0", "elev_pooled seed1", "elev_pooled seed2"],
    "film_elev":   ["film_elev seed0", "film_elev seed1", "film_elev seed2"],
}
BASE = "1a baseline (13ch)"
LOW_B, MID_B, HIGH_B = [0, 1, 2], [3], [4, 5]


def m_iou(conf):
    conf = np.asarray(conf, float)
    ious = []
    for c in range(4):
        tp = conf[c, c]; fn = conf[c, :].sum() - tp; fp = conf[:, c].sum() - tp
        u = tp + fp + fn
        if u:
            ious.append(tp / u)
    return float(np.mean(ious)) if ious else np.nan


def cls_iou(conf, c):
    conf = np.asarray(conf, float)
    tp = conf[c, c]; fn = conf[c, :].sum() - tp; fp = conf[:, c].sum() - tp
    u = tp + fp + fn
    return tp / u if u else np.nan


def se(model, surfs, bands):
    a = np.asarray(M[model]["conf_surf_elev"], float)
    return a[np.ix_(surfs, bands)].sum(axis=(0, 1))


def slope_conf(model, i):
    return np.asarray(M[model]["conf_slope"], float)[i]


def band(v):
    v = np.array(v, float)
    return f"{v.mean():.4f} [{v.min():.4f}, {v.max():.4f}]"


def verdict(a, b):
    return "SEPARATED" if (min(b) > max(a) or min(a) > max(b)) else "overlap"


# ---------------------------------------------------------------- aggregate
print("=" * 104)
print("AGGREGATE (stratification confusions summed over all strata), n=3 per config")
print("=" * 104)
print(f"  {'config':14s} {'mIoU':>26s}  {'thick':>8s} {'thin':>8s} {'shadow':>8s} {'clear':>8s}")
allc = {k: [np.asarray(M[m]["conf_slope"], float).sum(axis=0) for m in v]
        for k, v in CONFIGS.items()}
for k, confs in allc.items():
    mi = [m_iou(c) for c in confs]
    per = [np.mean([cls_iou(c, ci) for c in confs]) for ci in range(4)]
    print(f"  {k:14s} {band(mi):>26s}  {per[1]:8.4f} {per[2]:8.4f} {per[3]:8.4f} {per[0]:8.4f}")
b_all = np.asarray(M[BASE]["conf_slope"], float).sum(axis=0)
print(f"  {'1a baseline':14s} {m_iou(b_all):>26.4f}")

print("\n  pairwise verdicts on aggregate mIoU:")
keys = list(CONFIGS)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a = [m_iou(c) for c in allc[keys[i]]]
        b = [m_iou(c) for c in allc[keys[j]]]
        print(f"    {keys[i]:12s} vs {keys[j]:12s}  d={np.mean(b)-np.mean(a):+.4f}  {verdict(a,b)}")

# ------------------------------------------------------- slope dose-response
print()
print("=" * 104)
print("THE DECISIVE TEST -- slope dose-response, mIoU delta vs concat (n=3 each)")
print("=" * 104)
tots = np.asarray(M[BASE]["conf_slope"], float).sum(axis=(1, 2))
hdr = f"  {'slope':>8s} {'% px':>6s}"
for k in ["TCM", "gap", "elev_pooled", "film_elev"]:
    hdr += f"  {k:>22s}"
print(hdr)
for i, s in enumerate(SLOPE):
    cv = [m_iou(slope_conf(m, i)) for m in CONFIGS["concat"]]
    row = f"  {s:>8s} {tots[i]/tots.sum()*100:5.2f}%"
    for k in ["TCM", "gap", "elev_pooled", "film_elev"]:
        tv = [m_iou(slope_conf(m, i)) for m in CONFIGS[k]]
        row += f"  {np.mean(tv)-np.mean(cv):+8.4f} {verdict(cv,tv):>13s}"
    print(row)

print("\n  head-to-head film_elev vs gap, per slope bin (does the FiLM route match the branch?):")
for i, s in enumerate(SLOPE):
    gv = [m_iou(slope_conf(m, i)) for m in CONFIGS["gap"]]
    fv = [m_iou(slope_conf(m, i)) for m in CONFIGS["film_elev"]]
    print(f"    {s:>8s}  gap {band(gv)}   film {band(fv)}   "
          f"d={np.mean(fv)-np.mean(gv):+.4f}  {verdict(gv,fv)}")

print("\n  thin / shadow IoU delta vs concat, per slope bin, film_elev:")
for i, s in enumerate(SLOPE):
    row = f"    {s:>8s}"
    for c, name in [(1, "thick"), (2, "thin"), (3, "shadow")]:
        cc = [cls_iou(slope_conf(m, i), c) for m in CONFIGS["concat"]]
        ff = [cls_iou(slope_conf(m, i), c) for m in CONFIGS["film_elev"]]
        row += f"  {name} {np.mean(ff)-np.mean(cc):+.4f} {verdict(cc,ff):>9s}"
    print(row)

# ------------------------------------------------------------ snow x altitude
print()
print("=" * 104)
print("SNOW x ALTITUDE 2x2 -- mIoU delta vs concat")
print("=" * 104)
cells = [("snow  x LOW  (<1000m)", [1], LOW_B),
         ("snow  x HIGH (>2000m)", [1], HIGH_B),
         ("other x LOW  (<1000m)", [0], LOW_B),
         ("other x HIGH (>2000m)", [0], HIGH_B)]
tot = np.asarray(M[BASE]["conf_surf_elev"], float).sum()
hdr = f"  {'cell':24s} {'% px':>6s}"
for k in ["TCM", "gap", "elev_pooled", "film_elev"]:
    hdr += f"  {k:>22s}"
print(hdr)
for label, s, b in cells:
    n = se(BASE, s, b).sum()
    cv = [m_iou(se(m, s, b)) for m in CONFIGS["concat"]]
    row = f"  {label:24s} {n/tot*100:5.2f}%"
    for k in ["TCM", "gap", "elev_pooled", "film_elev"]:
        tv = [m_iou(se(m, s, b)) for m in CONFIGS[k]]
        row += f"  {np.mean(tv)-np.mean(cv):+8.4f} {verdict(cv,tv):>13s}"
    print(row)

# ----------------------------------------------------------- elevation bands
print()
print("=" * 104)
print("ELEVATION BANDS -- mIoU delta vs concat")
print("=" * 104)
ce = np.asarray(M[BASE]["conf_surf_elev"], float)
hdr = f"  {'band':>12s} {'% px':>6s}"
for k in ["gap", "elev_pooled", "film_elev"]:
    hdr += f"  {k:>22s}"
print(hdr)
for bi, bl in enumerate(ELEV):
    n = ce[:, bi].sum()
    cv = [m_iou(se(m, [0, 1, 2], [bi])) for m in CONFIGS["concat"]]
    row = f"  {bl:>12s} {n/ce.sum()*100:5.2f}%"
    for k in ["gap", "elev_pooled", "film_elev"]:
        tv = [m_iou(se(m, [0, 1, 2], [bi])) for m in CONFIGS[k]]
        row += f"  {np.mean(tv)-np.mean(cv):+8.4f} {verdict(cv,tv):>13s}"
    print(row)
