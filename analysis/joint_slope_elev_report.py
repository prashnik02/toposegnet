"""
Report on joint_stratification.json — snow vs altitude, and slope.


The decisive test is the 2x2: {snow, other} x {low altitude, high altitude}.
  * if the TCM gain is an ALTITUDE effect, both high-altitude cells gain
    regardless of snow;
  * if it is a SNOW effect, both snow cells gain regardless of altitude.
Snow and high altitude are correlated, which is exactly why the marginals
(reported previously) cannot separate them and the joint can.
"""
import json, os
import numpy as np

J = os.path.join(r"C:\Users\prash\toposegnet\experiments\joint_stratification",
                 "joint_stratification.json")
d = json.load(open(J))
CLASSES, SURF = d["class_names"], d["surf_labels"]
ELEV, SLOPE   = d["elev_labels"], d["slope_labels"]
M = d["models"]

CONCAT = ["2a concat seed1", "2a concat seed2", "2a concat seed3"]
TCM    = ["6 elev CROSS-ATTN (TCM)", "6 TCM seed1", "6 TCM seed2"]
BASE   = "1a baseline (13ch)"

# elevation bands -> LOW (<1000 m) / HIGH (>=2000 m); 1000-2000 held out as MID
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


def cls_recall(conf, c):
    conf = np.asarray(conf, float)
    tp = conf[c, c]; fn = conf[c, :].sum() - tp
    return tp / (tp + fn) if (tp + fn) else np.nan


def se(model, surfs, bands):
    """Sum the surface x elevation confusion over the given surface/band sets."""
    a = np.asarray(M[model]["conf_surf_elev"], float)
    return a[np.ix_(surfs, bands)].sum(axis=(0, 1))


def band(v, f="{:+.4f}"):
    v = np.array(v, float)
    return f"{f.format(v.mean())} [{f.format(v.min())}, {f.format(v.max())}]"


def verdict(cv, tv):
    return "SEPARATED" if (min(tv) > max(cv) or min(cv) > max(tv)) else "overlap"


print("=" * 104)
print("Q1  IS IT SNOW OR ALTITUDE?   2x2 joint stratification, n=3 per side")
print("=" * 104)
cells = [("snow  x LOW  (<1000m)",  [1], LOW_B),
         ("snow  x HIGH (>2000m)",  [1], HIGH_B),
         ("other x LOW  (<1000m)",  [0], LOW_B),
         ("other x HIGH (>2000m)",  [0], HIGH_B),
         ("snow  x MID  (1-2000m)", [1], MID_B),
         ("other x MID  (1-2000m)", [0], MID_B)]
tot = np.asarray(M[BASE]["conf_surf_elev"], float).sum()
print(f"  {'cell':24s} {'% px':>6s}   {'concat mIoU':>28s}   {'TCM mIoU':>28s}   {'delta':>8s}  verdict")
for label, s, b in cells:
    n = se(BASE, s, b).sum()
    cv = [m_iou(se(m, s, b)) for m in CONCAT]
    tv = [m_iou(se(m, s, b)) for m in TCM]
    print(f"  {label:24s} {n/tot*100:5.2f}%   {band(cv,'{:.4f}'):>28s}   "
          f"{band(tv,'{:.4f}'):>28s}   {np.mean(tv)-np.mean(cv):+8.4f}  {verdict(cv,tv)}")

print()
print("  same 2x2, THICK-CLOUD IoU only (the class the campaign headline is about):")
for label, s, b in cells:
    cv = [cls_iou(se(m, s, b), 1) for m in CONCAT]
    tv = [cls_iou(se(m, s, b), 1) for m in TCM]
    print(f"  {label:24s}          {band(cv,'{:.4f}'):>28s}   "
          f"{band(tv,'{:.4f}'):>28s}   {np.mean(tv)-np.mean(cv):+8.4f}  {verdict(cv,tv)}")

print()
print("  same 2x2, SHADOW IoU  [1a baseline in brackets]:")
for label, s, b in cells:
    cv = [cls_iou(se(m, s, b), 3) for m in CONCAT]
    tv = [cls_iou(se(m, s, b), 3) for m in TCM]
    print(f"  {label:24s}          {band(cv,'{:.4f}'):>28s}   "
          f"{band(tv,'{:.4f}'):>28s}   {np.mean(tv)-np.mean(cv):+8.4f}  "
          f"[1a {cls_iou(se(BASE,s,b),3):.4f}]  {verdict(cv,tv)}")

print()
print("=" * 104)
print("Q2  SLOPE — 1a baseline, per-class IoU by slope bin (clean v2 slope)")
print("=" * 104)
sl = np.asarray(M[BASE]["conf_slope"], float)
tots = sl.sum(axis=(1, 2))
print("  " + " " * 12 + "".join(f"{b:>12s}" for b in SLOPE) + "   (degrees)")
print(f"  {'% of px':12s}" + "".join(f"{n/tots.sum()*100:11.2f}%" for n in tots))
for c, name in enumerate(CLASSES):
    print(f"  {name:12s}" + "".join(f"{cls_iou(sl[i], c):12.4f}" for i in range(len(SLOPE))))
print(f"  {'mIoU':12s}" + "".join(f"{m_iou(sl[i]):12.4f}" for i in range(len(SLOPE))))
print(f"  {'shadow rec':12s}" + "".join(f"{cls_recall(sl[i],3):12.4f}"
                                        for i in range(len(SLOPE))))

print()
print("=" * 104)
print("Q2b  SLOPE — does the TCM gain track slope?   (delta = TCM n=3 - concat n=3)")
print("=" * 104)
print(f"  {'slope':>8s}  {'% px':>6s}  {'mIoU d':>9s}  {'thick d':>9s}  {'thin d':>9s}  "
      f"{'shadow d':>9s}  {'clear d':>9s}  verdict(mIoU)")
for i, b in enumerate(SLOPE):
    cv = [m_iou(np.asarray(M[m]["conf_slope"], float)[i]) for m in CONCAT]
    tv = [m_iou(np.asarray(M[m]["conf_slope"], float)[i]) for m in TCM]
    ds = []
    for c in range(4):
        cc = [cls_iou(np.asarray(M[m]["conf_slope"], float)[i], c) for m in CONCAT]
        tt = [cls_iou(np.asarray(M[m]["conf_slope"], float)[i], c) for m in TCM]
        ds.append(np.mean(tt) - np.mean(cc))
    print(f"  {b:>8s}  {tots[i]/tots.sum()*100:5.2f}%  {np.mean(tv)-np.mean(cv):+9.4f}  "
          f"{ds[1]:+9.4f}  {ds[2]:+9.4f}  {ds[3]:+9.4f}  {ds[0]:+9.4f}  {verdict(cv,tv)}")

print()
print("=" * 104)
print("Q2c  SLOPE x SURFACE — shadow IoU, 1a baseline (is slope's effect just snow?)")
print("=" * 104)
ss = np.asarray(M[BASE]["conf_slope_surf"], float)
print("  " + " " * 12 + "".join(f"{b:>12s}" for b in SLOPE))
for si, sname in enumerate(SURF):
    print(f"  {sname:12s}" + "".join(f"{cls_iou(ss[i, si], 3):12.4f}"
                                     for i in range(len(SLOPE))))
print("\n  pixel share of each slope bin within each surface:")
for si, sname in enumerate(SURF):
    tot_s = ss[:, si].sum()
    print(f"  {sname:12s}" + "".join(f"{ss[i,si].sum()/max(tot_s,1)*100:11.2f}%"
                                     for i in range(len(SLOPE))))
