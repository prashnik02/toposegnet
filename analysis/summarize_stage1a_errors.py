"""
Summarize the saved Stage 1a terrain-stratified error analysis.
Reads experiments/stage1a_error_analysis/{per_patch.csv, pixel_confusions.json}
(produced by stage1a_terrain_error_analysis.py) — no GPU, no inference.
"""

import json
import numpy as np
import pandas as pd

OUT_DIR = r"C:\Users\prash\toposegnet\experiments\stage1a_error_analysis"
NUM_CLASSES = 4
CLASS_NAMES = ["clear", "thick", "thin", "shadow"]

with open(f"{OUT_DIR}\\pixel_confusions.json") as f:
    pc = json.load(f)
df = pd.read_csv(f"{OUT_DIR}\\per_patch.csv")

conf_elev  = np.array(pc["conf_elev"],  dtype=np.float64)
conf_slope = np.array(pc["conf_slope"], dtype=np.float64)
conf_surf  = np.array(pc["conf_surf"],  dtype=np.float64)
total_px   = conf_elev.sum()


def summarize(conf, labels, title):
    print(f"\n{'='*74}\n{title}\n{'='*74}")
    print(f"{'bin':>12} {'Mpx':>8} {'px%':>6} {'err%':>7} "
          f"{'IoU_clr':>8} {'IoU_thk':>8} {'IoU_thn':>8} {'IoU_shd':>8} "
          f"{'clr>cld%':>9} {'shd_rec':>8}")
    for i, lab in enumerate(labels):
        c = conf[i]
        n = c.sum()
        if n == 0:
            print(f"{lab:>12} {'—':>8}")
            continue
        err = 1.0 - np.trace(c) / n
        ious = []
        for cls in range(NUM_CLASSES):
            tp = c[cls, cls]
            union = c[cls, :].sum() + c[:, cls].sum() - tp
            ious.append(tp / union if union > 0 else np.nan)
        clr2cld = (c[0, 1] + c[0, 2]) / max(c[0, :].sum(), 1)   # false cloud on clear ground
        shd_rec = c[3, 3] / max(c[3, :].sum(), 1)               # shadow recall
        print(f"{lab:>12} {n/1e6:>8.1f} {100*n/total_px:>5.1f}% {100*err:>6.2f}% "
              + " ".join(f"{v:>8.4f}" if np.isfinite(v) else f"{'—':>8}" for v in ious)
              + f" {100*clr2cld:>8.2f}% {shd_rec:>8.4f}")


summarize(conf_elev,  pc["elev_labels"],  "PIXEL ERRORS BY ELEVATION (m)")
summarize(conf_slope, pc["slope_labels"], "PIXEL ERRORS BY SLOPE (deg)")
summarize(conf_surf,  pc["surf_labels"],
          "PIXEL ERRORS BY SURFACE (NDSI>0.4, snow=NIR-bright, water=NIR-dark)")

# Full confusion per surface class — where do snow/water pixels actually go?
for si, sname in enumerate(pc["surf_labels"]):
    if sname == "other":
        continue
    print(f"\n{'='*74}\nROW-NORMALIZED CONFUSION, {sname.upper()} PIXELS (rows=true, cols=pred)\n{'='*74}")
    c = conf_surf[si]
    print(f"{'':>8}" + "".join(f"{n:>10}" for n in CLASS_NAMES) + f"{'Mpx':>10}")
    for i, n in enumerate(CLASS_NAMES):
        r = c[i] / max(c[i].sum(), 1)
        print(f"{n:>8}" + "".join(f"{v:>10.4f}" for v in r) + f"{c[i].sum()/1e6:>10.2f}")

# ── patch-level Spearman correlations ────────────────────────────────────────
def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    ra = pd.Series(a[m]).rank().values
    rb = pd.Series(b[m]).rank().values
    # Pearson on ranks, computed elementwise — np.corrcoef hard-crashes in
    # this env (numpy BLAS delay-load failure 0xc06d007f on Windows).
    da, db = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((da * da).sum() * (db * db).sum())
    return float((da * db).sum() / denom) if denom > 0 else np.nan


print(f"\n{'='*74}\nPATCH-LEVEL SPEARMAN CORRELATIONS (n={len(df)})\n{'='*74}")
targets = ["err_rate", "iou_shadow", "iou_thin", "iou_thick", "iou_clear"]
drivers = ["mean_elev", "max_elev", "std_elev", "mean_slope", "max_slope",
           "snow_frac", "clear_snow_frac"]
print(f"{'':>16}" + "".join(f"{t:>12}" for t in targets))
for d in drivers:
    vals = [spearman(df[d].values, df[t].values) for t in targets]
    print(f"{d:>16}" + "".join(f"{v:>12.3f}" for v in vals))

# ── flat vs mountainous patch split (paper-ready headline numbers) ───────────
print(f"\n{'='*74}\nPATCH GROUPS: FLAT (mean_slope<10) vs MOUNTAINOUS (mean_slope>=10)\n{'='*74}")
for name, grp in [("flat", df[df.mean_slope < 10]),
                  ("mountainous", df[df.mean_slope >= 10])]:
    print(f"{name:>12}: n={len(grp):>4}  err={grp.err_rate.mean():.4f}  "
          f"IoU shd={grp.iou_shadow.mean():.4f}  thn={grp.iou_thin.mean():.4f}  "
          f"thk={grp.iou_thick.mean():.4f}  clr={grp.iou_clear.mean():.4f}  "
          f"(patch-mean IoU, NaN patches excluded per class)")

# snow-bearing vs snow-free patches (snow_frac now excludes water: NDSI & NIR-bright)
print(f"\nPATCH GROUPS: SNOW-FREE (snow_frac=0) vs SNOWY (snow_frac>0.05)")
for name, grp in [("snow-free", df[df.snow_frac == 0]),
                  ("snowy>5%", df[df.snow_frac > 0.05])]:
    print(f"{name:>12}: n={len(grp):>4}  err={grp.err_rate.mean():.4f}  "
          f"IoU shd={grp.iou_shadow.mean():.4f}  thn={grp.iou_thin.mean():.4f}  "
          f"thk={grp.iou_thick.mean():.4f}  clr={grp.iou_clear.mean():.4f}")

# worst 15 patches by error rate, with their terrain context
print(f"\n{'='*74}\nWORST 15 PATCHES BY ERROR RATE\n{'='*74}")
cols = ["idx", "err_rate", "mean_elev", "mean_slope", "snow_frac",
        "frac_clear", "frac_thick", "frac_thin", "frac_shadow"]
print(df.nlargest(15, "err_rate")[cols].to_string(index=False,
      float_format=lambda x: f"{x:.3f}"))
