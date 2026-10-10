"""
Per-class IoU overall, by ELEVATION band and by SLOPE bin — NO snow stratum.


Why: the `snow` / `water_like` surfaces are a spectral heuristic (NDSI>0.4 split
by B8>0.11), not ground truth, and the heuristic is evaluated through cloud.
Every number here uses only (a) the CloudSEN12+ labels and (b) the DEM-derived
strata (raw EXTRA_elevation.dat bands; clean DERIVED_slope_v2.dat bins), which
are independent of both the spectral bands and the labels. All surfaces are
pooled, so the snow mask never enters.

Sources (no GPU): conf_all + conf_elev from stratified_comparison.json,
conf_slope from joint_stratification.json. Verified on 2026-10-01 that every
model's conf_elev and conf_slope sum exactly to its conf_all.

Convention: mean [min,max] over n=3 seeds; delta vs concat with SEP (seed
ranges disjoint) / ovl. 1a is a single seedless run, shown for reference only.

⚠️ Ignore the `<0` elevation band (0.5% px): it has ~no thin cloud, so thin IoU
is 0 for models that false-positive thin and NaN for ones that don't, which
makes mIoU average over different class sets (film_elev's "+0.07" there is
this artifact, not an effect).
"""
import json, os
import numpy as np

ROOT = r"C:\Users\prash\toposegnet\experiments"
S = json.load(open(os.path.join(ROOT, "stratified_comparison", "stratified_comparison.json")))
J = json.load(open(os.path.join(ROOT, "joint_stratification", "joint_stratification.json")))
CLASSES = S["class_names"]
ELEV, SLOPE = S["elev_labels"], J["slope_labels"]

CONFIGS = {   # display name -> model keys in both JSONs
    "1a":          ["1a baseline (13ch)"],
    "concat":      ["2a concat seed1", "2a concat seed2", "2a concat seed3"],
    "elev_pooled": ["elev_pooled seed0", "elev_pooled seed1", "elev_pooled seed2"],
    "film_elev":   ["film_elev seed0", "film_elev seed1", "film_elev seed2"],
    "TCM":         ["6 elev CROSS-ATTN (TCM)", "6 TCM seed1", "6 TCM seed2"],
    "gap":         ["gap seed0", "gap seed1", "gap seed2"],
}
COMPARE = ["elev_pooled", "film_elev", "TCM", "gap"]


def ious(conf):
    """conf (4,4) -> per-class IoU (4,) + mIoU appended -> (5,)."""
    c = np.asarray(conf, dtype=np.float64)
    tp = np.diag(c)
    denom = c.sum(0) + c.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return np.append(iou, np.nanmean(iou))


def stratum_table(get_conf, n_bins):
    """-> {config: array (n_seeds, n_bins, 5)}"""
    return {cfg: np.stack([[ious(get_conf(k)[b]) for b in range(n_bins)] for k in keys])
            for cfg, keys in CONFIGS.items()}


overall = {cfg: np.stack([ious(S["models"][k]["conf_all"])[None] for k in keys])
           for cfg, keys in CONFIGS.items()}
by_elev  = stratum_table(lambda k: S["models"][k]["conf_elev"],  len(ELEV))
by_slope = stratum_table(lambda k: J["models"][k]["conf_slope"], len(SLOPE))
px_elev  = np.array(S["models"]["gap seed0"]["conf_elev"]).sum((1, 2))
px_slope = np.array(J["models"]["gap seed0"]["conf_slope"]).sum((1, 2))


def verdict(a, b):
    """a, b: per-seed values. SEP if the seed ranges are disjoint."""
    return "SEP" if (a.min() > b.max() or a.max() < b.min()) else "ovl"


def report(title, tab, labels, px):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    metrics = CLASSES + ["mIoU"]
    for b, lab in enumerate(labels):
        share = f"{100 * px[b] / px.sum():.1f}% px" if px is not None else ""
        print(f"\n── {lab}  {share}")
        print(f"   {'config':12s}" + "".join(f"{m:>26s}" for m in metrics))
        for cfg in CONFIGS:
            v = tab[cfg][:, b, :]
            cells = [f"{v[:, i].mean():.4f}" + (f" [{v[:, i].min():.4f},{v[:, i].max():.4f}]"
                     if len(v) > 1 else " " * 16) for i in range(5)]
            print(f"   {cfg:12s}" + "".join(f"{c:>26s}" for c in cells))
        for cfg in COMPARE:
            a, c = tab[cfg][:, b, :], tab["concat"][:, b, :]
            cells = [f"{a[:, i].mean() - c[:, i].mean():+.4f} {verdict(a[:, i], c[:, i])}" for i in range(5)]
            print(f"   {'Δ ' + cfg:12s}" + "".join(f"{x:>26s}" for x in cells))


report("OVERALL (all test pixels)", overall, ["all"], None)
report("BY ELEVATION BAND (m, raw DEM)", by_elev, ELEV, px_elev)
report("BY SLOPE BIN (degrees, v2 slope)", by_slope, SLOPE, px_slope)

out = {name: {cfg: arr.tolist() for cfg, arr in tab.items()}
       for name, tab in (("overall", overall), ("by_elev", by_elev), ("by_slope", by_slope))}
out.update(metrics=CLASSES + ["mIoU"], elev_labels=ELEV, slope_labels=SLOPE,
           configs={c: k for c, k in CONFIGS.items()},
           px_elev=px_elev.tolist(), px_slope=px_slope.tolist())
path = os.path.join(ROOT, "stratified_comparison", "snow_free_report.json")
json.dump(out, open(path, "w"), indent=1)
print(f"\nsaved → {path}")
