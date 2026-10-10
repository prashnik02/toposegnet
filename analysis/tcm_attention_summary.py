"""
Summarise tcm_attention_analysis.json in the campaign's reporting convention:
mean [min,max] across the 3 TCM seeds, and a PAIRED per-patch test for the
class-conditional attention gap (pairing removes between-patch elevation
variance, which is enormous and would otherwise swamp a ~65 m effect).

"""
import json, os
import numpy as np

OUT = r"C:\Users\prash\toposegnet\experiments\tcm_attention_analysis"
d = json.load(open(os.path.join(OUT, "attention_stats.json")))
seeds = d["seeds"]
NAMES = list(seeds.keys())


def band(vals, fmt="{:+.4f}"):
    v = np.array(vals, dtype=float)
    return f"{fmt.format(v.mean())} [{fmt.format(v.min())}, {fmt.format(v.max())}]"


def paired_gap(rows, a, b):
    """Per-patch (attended elev of class a) - (class b), only patches with both."""
    g = [r["per_class"][a]["e_att"] - r["per_class"][b]["e_att"]
         for r in rows if a in r["per_class"] and b in r["per_class"]]
    return np.array(g, dtype=float)


print("=" * 80)
print("T2  ELEVATION SELECTIVITY — the snow-vs-control CONTRAST")
print("    absolute rho is seed-inconsistent; the contrast is what to look at")
print("=" * 80)
for lvl in ("L4", "L5"):
    diffs, s_all, c_all = [], [], []
    for n in NAMES:
        s = np.mean([r["rho_mass"] for r in seeds[n]["levels"][lvl]["snow"]])
        c = np.mean([r["rho_mass"] for r in seeds[n]["levels"][lvl]["ctrl"]])
        diffs.append(s - c); s_all.append(s); c_all.append(c)
    print(f"  {lvl}  snow {band(s_all)}   ctrl {band(c_all)}")
    print(f"      snow-minus-ctrl {band(diffs)}   "
          f"{'ALL POSITIVE' if min(diffs) > 0 else 'sign varies'}")

print()
print("=" * 80)
print("T3  ATTENTION SHAPE — how far from uniform is it?")
print("=" * 80)
for lvl in ("L4", "L5"):
    ent, ratio, loc = [], [], []
    for n in NAMES:
        rows = seeds[n]["levels"][lvl]["snow"]
        ent.append(np.mean([r["entropy"] for r in rows]))
        ratio.append(np.mean([r["spread_att"] / r["spread_terrain"] for r in rows]))
        loc.append(np.mean([r["rho_local"] for r in rows]))
    print(f"  {lvl}  entropy(1=uniform) {band(ent, '{:.4f}')}   "
          f"sd(attended)/sd(terrain) {band(ratio, '{:.4f}')}   "
          f"rho(attended, own elev) {band(loc)}")

print()
print("=" * 80)
print("T4  CLASS-CONDITIONAL ATTENDED ELEVATION — PAIRED per patch, metres")
print("    positive = that class attends HIGHER terrain than `clear` does")
print("=" * 80)
for lvl in ("L4", "L5"):
    for cls in ("thick", "thin", "shadow"):
        for grp in ("snow", "ctrl"):
            means, ts, ns = [], [], []
            for n in NAMES:
                g = paired_gap(seeds[n]["levels"][lvl][grp], cls, "clear")
                if len(g) < 5:
                    continue
                means.append(g.mean())
                ts.append(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))
                ns.append(len(g))
            if not means:
                continue
            flag = ""
            if min(means) > 0 and min(ts) > 3:
                flag = "  <-- consistent, all t>3"
            elif max(means) < 0 and max(ts) < -3:
                flag = "  <-- consistent, all t<-3"
            print(f"  {lvl} {grp:4s} {cls:6s}-clear  {band(means, '{:+7.1f}')} m   "
                  f"t {band(ts, '{:+.1f}')}  n~{int(np.mean(ns))}{flag}")
    print()

print("=" * 80)
print("T4b  IS THE thick-clear GAP SNOW-SPECIFIC?   ratio snow/ctrl per seed")
print("=" * 80)
for lvl in ("L4", "L5"):
    rs = []
    for n in NAMES:
        s = paired_gap(seeds[n]["levels"][lvl]["snow"], "thick", "clear").mean()
        c = paired_gap(seeds[n]["levels"][lvl]["ctrl"], "thick", "clear").mean()
        rs.append(s / c if c != 0 else np.nan)
    print(f"  {lvl}  snow/ctrl ratio {band(rs, '{:.2f}')}")

print()
print("=" * 80)
print("T1  HOW HARD IS THE CONDITIONING DRIVEN?  ||attn residual|| / ||features||")
print("=" * 80)
for lvl in ("L4", "L5"):
    v = [np.mean([r["rel_residual"] for r in seeds[n]["levels"][lvl]["snow"]])
         for n in NAMES]
    print(f"  {lvl}  {band(v, '{:.4f}')}   per-seed " +
          ", ".join(f"{n.split()[-1]} {x:.3f}" for n, x in zip(NAMES, v)))
