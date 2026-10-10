"""
Does TCM's attention become spatially selective where there is terrain to
select on?

Motivation: gap_terrain (TCM with attention deleted) matched TCM. One possible
explanation is dataset composition — 52.5% of test px are <2° slope and 59% sit
at 0–500 m. On a flat patch the elevation field is ~constant, so ANY attention
pattern returns ~the same attended value and TCM ≡ gap trivially. If attention
were doing real work, it should sharpen (entropy fall) on HIGH-RELIEF patches.

Test: per patch, normalised attention entropy (1 = uniform) vs within-patch
terrain relief, i.e. spread_terrain = SD of elevation (metres) over the key
grid at that level. Reads the stored per-patch rows of
experiments/tcm_attention_analysis/attention_stats.json — no GPU.
Sample = 100 snow-heavy + 100 snow-free test patches per seed (the extraction
script's selection), pooled here since relief spans ~0.6–370 m across them.

Result on first run (2026-10-01): attention stays near-uniform even on the
highest-relief quintile (L4 ≈0.98–0.99, L5 ≈0.93–0.97; minimum over all
patches 0.868). Only seed 1 sharpens with relief, and only 0.998 → 0.988 at L4.
→ Where attention COULD be selective, it is not. Caveat that this cannot
address: the model was TRAINED on mostly-flat data, so whether selective
attention would emerge from relief-dominated training is untested.
"""
import json, os
import numpy as np
from scipy.stats import spearmanr

OUT = r"C:\Users\prash\toposegnet\experiments\tcm_attention_analysis"
d = json.load(open(os.path.join(OUT, "attention_stats.json")))
seeds = d["seeds"]
NAMES = list(seeds.keys())
Q_LO, Q_HI = 0.2, 0.8                       # bottom / top relief quintiles


def band(vals, fmt="{:.4f}"):
    v = np.array(vals, dtype=float)
    return f"{fmt.format(v.mean())} [{fmt.format(v.min())}, {fmt.format(v.max())}]"


results = {}
print("=" * 88)
print("ATTENTION ENTROPY vs TERRAIN RELIEF  (entropy 1.0 = uniform; relief = elev SD, m)")
print("=" * 88)
for lvl in ("L4", "L5"):
    results[lvl] = {}
    print(f"\n── {lvl} " + "─" * 80)
    for n in NAMES:
        rows = seeds[n]["levels"][lvl]["snow"] + seeds[n]["levels"][lvl]["ctrl"]
        ent    = np.array([r["entropy"]        for r in rows])
        relief = np.array([r["spread_terrain"] for r in rows])
        elev   = np.array([r["elev_mean"]      for r in rows])

        lo_cut, hi_cut = np.quantile(relief, [Q_LO, Q_HI])
        rho_rel, p_rel = spearmanr(relief, ent)
        rho_elv, _     = spearmanr(elev, ent)   # relief and altitude co-vary; report both

        r = dict(n=len(rows),
                 rho_entropy_relief=float(rho_rel), p_entropy_relief=float(p_rel),
                 rho_entropy_elev=float(rho_elv),
                 entropy_low_relief=float(ent[relief <= lo_cut].mean()),
                 entropy_high_relief=float(ent[relief >= hi_cut].mean()),
                 entropy_min=float(ent.min()),
                 relief_range_m=[float(relief.min()), float(relief.max())])
        results[lvl][n] = r
        print(f"  {n:10s}  rho(ent,relief) {r['rho_entropy_relief']:+.3f} (p={p_rel:.1e})  "
              f"rho(ent,elev) {r['rho_entropy_elev']:+.3f}  "
              f"ent lowQ {r['entropy_low_relief']:.4f}  highQ {r['entropy_high_relief']:.4f}  "
              f"min {r['entropy_min']:.4f}")

    per = results[lvl].values()
    print(f"  ACROSS SEEDS  low-relief {band([r['entropy_low_relief'] for r in per])}   "
          f"high-relief {band([r['entropy_high_relief'] for r in per])}   "
          f"rho(ent,relief) {band([r['rho_entropy_relief'] for r in per], '{:+.3f}')}")

out_path = os.path.join(OUT, "entropy_vs_relief.json")
json.dump(results, open(out_path, "w"), indent=2)
print(f"\nsaved → {out_path}")
