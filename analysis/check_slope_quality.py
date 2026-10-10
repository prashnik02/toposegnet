"""Sanity-check the DERIVED_slope.dat distribution.
The stratified analysis showed a bimodal slope distribution (68% at 0-5 deg,
22% at >45 deg, ~1% in between) which is not how natural terrain behaves.
Hypothesis: nearest-neighbour DEM resampling produced a staircase DEM, so the
derived slope is ~0 on plateaus and huge at the steps."""

import numpy as np

SHAPE = (975, 512, 512)
ROOT  = r"C:\Users\prash\toposegnet\data\high_509\Test"

slope = np.memmap(ROOT + r"\DERIVED_slope.dat", dtype="float32", mode="r", shape=SHAPE)
elev  = np.memmap(ROOT + r"\EXTRA_elevation.dat", dtype="int16",  mode="r", shape=SHAPE)

# ── global slope distribution (every 25th patch) ─────────────────────────────
s = np.stack([slope[i] for i in range(0, 975, 25)]).ravel()
print("slope percentiles (39-patch sample):")
for p in [1, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{p:>3} = {np.percentile(s, p):8.2f}")
print(f"frac slope < 5  : {(s < 5).mean():.4f}")
print(f"frac 5..45      : {((s >= 5) & (s <= 45)).mean():.4f}")
print(f"frac slope > 45 : {(s > 45).mean():.4f}")

# ── staircase check on a mountainous patch ───────────────────────────────────
# patch 208: mean_elev ~1457 m, mean_slope ~38 deg in per_patch.csv
for i in [208, 172, 236]:
    e = elev[i].astype(np.float32)
    d = np.abs(np.diff(e, axis=1))                    # horizontal neighbour diffs
    sl = slope[i]
    nz = d[d > 0]
    print(f"\npatch {i}: elev range [{e.min():.0f},{e.max():.0f}] m, "
          f"unique elev values: {len(np.unique(e))}")
    print(f"  frac horizontal elev-diff == 0 : {(d == 0).mean():.4f}")
    if nz.size:
        print(f"  nonzero diffs p50/p90/p99      : "
              f"{np.percentile(nz,50):.1f} / {np.percentile(nz,90):.1f} / {np.percentile(nz,99):.1f} m per 10m px")
    print(f"  slope frac<5: {(sl < 5).mean():.4f}   frac>45: {(sl > 45).mean():.4f}")
