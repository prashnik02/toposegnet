"""Pre-flight validation of regenerate_slope_aspect.py:
  1. numpy Horn slope must match richdem (with CORRECT 10 m geotransform)
     on the same smoothed surface.
  2. v2 output on a real mountainous patch must have a physically plausible
     slope distribution (continuous, not bimodal 0/90).
"""

import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import numpy as np
from scipy.ndimage import gaussian_filter, correlate
from regenerate_slope_aspect import KX, KY, slope_aspect_v2

# ── 1. numpy Horn vs richdem on identical smoothed input ─────────────────────
elev = np.memmap(r"C:\Users\prash\toposegnet\data\high_509\Test\EXTRA_elevation.dat",
                 dtype="int16", mode="r", shape=(975, 512, 512))

patch = elev[208].astype(np.float32)             # mountainous, relief ~1900 m
smoothed = gaussian_filter(patch, sigma=3.0, mode="nearest")

p = correlate(smoothed, KX, mode="nearest")
q = correlate(smoothed, KY, mode="nearest")
slope_np = np.degrees(np.arctan(np.hypot(p, q)))

try:
    import richdem as rd
    rda = rd.rdarray(smoothed, no_data=-9999)
    rda.geotransform = [0.0, 10.0, 0.0, 0.0, 0.0, -10.0]   # 10 m pixels
    slope_rd = np.array(rd.TerrainAttribute(rda, attrib="slope_degrees"))
    interior = (slice(2, -2), slice(2, -2))                  # exclude edge handling diffs
    diff = np.abs(slope_np[interior] - slope_rd[interior])
    print(f"numpy-Horn vs richdem(10m): max|diff|={diff.max():.4f} deg, "
          f"mean|diff|={diff.mean():.6f} deg")
except ImportError:
    print("richdem not importable - skipping cross-check")

# ── 2. full v2 pipeline output sanity on 3 mountainous patches ───────────────
for i in [208, 172, 236]:
    slope, sin_a, cos_a = slope_aspect_v2(elev[i])
    flat_frac = float(((sin_a == 0) & (cos_a == 0)).mean())
    mag = np.hypot(sin_a, cos_a)
    print(f"\npatch {i}: relief {int(elev[i].max() - elev[i].min())} m")
    print(f"  slope: p25={np.percentile(slope,25):.2f} p50={np.percentile(slope,50):.2f} "
          f"p75={np.percentile(slope,75):.2f} p99={np.percentile(slope,99):.2f} max={slope.max():.2f}")
    print(f"  frac<5deg={float((slope<5).mean()):.3f}  frac 5-45={float(((slope>=5)&(slope<=45)).mean()):.3f}  "
          f"frac>45={float((slope>45).mean()):.3f}")
    print(f"  aspect: flat(0,0) frac={flat_frac:.3f}; nonflat |(sin,cos)| in "
          f"[{mag[mag>0].min():.4f}, {mag.max():.4f}] (expect ~1.0)")
    assert np.isfinite(slope).all() and np.isfinite(sin_a).all() and np.isfinite(cos_a).all()

print("\nvalidation done")
