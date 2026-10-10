"""
Regenerate DERIVED_slope / DERIVED_sin_aspect / DERIVED_cos_aspect  (v2)
========================================================================

Why v1 is broken (July 2026 error analysis, see project notes):
  1. EXTRA_elevation.dat is a STAIRCASE: the dataset's elevation was
     nearest-neighbour resampled from a ~3 arc-sec (~40-90 m) DEM onto the
     10 m patch grid. 84-89% of adjacent pixels are identical, so any local
     derivative is 0 on plateaus and huge at block steps.
  2. Slope.ipynb passed no geotransform to richdem, so pixel spacing
     defaulted to 1 m instead of 10 m — every slope inflated ~10x
     (17 m step read as arctan(17/1)=86.6 deg instead of arctan(17/10)=59.5).
  3. richdem returns aspect = -1 for flat cells; the notebook fed that -1
     straight into sin/cos, so ~70% of all pixels got a spurious
     "facing north" aspect instead of "no aspect".

The v2 fix, per patch:
  1. Gaussian-smooth the staircase back into a continuous surface.
     sigma is ADAPTIVE: half the measured staircase block size (median run
     length of identical values), clipped to [1.5, 5] px. Rationale: the
     source DEM has no real information below its own resolution, so
     smoothing at ~half a block removes steps without destroying signal.
  2. Horn's (1981) 3x3 finite differences at the correct 10 m spacing
     (same method as GIS slope tools / richdem, implemented in numpy).
  3. slope = arctan(|gradient|) in degrees [0, 90].
     aspect encoded as sin/cos; pixels with slope < FLAT_DEG get
     sin = cos = 0 — an explicit "no direction" code (aspect is
     mathematically undefined on flat ground).

Outputs (written NEXT TO the v1 files, which are kept for reproducibility
of Stages 3-4):
    {split}/DERIVED_slope_v2.dat        float32 (n, 512, 512)
    {split}/DERIVED_sin_aspect_v2.dat   float32 (n, 512, 512)
    {split}/DERIVED_cos_aspect_v2.dat   float32 (n, 512, 512)

Also prints train-split slope mean/std — the v2 replacements for
SLOPE_MEAN / SLOPE_STD in dataset.py.

Run:  python src/regenerate_slope_aspect.py            (~15-25 min, CPU only)
"""

import numpy as np
from scipy.ndimage import gaussian_filter, correlate
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\prash\toposegnet\data\high_509"
SPLITS    = {"Train": 8490, "Val": 535, "Test": 975}
PIXEL_M   = 10.0        # Sentinel-2 patch grid spacing in metres
FLAT_DEG  = 1.0         # below this slope, aspect is undefined -> sin=cos=0
SIGMA_MIN, SIGMA_MAX = 1.5, 5.0

# Horn (1981) 3x3 kernels. Window rows run north->south (row+1 is southward).
#   p = dz/dx (east),  q = dz/dsouth
KX = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32) / (8.0 * PIXEL_M)
KY = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [ 1,  2,  1]], dtype=np.float32) / (8.0 * PIXEL_M)


def estimate_block_px(patch, n_rows=8):
    """Median run length of identical values along sampled rows = the
    nearest-neighbour upsampling factor (staircase block size) in pixels.
    Returns np.nan for constant patches (e.g. open ocean)."""
    runs = []
    for r in range(0, patch.shape[0], patch.shape[0] // n_rows):
        row = patch[r]
        change = np.flatnonzero(np.diff(row)) + 1
        edges = np.concatenate([[0], change, [len(row)]])
        rl = np.diff(edges)
        runs.append(rl[rl < len(row)])          # drop all-constant rows
    runs = np.concatenate(runs)
    return float(np.median(runs)) if runs.size else np.nan


def slope_aspect_v2(elev_int16):
    """Staircase elevation patch (int16, metres) -> (slope_deg, sin_asp, cos_asp).

    Shapes: (512,512) int16 -> three (512,512) float32.
    Example: a smoothed 17 m rise over 30 m of ground -> slope ~29.5 deg.
    """
    e = elev_int16.astype(np.float32)

    # 1. adaptive Gaussian: sigma = half the staircase block, clipped
    block = estimate_block_px(elev_int16)
    sigma = SIGMA_MIN if np.isnan(block) else float(np.clip(block / 2.0, SIGMA_MIN, SIGMA_MAX))
    e = gaussian_filter(e, sigma=sigma, mode="nearest")

    # 2. Horn gradients at 10 m spacing ('nearest' replicates edges)
    p = correlate(e, KX, mode="nearest")        # dz/dx, east-positive
    q = correlate(e, KY, mode="nearest")        # dz/dsouth, south-positive

    # 3. slope in degrees
    slope = np.degrees(np.arctan(np.hypot(p, q))).astype(np.float32)

    # 4. aspect: compass direction the surface FACES (downslope direction),
    #    clockwise from north. Downslope = -grad(z):
    #      east component  = -p
    #      north component = +q   (dz/dnorth = -q  =>  -(dz/dnorth) = q)
    #    We never need the angle itself — sin/cos come from atan2 directly.
    az = np.arctan2(-p, q)                      # radians, 0 = north, cw = +east
    sin_a = np.sin(az).astype(np.float32)
    cos_a = np.cos(az).astype(np.float32)

    # 5. flat mask: no meaningful direction below FLAT_DEG
    flat = slope < FLAT_DEG
    sin_a[flat] = 0.0
    cos_a[flat] = 0.0

    return slope, sin_a, cos_a


def main():
    shape_of = lambda n: (n, 512, 512)

    # Welford-free accumulation for train slope stats (float64 is ample here)
    train_sum = 0.0
    train_sumsq = 0.0
    train_n = 0

    for split, n in SPLITS.items():
        folder = f"{DATA_ROOT}\\{split}"
        print(f"\n=== {split} ({n} patches) ===")

        elev = np.memmap(f"{folder}\\EXTRA_elevation.dat",
                         dtype="int16", mode="r", shape=shape_of(n))
        slope_mm = np.memmap(f"{folder}\\DERIVED_slope_v2.dat",
                             dtype="float32", mode="w+", shape=shape_of(n))
        sin_mm   = np.memmap(f"{folder}\\DERIVED_sin_aspect_v2.dat",
                             dtype="float32", mode="w+", shape=shape_of(n))
        cos_mm   = np.memmap(f"{folder}\\DERIVED_cos_aspect_v2.dat",
                             dtype="float32", mode="w+", shape=shape_of(n))

        for i in tqdm(range(n), desc=split):
            slope, sin_a, cos_a = slope_aspect_v2(elev[i])
            slope_mm[i] = slope
            sin_mm[i]   = sin_a
            cos_mm[i]   = cos_a
            if split == "Train":
                s64 = slope.astype(np.float64)
                train_sum   += s64.sum()
                train_sumsq += (s64 * s64).sum()
                train_n     += s64.size

        slope_mm.flush(); sin_mm.flush(); cos_mm.flush()

        # per-split sanity summary (sampled)
        s = np.stack([slope_mm[i] for i in range(0, n, max(n // 40, 1))]).ravel()
        print(f"  slope_v2 sample: median={np.median(s):.2f}  "
              f"frac<5deg={(s < 5).mean():.3f}  frac 5-45={((s >= 5) & (s <= 45)).mean():.3f}  "
              f"frac>45={(s > 45).mean():.3f}  max={s.max():.2f}")
        assert np.isfinite(s).all(), f"{split}: non-finite slope values!"

    mean = train_sum / train_n
    std = float(np.sqrt(train_sumsq / train_n - mean * mean))
    print("\n" + "=" * 60)
    print("v2 TRAIN slope stats for dataset.py (replaces 17.92 / 28.99):")
    print(f"  SLOPE_MEAN = {mean:.4f}")
    print(f"  SLOPE_STD  = {std:.4f}")
    print("=" * 60)
    print("sin/cos aspect stay unstandardized (bounded [-1,1]; flat pixels = 0).")


if __name__ == "__main__":
    main()
