"""Estimate the staircase block size of EXTRA_elevation.dat.
The DEM was nearest-neighbour resampled from a coarser grid; the run length
of identical values along a row tells us the upsampling factor, which sets
the Gaussian sigma for smoothing before slope/aspect recomputation."""

import numpy as np

elev = np.memmap(r"C:\Users\prash\toposegnet\data\high_509\Train\EXTRA_elevation.dat",
                 dtype="int16", mode="r", shape=(8490, 512, 512))


def run_lengths(row):
    """lengths of consecutive identical values in a 1D array"""
    change = np.flatnonzero(np.diff(row)) + 1
    edges = np.concatenate([[0], change, [len(row)]])
    return np.diff(edges)


meds = []
for i in range(0, 8490, 100):                    # 85 patches
    e = elev[i]
    if e.max() - e.min() < 200:                  # need relief for meaningful runs
        continue
    rl = np.concatenate([run_lengths(e[r]) for r in range(0, 512, 64)])
    rl = rl[rl < 512]                            # drop full-row constants (ocean)
    if rl.size:
        meds.append(np.median(rl))

meds = np.array(meds)
print(f"patches checked: {len(meds)}")
print(f"median run length per patch: p10={np.percentile(meds,10):.1f} "
      f"p50={np.percentile(meds,50):.1f} p90={np.percentile(meds,90):.1f}")
print(f"(run length k => source DEM ~ k*10 m; e.g. 3 => 30 m SRTM, 9 => 90 m)")
