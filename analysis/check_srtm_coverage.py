"""Test the SRTM-coverage hypothesis: SRTM covers ~60N..56S. If patches
poleward of 60N have zero-filled elevation, every 'terrain' channel is
garbage exactly where snow is most common (high latitude)."""

import numpy as np
import pandas as pd

SHAPE = (975, 512, 512)
ROOT  = r"C:\Users\prash\toposegnet\data\high_509\Test"

elev = np.memmap(ROOT + r"\EXTRA_elevation.dat", dtype="int16", mode="r", shape=SHAPE)
df = pd.read_csv(r"C:\Users\prash\toposegnet\experiments\stage1a_error_analysis\per_patch.csv")

# fraction of exactly-zero elevation pixels per patch (sampled stride 4 for speed)
zero_frac = np.array([(elev[i][::4, ::4] == 0).mean() for i in range(975)])
df["elev_zero_frac"] = zero_frac

print("lat range in test split:", df.lat.min(), "to", df.lat.max())

for name, grp in [("lat > 60N",        df[df.lat > 60]),
                  ("lat 56S..60N",     df[(df.lat >= -56) & (df.lat <= 60)]),
                  ("lat < 56S",        df[df.lat < -56])]:
    if len(grp) == 0:
        print(f"{name:>14}: none")
        continue
    print(f"{name:>14}: n={len(grp):>4}  mean elev_zero_frac={grp.elev_zero_frac.mean():.3f}  "
          f"all-zero patches={int((grp.elev_zero_frac > 0.99).sum()):>3}  "
          f"mean snow_frac={grp.snow_frac.mean():.3f}  mean err={grp.err_rate.mean():.4f}")

# all-zero-elevation patches overall
az = df[df.elev_zero_frac > 0.99]
print(f"\nall-zero-elevation patches: {len(az)} of 975 "
      f"({100*len(az)/975:.1f}%), their mean lat = {az.lat.mean():.1f}, "
      f"mean snow_frac = {az.snow_frac.mean():.3f}, mean err = {az.err_rate.mean():.4f}")
nz = df[df.elev_zero_frac <= 0.99]
print(f"remaining patches:          mean err = {nz.err_rate.mean():.4f}")
