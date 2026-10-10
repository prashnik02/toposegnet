"""Confirm the staircase-DEM slope artifact also affects the Train split
(the ablation stages trained on it)."""
import numpy as np

slope = np.memmap(r"C:\Users\prash\toposegnet\data\high_509\Train\DERIVED_slope.dat",
                  dtype="float32", mode="r", shape=(8490, 512, 512))
elev  = np.memmap(r"C:\Users\prash\toposegnet\data\high_509\Train\EXTRA_elevation.dat",
                  dtype="int16",  mode="r", shape=(8490, 512, 512))

s = np.stack([slope[i] for i in range(0, 8490, 200)]).ravel()   # 43-patch sample
print(f"Train slope sample: frac<5deg={(s<5).mean():.4f}  "
      f"frac 5..45={((s>=5)&(s<=45)).mean():.4f}  frac>45={(s>45).mean():.4f}  "
      f"median={np.percentile(s,50):.2f}")

# staircase: fraction of identical horizontal neighbours among high-relief patches
checked = 0
for i in range(0, 8490, 200):
    e = elev[i].astype(np.float32)
    if e.max() - e.min() > 500:          # only meaningful on real relief
        d = np.abs(np.diff(e, axis=1))
        print(f"  patch {i:>5}: relief {e.max()-e.min():6.0f} m, "
              f"frac identical horiz neighbours = {(d==0).mean():.3f}")
        checked += 1
        if checked >= 5:
            break
