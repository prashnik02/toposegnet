# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:38:07 2026

@author: prash
"""

import numpy as np

print("Loading elevation memmap...")
elev_train = np.memmap(
    r"C:\Users\prash\toposegnet\data\high_509\Train\EXTRA_elevation.dat",
    dtype='int16', mode='r', shape=(8490, 512, 512)
)

print("Computing stats (this may take a moment)...")
# Cast to float32 to avoid int16 overflow during computation
elev_f = elev_train[:].astype(np.float32)

mean = elev_f.mean()
std  = elev_f.std()
mn   = elev_f.min()
mx   = elev_f.max()

print(f"Elevation stats over full train set:")
print(f"  Mean : {mean:.4f}")
print(f"  Std  : {std:.4f}")
print(f"  Min  : {mn:.4f}")
print(f"  Max  : {mx:.4f}")