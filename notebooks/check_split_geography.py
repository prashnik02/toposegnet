# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:28:50 2026

@author: prash
"""

"""
Check for geographic leakage across train/val/test splits in CloudSEN12+.

Run this on your local machine (toposegnet env) where the actual
metadata.csv files live, e.g.:

    python check_split_geography.py \
        --train train/metadata.csv \
        --val val/metadata.csv \
        --test test/metadata.csv

What it checks:
1. ROI overlap: do any roi_id values appear in more than one split?
   (would be direct leakage, not just proximity)
2. Geographic proximity: for each val/test centroid, what's the distance
   to the nearest train centroid? If val/test patches sit right next to
   train patches, lat/lon could act as a near-lookup-table key rather
   than a genuine terrain-adjacent feature.

Distances are computed in degrees first (fast, for the haversine calc)
then converted to km via haversine — good enough for a diagnostic, not
survey-grade.
"""

import argparse
import numpy as np
import pandas as pd
from shapely import wkt


def load_centroids(path):
    df = pd.read_csv(path)
    centroids = df['proj_centroid'].apply(wkt.loads)
    lon = centroids.apply(lambda p: p.x).values.astype(np.float64)
    lat = centroids.apply(lambda p: p.y).values.astype(np.float64)
    roi_id = df['roi_id'].values
    return lat, lon, roi_id


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km. lat1/lon1 are arrays (N,),
    lat2/lon2 are arrays (M,). Returns (N, M) distance matrix."""
    R = 6371.0
    lat1r, lon1r = np.radians(lat1)[:, None], np.radians(lon1)[:, None]
    lat2r, lon2r = np.radians(lat2)[None, :], np.radians(lon2)[None, :]
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def check_split(name, lat, lon, roi_id, train_lat, train_lon, train_roi):
    print(f"\n--- {name} vs train ---")

    # 1. Direct ROI overlap
    overlap = set(roi_id) & set(train_roi)
    if overlap:
        print(f"  [WARNING] {len(overlap)} roi_id values appear in BOTH "
              f"{name} and train: {sorted(overlap)[:10]}"
              f"{' ...' if len(overlap) > 10 else ''}")
    else:
        print(f"  OK: no roi_id overlap between {name} and train.")

    # 2. Nearest-train-centroid distance for each patch in this split
    # Batch to avoid huge memory if patch counts are large
    batch = 200
    min_dists = np.empty(len(lat))
    for i in range(0, len(lat), batch):
        d = haversine_km(lat[i:i+batch], lon[i:i+batch], train_lat, train_lon)
        min_dists[i:i+batch] = d.min(axis=1)

    print(f"  Nearest-train-centroid distance (km): "
          f"min={min_dists.min():.2f}, median={np.median(min_dists):.2f}, "
          f"max={min_dists.max():.2f}")

    thresholds_km = [1, 5, 10, 50]
    for t in thresholds_km:
        frac = (min_dists < t).mean()
        print(f"  Fraction of {name} patches within {t:>3} km of a train patch: "
              f"{frac*100:.1f}%")

    return min_dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--test', required=True)
    args = ap.parse_args()

    train_lat, train_lon, train_roi = load_centroids(args.train)
    val_lat, val_lon, val_roi = load_centroids(args.val)
    test_lat, test_lon, test_roi = load_centroids(args.test)

    print(f"Train patches: {len(train_lat)} | unique ROIs: {len(set(train_roi))}")
    print(f"Val patches:   {len(val_lat)} | unique ROIs: {len(set(val_roi))}")
    print(f"Test patches:  {len(test_lat)} | unique ROIs: {len(set(test_roi))}")

    check_split("val", val_lat, val_lon, val_roi, train_lat, train_lon, train_roi)
    check_split("test", test_lat, test_lon, test_roi, train_lat, train_lon, train_roi)


if __name__ == "__main__":
    main()


def run_in_notebook(train_path, val_path, test_path):
    """
    Call this directly from a Jupyter cell instead of running the script
    from the command line, e.g.:

        from check_split_geography import run_in_notebook
        run_in_notebook("train/metadata.csv", "val/metadata.csv", "test/metadata.csv")
    """
    train_lat, train_lon, train_roi = load_centroids(train_path)
    val_lat, val_lon, val_roi = load_centroids(val_path)
    test_lat, test_lon, test_roi = load_centroids(test_path)

    print(f"Train patches: {len(train_lat)} | unique ROIs: {len(set(train_roi))}")
    print(f"Val patches:   {len(val_lat)} | unique ROIs: {len(set(val_roi))}")
    print(f"Test patches:  {len(test_lat)} | unique ROIs: {len(set(test_roi))}")

    check_split("val", val_lat, val_lon, val_roi, train_lat, train_lon, train_roi)
    check_split("test", test_lat, test_lon, test_roi, train_lat, train_lon, train_roi)