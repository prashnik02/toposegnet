"""
Fetch the Sen2Cor Scene Classification Layer (SCL) for every CloudSEN12+ patch
and store it as a memmap co-registered with the existing .dat files.


WHY
---
The `snow` stratum used so far is a homemade per-pixel NDSI rule evaluated
THROUGH cloud (see project notes, SNOW-FREE RE-ANALYSIS → OPEN ITEM). SCL class 11
("snow or ice") from ESA's own L2A processor on the SAME acquisition is the
standard, citable replacement. The CloudSEN12+ bundled `cloudmask_sen2cor` is
useless for this — it was collapsed to the 4-class CloudSEN12 schema, so snow
is gone. Hence we go back to the source product.

HOW
---
For each patch (row i of metadata.csv ↔ memmap index i):
  1. Parse the L1C id, e.g.
        S2A_MSIL1C_20190212T142031_N0207_R010_T19FDF_20190212T191443
     → mission S2A, sensing 20190212T142031, orbit R010, tile T19FDF.
     The matching L2A product shares everything except level, baseline and
     processing time:  S2A_MSIL2A_20190212T142031_R010_T19FDF_<proc>  (PC id).
  2. STAC search Planetary Computer `sentinel-2-l2a` at the patch centroid,
     ±1 h around sensing time; keep items whose id starts with that prefix
     (latest processing time wins if several).
  3. Read the 20 m SCL COG through a WarpedVRT onto the patch's own 10 m grid
     (proj_epsg + proj_transform, 509×509, nearest) — only the overlapping COG
     blocks are downloaded, never the full 110 km tile.
  4. Write into AUX_SCL.dat at [1:510, 1:510] — the same 1-px top/left,
     2-px bottom/right padding the CloudSEN12+ memmaps use (verified on all 975
     Test patches 2026-10-01: row/col 0 and 510–511 are zero, 1 and 509 are not).
     Padding and unmatched patches stay 0 = SCL "No data".

OUTPUTS (in data/high_509/<split>/)
-------
  AUX_SCL.dat      uint8 memmap (N, 512, 512), SCL codes 0–11
                   (0 no-data · 1 saturated · 2 dark/topo-shadow · 3 cloud shadow
                    4 vegetation · 5 not-vegetated · 6 water · 7 unclassified
                    8 cloud med · 9 cloud high · 10 thin cirrus · 11 SNOW/ICE)
  AUX_SCL_log.csv  one row per patch, written incrementally:
                   status, matched L2A id, processing baseline, and
                   snow_frac_clear = SCL-snow share of GT-`clear` pixels — the
                   SCENE-level "is the visible ground snowy?" number the
                   stratification needs — plus cloud_agree, an alignment
                   sanity check (share of GT-thick px SCL also calls cloud).

RESUMABLE: the memmap is flushed and the log row appended after every patch,
so a sleep/kill loses at most one patch (three such incidents on record).
Re-running skips patches already logged `ok` / `no_item`; --retry-missing
re-attempts `no_item` and `error` rows.

USAGE (toposegnet conda env)
-----
  python src/fetch_scl.py --dry-run --limit 20   # search only, no raster reads
  python src/fetch_scl.py --limit 5              # small real test
  python src/fetch_scl.py                        # full Test split (default)

Size: AUX_SCL.dat is N×512×512 bytes (Test: 975 → 256 MB on disk). Network
transfer is the overlapping compressed COG blocks only — order of 0.1–0.5 MB
per patch.
Coverage (dry run 2026-10-01): PC serves REPROCESSED L2A (processing dates
2020–2022, not ESA's original products), so even the 38 Test patches that
predate global L2A (Dec 2018) all match (baseline 02.12). Cite the SCL as
"Sen2Cor L2A, Microsoft Planetary Computer reprocessing" and report the
baselines from the log. Any `no_item` that does occur: report as missing,
do not impute.
"""
import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

DATA_ROOT     = r"C:\Users\prash\toposegnet\data\high_509"
STAC_SEARCH   = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
COLLECTION    = "sentinel-2-l2a"

FRAME, PAD, N_PX = 512, 1, 509          # data lives at [PAD:PAD+N_PX] in a FRAME² memmap
OUT_NAME, LOG_NAME = "AUX_SCL.dat", "AUX_SCL_log.csv"
SCL_NODATA, SCL_SNOW = 0, 11
SCL_CLOUD = (8, 9, 10)                  # cloud medium, cloud high, thin cirrus
GT_CLEAR, GT_THICK = 0, 1               # LABEL_manual_hq codes
TOKEN_TTL_S = 30 * 60                   # refresh the PC SAS token well before expiry

LOG_FIELDS = ["idx", "s2_id", "status", "l2a_id", "processing_baseline",
              "n_candidates", "valid_frac", "scl_snow_frac", "scl_cloud_frac",
              "n_gt_clear", "snow_frac_clear", "cloud_agree", "error"]

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",   # don't list blob containers
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                VSI_CACHE="TRUE")


# ── network helpers ──────────────────────────────────────────────────────────
def http_json(url, payload=None, retries=4):
    """GET (payload None) or POST JSON, with exponential back-off."""
    for attempt in range(retries):
        try:
            data = None if payload is None else json.dumps(payload).encode()
            hdrs = {} if data is None else {"Content-Type": "application/json"}
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=hdrs),
                                        timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * 2 ** attempt)


class SasToken:
    """Planetary Computer blob reads need a short-lived SAS token (no account)."""
    def __init__(self):
        self.token, self.t = None, 0.0

    def get(self, force=False):
        if force or self.token is None or time.time() - self.t > TOKEN_TTL_S:
            self.token, self.t = http_json(SAS_TOKEN_URL)["token"], time.time()
        return self.token


# ── matching ─────────────────────────────────────────────────────────────────
def l2a_prefix(s2_id):
    """L1C product id → (PC L2A id prefix, sensing datetime)."""
    mission, _level, sensing, _baseline, orbit, tile, _proc = s2_id.split("_")
    return (f"{mission}_MSIL2A_{sensing}_{orbit}_{tile}_",
            datetime.strptime(sensing, "%Y%m%dT%H%M%S"))


def parse_point(wkt):
    """'POINT (-69.78524 -49.00794)' → [lon, lat] (metadata centroid is WGS84)."""
    lon, lat = wkt.strip()[wkt.index("(") + 1:-1].split()
    return [float(lon), float(lat)]


def find_item(s2_id, centroid_wkt):
    prefix, t = l2a_prefix(s2_id)
    payload = {"collections": [COLLECTION],
               "intersects": {"type": "Point", "coordinates": parse_point(centroid_wkt)},
               "datetime": f"{(t - timedelta(hours=1)).isoformat()}Z/"
                           f"{(t + timedelta(hours=1)).isoformat()}Z",
               "limit": 50}
    feats = http_json(STAC_SEARCH, payload).get("features", [])
    hits = sorted((f for f in feats if f["id"].startswith(prefix)),
                  key=lambda f: f["id"].rsplit("_", 1)[-1])   # processing time
    return (hits[-1] if hits else None), len(feats)


# ── raster read ──────────────────────────────────────────────────────────────
def read_scl(href, token, epsg, transform_str):
    """Warp the 20 m SCL onto the patch's 10 m grid → (509, 509) uint8."""
    # metadata order "10, 0, 440030, 0, -10, 4573910" == Affine(a, b, c, d, e, f)
    dst_transform = Affine(*[float(v) for v in transform_str.split(",")])
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(f"{href}?{token}") as src:
            with WarpedVRT(src, crs=CRS.from_epsg(int(epsg)), transform=dst_transform,
                           width=N_PX, height=N_PX, resampling=Resampling.nearest,
                           nodata=SCL_NODATA) as vrt:
                return vrt.read(1).astype(np.uint8)


def patch_stats(scl, label):
    valid = scl != SCL_NODATA
    clear = (label == GT_CLEAR) & valid
    thick = (label == GT_THICK) & valid
    n_clear = int(clear.sum())
    return dict(
        valid_frac=round(float(valid.mean()), 4),
        scl_snow_frac=round(float((scl == SCL_SNOW).mean()), 4),
        scl_cloud_frac=round(float(np.isin(scl, SCL_CLOUD).mean()), 4),
        n_gt_clear=n_clear,
        snow_frac_clear=(round(float(((scl == SCL_SNOW) & clear).sum() / n_clear), 4)
                         if n_clear else ""),          # undefined on fully cloudy patches
        cloud_agree=(round(float((np.isin(scl, SCL_CLOUD) & thick).sum() / thick.sum()), 4)
                     if thick.any() else ""),
    )


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--split", default="Test", choices=["Train", "Val", "Test"])
    ap.add_argument("--limit", type=int, default=None, help="process only the first N pending patches")
    ap.add_argument("--dry-run", action="store_true", help="STAC search only; no raster reads, nothing written")
    ap.add_argument("--retry-missing", action="store_true", help="re-attempt rows logged no_item / error")
    args = ap.parse_args()

    folder = os.path.join(DATA_ROOT, args.split)
    meta = pd.read_csv(os.path.join(folder, "metadata.csv"))
    n = len(meta)
    # `index` is the GLOBAL id across all 10,000 CloudSEN12+ samples, not 0..n-1; rows are in memmap
    # order iff it is strictly increasing (same positional assumption dataset.py:140 makes for lat/lon)
    assert (np.diff(meta["index"].to_numpy()) > 0).all(), "metadata rows not in memmap order"
    assert (meta["proj_shape"] == N_PX).all(), "expected 509-px patches"

    labels = np.memmap(os.path.join(folder, "LABEL_manual_hq.dat"), dtype="int8",
                       mode="r", shape=(n, FRAME, FRAME))
    out_path, log_path = os.path.join(folder, OUT_NAME), os.path.join(folder, LOG_NAME)

    done = {}
    if os.path.exists(log_path):
        for r in csv.DictReader(open(log_path, newline="")):
            done[int(r["idx"])] = r["status"]
    skip = {"ok"} if args.retry_missing else {"ok", "no_item"}
    pending = [i for i in range(n) if done.get(i) not in skip]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{args.split}: {n} patches · {len(done)} already logged · {len(pending)} to process"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    if not pending:
        return

    out = None
    if not args.dry_run:
        out = np.memmap(out_path, dtype="uint8", mode="r+" if os.path.exists(out_path) else "w+",
                        shape=(n, FRAME, FRAME))
    log_f = None if args.dry_run else open(log_path, "a", newline="")
    writer = None
    if log_f is not None:
        writer = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
        if log_f.tell() == 0:
            writer.writeheader()

    token, counts, t0 = SasToken(), {}, time.time()
    for k, i in enumerate(pending, 1):
        row = meta.iloc[i]
        rec = dict(idx=i, s2_id=row["s2_id"])
        try:
            item, n_cand = find_item(row["s2_id"], row["proj_centroid"])
            rec["n_candidates"] = n_cand
            if item is None:
                rec["status"] = "no_item"
            else:
                rec.update(l2a_id=item["id"],
                           processing_baseline=item["properties"].get("s2:processing_baseline", ""))
                if args.dry_run:
                    rec["status"] = "match"
                else:
                    href = item["assets"]["SCL"]["href"]
                    try:
                        scl = read_scl(href, token.get(), row["proj_epsg"], row["proj_transform"])
                    except rasterio.errors.RasterioIOError:      # expired token → refresh once
                        scl = read_scl(href, token.get(force=True), row["proj_epsg"], row["proj_transform"])
                    out[i, PAD:PAD + N_PX, PAD:PAD + N_PX] = scl
                    out.flush()
                    rec.update(patch_stats(scl, labels[i, PAD:PAD + N_PX, PAD:PAD + N_PX]))
                    rec["status"] = "ok"
        except Exception as e:                                   # log and move on; --retry-missing later
            rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300])

        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
        if writer is not None:
            writer.writerow(rec)
            log_f.flush()
        if args.dry_run or k % 25 == 0 or rec["status"] != "ok":
            rate = (time.time() - t0) / k
            print(f"  [{k}/{len(pending)}] idx {i} {rec['status']:8s} {rec.get('l2a_id', '')} "
                  f"{rec.get('error', '')}  ({rate:.1f} s/patch)")

    if log_f is not None:
        log_f.close()
    print(f"\nthis run: {counts}")

    if not args.dry_run:                                         # whole-split summary from the log
        log = pd.read_csv(log_path).drop_duplicates("idx", keep="last")
        ok = log[log.status == "ok"]
        print(f"split status: {log.status.value_counts().to_dict()}  (of {n})")
        if len(ok):
            agree = pd.to_numeric(ok.cloud_agree, errors="coerce").dropna()
            snow = pd.to_numeric(ok.snow_frac_clear, errors="coerce").dropna()
            print(f"alignment check — median share of GT-thick px SCL calls cloud: {agree.median():.3f} "
                  f"(n={len(agree)}; a value near 0 would mean the grids are misregistered)")
            print(f"patches with visible-ground snow fraction ≥ 0.10: {(snow >= 0.10).sum()} / {len(snow)}")


if __name__ == "__main__":
    main()
