import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
DATA_ROOT  = r"C:\Users\prash\toposegnet\data\high_509"

SPLIT_SIZES = {"Train": 8490, "Val": 535, "Test": 975}

BAND_FILES = [
    "L1C_B1.dat",  "L1C_B2.dat",  "L1C_B3.dat",  "L1C_B4.dat",
    "L1C_B5.dat",  "L1C_B6.dat",  "L1C_B7.dat",  "L1C_B8.dat",
    "L1C_B8A.dat", "L1C_B9.dat",  "L1C_B10.dat", "L1C_B11.dat",
    "L1C_B12.dat",
]

# normalization constants (computed from train split only)
S2_SCALE   = 10000.0
ELEV_MEAN  = 683.6302;  ELEV_STD   = 1025.6890
# Slope stats are VERSION-SPECIFIC. v1 = original staircase-DEM derivation
# (corrupted, see project notes); v2 = regenerated 2026-07-12 (correct 10 m spacing
# + adaptive smoothing). A "_v2" terrain_mode loads DERIVED_slope_v2.dat and
# MUST use the v2 stats — never mix v1 data with v2 stats or vice versa.
SLOPE_MEAN_V1 = 17.9182;  SLOPE_STD_V1 = 28.9855
SLOPE_MEAN_V2 =  8.6778;  SLOPE_STD_V2 = 16.8760

# ── ablation modes ───────────────────────────────────────────────────────────
# "none"    → 13 channels
# "elev"    → 14 channels
# "optionA" → 15 channels (elev + slope, no aspect)
# "optionB" → 17 channels (elev + slope + sin_asp + cos_asp)
# "optionC" → 19 channels (optionB + lat + lon)
# The "_v2" variants (optionA_v2, optionB_v2, optionC_v2) stack the SAME
# channels but source slope/aspect from the regenerated clean memmaps. Channel
# count is unchanged, so a v2 run is directly comparable to its v1 counterpart.
# "latlon_only" → 15 channels (13 bands + lat + lon broadcast planes, NO other
# terrain). Stage 5b(1) diagnostic: isolates the patch-level lat/lon signal via
# naive concat, methodologically parallel to Stage 2a (one variable vs baseline).
# "elev_pooled" → 14 channels, IDENTICAL to "elev" except the elevation plane is
# replaced by its own PATCH MEAN, broadcast back to 512x512. The control for the
# gap_terrain ablation (2026-08-13): gap beat concat while differing in TWO ways
# at once — pooled-not-per-pixel AND separate-branch-not-input-channel. This mode
# holds the delivery route fixed (still a concatenated input channel, still a
# plain U-Net) and varies ONLY pooled-vs-per-pixel, so the two are separable.
# Pooling before vs after standardisation is equivalent (standardisation is
# affine), so this is exactly mean(elev) pushed through the same normalisation.
# "elev_latlon" → 16 channels: 13 bands + per-pixel elevation + lat + lon. Feeds
# arch="gap_latlon": elevation -> gap terrain branch, lat/lon -> bottleneck FiLM.
# Elevation is byte-identical to "elev" and lat/lon to "latlon_only", so the only
# new thing vs gap is location via FiLM.
TERRAIN_MODES = ["none", "elev", "optionA", "optionB", "optionC",
                 "optionA_v2", "optionB_v2", "optionC_v2",
                 "latlon_only",   # Stage 5b(1) diagnostic mode
                 "elev_pooled",   # pooled-elevation control
                 "elev_latlon",   # gap + lat/lon FiLM exploratory run
                 "rgbn",          # spectral-ablation follow-up: Gaofen-1-equivalent 4 bands only
                 "rgbn_elev"]     # same 4 bands + per-pixel elevation

# Spectral-ablation follow-up (project notes ACTIVE PLAN, 2026-10-10): H-spec says
# elevation only adds information when the spectrum cannot separate cloud from
# snow itself. "rgbn" keeps just the Gaofen-1 WFV-equivalent bands — B2 blue,
# B3 green, B4 red, B8 NIR (B8 not B8A: its 785-899 nm matches GF-1 NIR) — i.e.
# no cirrus (B10) and no SWIR (B11/B12). Bands are selected by index from the
# same memmaps with the same /10000 scaling, so no new data files and no
# resolution/bit-depth confound.
RGBN_BAND_IDX = [1, 2, 3, 7]

# ── augmentations ─────────────────────────────────────────────────────────────
def get_transforms(split, augment=True):
    if split == "Train" and augment:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
    return None   # no augmentation for val/test or when augment=False

# ── dataset ───────────────────────────────────────────────────────────────────
class CloudSEN12Dataset(Dataset):
    """
    Args:
        split        : "Train" | "Val" | "Test"
        terrain_mode : "none" | "elev" | "optionA" | "optionB" | "optionC"
                       | "*_v2" variants | "latlon_only"
        augment      : True/False — only applies to Train split

    Returns:
        image : float32 tensor (C, 512, 512)
                C = 13/14/15/17/19 depending on terrain_mode
                ("latlon_only" = 15: 13 bands + lat + lon)
        label : long tensor (512, 512), values 0-3
    """

    def __init__(self, split="Train", terrain_mode="none", augment=True):
        assert split        in SPLIT_SIZES,   f"split must be one of {list(SPLIT_SIZES)}"
        assert terrain_mode in TERRAIN_MODES, f"terrain_mode must be one of {TERRAIN_MODES}"

        self.split        = split
        self.terrain_mode = terrain_mode
        self.n            = SPLIT_SIZES[split]
        self.transforms   = get_transforms(split, augment)

        # v2 modes source clean slope/aspect files + their matching stats.
        # The base mode name (with "_v2" stripped) decides WHICH channels stack.
        self.v2      = terrain_mode.endswith("_v2")
        base_mode    = terrain_mode[:-3] if self.v2 else terrain_mode
        slope_fname  = "DERIVED_slope_v2.dat"      if self.v2 else "DERIVED_slope.dat"
        sin_fname    = "DERIVED_sin_aspect_v2.dat" if self.v2 else "DERIVED_sin_aspect.dat"
        cos_fname    = "DERIVED_cos_aspect_v2.dat" if self.v2 else "DERIVED_cos_aspect.dat"
        self.slope_mean = SLOPE_MEAN_V2 if self.v2 else SLOPE_MEAN_V1
        self.slope_std  = SLOPE_STD_V2  if self.v2 else SLOPE_STD_V1

        folder = f"{DATA_ROOT}\\{split}"
        shape  = (self.n, 512, 512)

        # ── spectral bands ───────────────────────────────────────────────────
        self.bands = [
            np.memmap(f"{folder}\\{f}", dtype='uint16', mode='r', shape=shape)
            for f in BAND_FILES
        ]

        # ── terrain memmaps ──────────────────────────────────────────────────
        self.elev    = None
        self.slope   = None
        self.sin_asp = None
        self.cos_asp = None
        self.lats    = None
        self.lons    = None

        # elev_pooled loads the SAME elevation memmap as elev; the pooling is
        # applied per patch in __getitem__.
        self.elev_pooled = (base_mode == "elev_pooled")
        # which spectral bands enter the tensor: all 13, or the 4 GF-1-equivalent
        # ones for the spectral-ablation modes
        self.band_idx = RGBN_BAND_IDX if base_mode in ["rgbn", "rgbn_elev"] else list(range(len(BAND_FILES)))

        if base_mode in ["elev", "elev_pooled", "elev_latlon", "rgbn_elev", "optionA", "optionB", "optionC"]:   # elev_latlon reuses the elev pipeline; rgbn_elev likewise, so its elevation plane is byte-identical to elev's
            self.elev = np.memmap(f"{folder}\\EXTRA_elevation.dat",
                                  dtype='int16', mode='r', shape=shape)

        if base_mode in ["optionA", "optionB", "optionC"]:
            self.slope = np.memmap(f"{folder}\\{slope_fname}",
                                   dtype='float32', mode='r', shape=shape)

        if base_mode in ["optionB", "optionC"]:
            self.sin_asp = np.memmap(f"{folder}\\{sin_fname}",
                                     dtype='float32', mode='r', shape=shape)
            self.cos_asp = np.memmap(f"{folder}\\{cos_fname}",
                                     dtype='float32', mode='r', shape=shape)

        if base_mode in ["optionC", "latlon_only", "elev_latlon"]:   # latlon_only reuses optionC's lat/lon pipeline with no other terrain; elev_latlon likewise
            meta = pd.read_csv(f"{folder}\\metadata.csv")
            centroids = meta["proj_centroid"].str.extract(
                r'POINT \(([0-9.\-]+) ([0-9.\-]+)\)'
            )
            self.lons = centroids[0].astype(float).values  # (n,)
            self.lats = centroids[1].astype(float).values  # (n,)

        # ── label ────────────────────────────────────────────────────────────
        self.label = np.memmap(f"{folder}\\LABEL_manual_hq.dat",
                               dtype='int8', mode='r', shape=shape)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # ── load spectral patch → (512, 512, 13) ────────────────────────────
        image = np.stack([self.bands[i][idx] for i in self.band_idx], axis=-1).astype(np.float32)   # band_idx = all 13, or the 4 rgbn bands
        label = self.label[idx].astype(np.int64)

        # ── collect terrain as extra masks for albumentations ────────────────
        # albumentations applies identical spatial transforms to all masks
        extra_masks = []

        if self.elev is not None:
            e = self.elev[idx].astype(np.float32)
            if self.elev_pooled:
                # replace the plane by its patch mean -> the encoder gets the
                # SAME scene-level elevation summary the gap branch pools, but
                # delivered as a concatenated input channel. Pooling here (before
                # standardisation) is equivalent to pooling after, since the
                # normalisation is affine.
                e = np.full_like(e, float(e.mean()))
            extra_masks.append(e)
        if self.slope is not None:
            extra_masks.append(self.slope[idx].copy())
        if self.sin_asp is not None:
            extra_masks.append(self.sin_asp[idx].copy())
            extra_masks.append(self.cos_asp[idx].copy())

        # ── augment (train only) ─────────────────────────────────────────────
        if self.transforms is not None:
            out         = self.transforms(image=image, mask=label, masks=extra_masks)
            image       = out["image"]
            label       = out["mask"]
            extra_masks = out["masks"]

        # ── normalize spectral ───────────────────────────────────────────────
        image = image / S2_SCALE   # → ~0.0 to ~2.8

        # ── normalize terrain ────────────────────────────────────────────────
        terrain_channels = []
        ptr = 0

        if self.elev is not None:
            terrain_channels.append((extra_masks[ptr] - ELEV_MEAN) / ELEV_STD)
            ptr += 1

        if self.slope is not None:
            terrain_channels.append((extra_masks[ptr] - self.slope_mean) / self.slope_std)
            ptr += 1

        if self.sin_asp is not None:
            terrain_channels.append(extra_masks[ptr])      # sin aspect, already [-1,1]
            terrain_channels.append(extra_masks[ptr + 1])  # cos aspect, already [-1,1]
            ptr += 2

        if self.lats is not None:
            h, w  = label.shape
            lat_n = self.lats[idx] / 90.0    # normalize to [-1, 1]
            lon_n = self.lons[idx] / 180.0   # normalize to [-1, 1]
            terrain_channels.append(np.full((h, w), lat_n, dtype=np.float32))
            terrain_channels.append(np.full((h, w), lon_n, dtype=np.float32))

        # ── to tensor → (C, 512, 512) ────────────────────────────────────────
        image_t = torch.from_numpy(image.transpose(2, 0, 1))  # (13, 512, 512)

        if terrain_channels:
            terrain_t = torch.from_numpy(
                np.stack(terrain_channels, axis=0)            # (T, 512, 512)
            )
            image_t = torch.cat([image_t, terrain_t], dim=0) # (13+T, 512, 512)

        label_t = torch.from_numpy(label)                     # (512, 512)

        return image_t, label_t