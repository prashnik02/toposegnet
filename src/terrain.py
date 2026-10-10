"""
terrain.py
----------
Terrain preprocessing utilities for TopoSegNet.

Responsibilities:
  1. Load a DEM GeoTIFF file
  2. Compute slope and aspect from raw elevation
  3. Align (reproject + resample) DEM to match a satellite image exactly
  4. Normalise all three channels to [0, 1] using fixed physical ranges
  5. Save processed terrain channels to disk
  6. Quick visualisation for sanity checking

Run this file directly to test with a dummy DEM:
  python src/terrain.py
"""

import os
import numpy as np
import rasterio  #Used for procesing DEM files
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import richdem as rd


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_dem(dem_path):
    """
    Load a DEM GeoTIFF from disk.

    Args:
        dem_path : path to .tif DEM file

    Returns:
        elevation : 2D numpy array of elevation values in metres  (H, W)
        meta      : rasterio metadata dict (crs, transform, etc.)
    """
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float32)  # band 1 = elevation
        meta = src.meta.copy()

    # Replace no-data values with 0 (ocean / missing)
    no_data = meta.get("nodata", -32768)
    if no_data is not None:
        elevation[elevation == no_data] = 0.0

    return elevation, meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE SLOPE AND ASPECT
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope(elevation):
    """
    Compute terrain slope in degrees from a 2D elevation array.

    Slope = how steep the terrain is at each pixel.
    0  = perfectly flat
    90 = vertical cliff

    Uses richdem which implements Horn's (1981) method —
    the same algorithm used in GIS software like QGIS and ArcGIS.

    Args:
        elevation : 2D numpy float32 array (H, W) in metres

    Returns:
        slope : 2D numpy float32 array (H, W) in degrees [0, 90]
    """
    dem_rd = rd.rdarray(elevation, no_data=-9999)
    slope  = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
    slope  = np.array(slope, dtype=np.float32)

    # Clip to valid range — richdem can produce tiny negatives at edges
    slope  = np.clip(slope, 0.0, 90.0)
    return slope


def compute_aspect(elevation):
    """
    Compute terrain aspect in degrees from a 2D elevation array.

    Aspect = which compass direction the slope faces.
    0   = north facing
    90  = east facing
    180 = south facing
    270 = west facing

    Why aspect matters for snow:
      North-facing slopes (in northern hemisphere) receive less solar
      radiation, are cooler, and snow persists longer.
      South-facing slopes warm faster and snow melts sooner.
      The model can learn this physical relationship from aspect maps.

    Args:
        elevation : 2D numpy float32 array (H, W) in metres

    Returns:
        aspect : 2D numpy float32 array (H, W) in degrees [0, 360]
    """
    dem_rd = rd.rdarray(elevation, no_data=-9999)
    aspect = rd.TerrainAttribute(dem_rd, attrib='aspect')
    aspect = np.array(aspect, dtype=np.float32)

    # richdem returns -1 for flat areas — set to 0 (flat = no meaningful aspect)
    aspect[aspect < 0] = 0.0
    aspect = np.clip(aspect, 0.0, 360.0)
    return aspect


# ─────────────────────────────────────────────────────────────────────────────
# 3. ALIGN DEM TO SATELLITE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def align_dem_to_image(dem_path, image_path, output_path=None):
    """
    Reproject and resample DEM to exactly match a satellite image.

    This is the most important function in terrain.py.
    The DEM and satellite image may be in different coordinate reference
    systems (CRS), have different spatial resolutions (30m DEM vs 16m image),
    and cover slightly different extents.

    This function warps the DEM so every DEM pixel corresponds
    exactly to the matching satellite image pixel.

    Args:
        dem_path    : path to source DEM GeoTIFF
        image_path  : path to satellite image GeoTIFF (the target grid)
        output_path : if given, save aligned DEM here; else return array only

    Returns:
        aligned_elevation : 2D numpy float32 array matching image grid (H, W)
        image_meta        : metadata of the satellite image
    """
    # Read satellite image metadata — this defines the TARGET grid
    with rasterio.open(image_path) as img_src:
        image_meta       = img_src.meta.copy()
        target_crs       = img_src.crs
        target_transform = img_src.transform
        target_height    = img_src.height
        target_width     = img_src.width

    # Read source DEM
    with rasterio.open(dem_path) as dem_src:
        source_crs       = dem_src.crs
        source_transform = dem_src.transform
        source_data      = dem_src.read(1).astype(np.float32)
        source_nodata    = dem_src.nodata or -32768

    # Allocate output array — same shape as satellite image
    aligned = np.zeros((target_height, target_width), dtype=np.float32)

    # Reproject DEM onto satellite image grid
    reproject(
        source=source_data,
        destination=aligned,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,  # bilinear for continuous elevation
        src_nodata=source_nodata,
        dst_nodata=0.0
    )

    # Optionally save to disk
    if output_path is not None:
        out_meta = image_meta.copy()
        out_meta.update({"count": 1, "dtype": "float32", "nodata": 0.0})
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(aligned, 1)
        print(f"Saved aligned DEM to {output_path}")

    return aligned, image_meta


# ─────────────────────────────────────────────────────────────────────────────
# 4. NORMALISE
# ─────────────────────────────────────────────────────────────────────────────

def normalize_terrain(altitude, slope, aspect):
    """
    Normalise terrain channels to [0, 1] using FIXED physical ranges.

    CRITICAL DESIGN DECISION:
    We use fixed global physical ranges, NOT per-image min-max normalisation.

    Why? If we normalised per-image:
      - A scene at 4000m and a scene at 400m both become [0, 1]
      - The model loses absolute elevation information entirely
      - A pixel at 4000m looks identical to a pixel at 400m
      - This destroys the most important discriminating signal

    Fixed ranges preserve physical meaning:
      - A pixel normalised to 0.45 always means ~4000m elevation
      - The model can learn "0.45+ altitude means snow is possible"

    Ranges:
      Altitude : 0m (sea level) to 8848m (Everest)
      Slope    : 0 (flat) to 90 (vertical)
      Aspect   : 0 to 360 (full compass)

    Args:
        altitude : 2D numpy array in metres
        slope    : 2D numpy array in degrees [0, 90]
        aspect   : 2D numpy array in degrees [0, 360]

    Returns:
        alt_norm, slp_norm, asp_norm : each 2D float32 array in [0, 1]
    """
    alt_norm = np.clip(altitude, 0.0, 8848.0) / 8848.0
    slp_norm = np.clip(slope,    0.0,   90.0) /   90.0
    asp_norm = np.clip(aspect,   0.0,  360.0) /  360.0

    return (
        alt_norm.astype(np.float32),
        slp_norm.astype(np.float32),
        asp_norm.astype(np.float32)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_terrain(dem_path, image_path, output_dir=None):
    """
    Full terrain preprocessing pipeline for one scene.

    Steps:
      1. Align DEM to satellite image grid
      2. Compute slope from aligned elevation
      3. Compute aspect from aligned elevation
      4. Normalise all three channels
      5. Optionally save to disk

    Args:
        dem_path   : path to SRTM or Copernicus DEM tile
        image_path : path to Levir_CS satellite image
        output_dir : if given, saves altitude.tif, slope.tif, aspect.tif here

    Returns:
        terrain : stacked numpy array of shape (3, H, W)
                  channel 0 = normalised altitude
                  channel 1 = normalised slope
                  channel 2 = normalised aspect
    """
    print("Aligning DEM to image...")
    elevation, meta = align_dem_to_image(dem_path, image_path)

    print("Computing slope...")
    slope  = compute_slope(elevation)

    print("Computing aspect...")
    aspect = compute_aspect(elevation)

    alt_norm, slp_norm, asp_norm = normalize_terrain(elevation, slope, aspect)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_channel(alt_norm, meta, os.path.join(output_dir, "altitude.tif"))
        _save_channel(slp_norm, meta, os.path.join(output_dir, "slope.tif"))
        _save_channel(asp_norm, meta, os.path.join(output_dir, "aspect.tif"))
        print(f"Saved terrain channels to {output_dir}")

    terrain = np.stack([alt_norm, slp_norm, asp_norm], axis=0)
    return terrain


def _save_channel(data, meta, path):
    """Save a single terrain channel as GeoTIFF."""
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "float32"})
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(data, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SANITY CHECK — runs without any real data
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_dem(path, size=256):
    """
    Create a synthetic mountain DEM GeoTIFF for testing.
    No real data needed — generates a Gaussian mountain peak.
    """
    y, x  = np.mgrid[0:size, 0:size]   #Gives x and y value to 
    cx, cy = size // 2, size // 2

    # Gaussian mountain: peak at centre, base elevation 500m, peak ~4000m
    elevation = 4000 * np.exp(
        -((x - cx)**2 + (y - cy)**2) / (2 * (size // 4)**2)
    ) + 500
    elevation = elevation.astype(np.float32)

    """transform = from_bounds(
        left=100.0, bottom=30.0,
        right=100.3, top=30.3,
        width=size, height=size
    )"""
    
    transform = from_bounds(100.0, 30.0, 100.3, 30.3, size, size)

    meta = {
        "driver": "GTiff",
        "dtype":  "float32",
        "width":  size,
        "height": size,
        "count":  1,
        "crs":    "EPSG:4326",
        "transform": transform,
        "nodata": -32768.0
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(elevation, 1)
    return path


if __name__ == "__main__":
    import tempfile #This 2 packages are imported here raathar than at the top cause they are inly required for testing
    import matplotlib.pyplot as plt

    print("=" * 55)
    print("terrain.py — sanity check with synthetic DEM")
    print("=" * 55)

    # Create temporary dummy DEM
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        dummy_path = f.name
    _make_dummy_dem(dummy_path, size=256)
    print(f"\nDummy DEM created at: {dummy_path}")

    # Load
    elevation, meta = load_dem(dummy_path)
    print(f"\nElevation shape : {elevation.shape}")
    print(f"Elevation range : {elevation.min():.1f}m to {elevation.max():.1f}m")

    # Compute
    slope  = compute_slope(elevation)
    aspect = compute_aspect(elevation)
    print(f"\nSlope range  : {slope.min():.1f} to {slope.max():.1f} degrees")
    print(f"Aspect range : {aspect.min():.1f} to {aspect.max():.1f} degrees")

    # Normalise
    alt_n, slp_n, asp_n = normalize_terrain(elevation, slope, aspect)
    print(f"\nNorm altitude : {alt_n.min():.3f} to {alt_n.max():.3f}")
    print(f"Norm slope    : {slp_n.min():.3f} to {slp_n.max():.3f}")
    print(f"Norm aspect   : {asp_n.min():.3f} to {asp_n.max():.3f}")

    # Stack
    terrain = np.stack([alt_n, slp_n, asp_n], axis=0)
    print(f"\nStacked terrain shape : {terrain.shape}  (expect (3, 256, 256))")

    # Visualise
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    items = [
        (alt_n, "Altitude (normalised)", "terrain"),
        (slp_n, "Slope (normalised)",    "YlOrRd"),
        (asp_n, "Aspect (normalised)",   "hsv"),
    ]
    for ax, (data, title, cmap) in zip(axes, items):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("terrain.py — synthetic mountain DEM test", fontsize=12)
    plt.tight_layout()
    plt.savefig("terrain_test.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("\nVisualization saved to terrain_test.png")

    # Cleanup
    os.unlink(dummy_path)
    print("All terrain.py tests passed.")