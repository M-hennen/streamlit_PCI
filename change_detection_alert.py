import numpy as np
from scipy.ndimage import label
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np  
from rasterio.features import shapes, rasterize
from shapely.geometry import shape

def connectivity_score(raster, class_value=1, min_patch_size=20, connectivity=8):
    """
    Compute connectivity scores (0–1) for a given binary class in a raster.

    Parameters
    ----------
    raster : 2D np.ndarray
        Binary raster mask (e.g. 1 = target class, 0 = background).
    class_value : int, default=1
        Target class to analyze (1 for connected loss regions).
    min_patch_size : int, default=20
        Minimum patch size for full connectivity (score=1).
    connectivity : {4, 8}, default=8
        Defines pixel neighborhood connectivity.

    Returns
    -------
    scores : 2D np.ndarray of float
        Connectivity scores between 0 and 1.
    labeled : 2D np.ndarray of int
        Patch IDs.
    """
    # Structuring element for 4- or 8-connectivity
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]], dtype=int)

    # Mask of the class of interest
    mask = (raster == class_value)

    # Label connected patches
    labeled, num_features = label(mask, structure=structure)

    # Count pixels in each patch
    patch_sizes = np.bincount(labeled.ravel())

    # Create empty score map
    scores = np.zeros_like(raster, dtype=float)

    # Assign connectivity-based scores
    for pid, size in enumerate(patch_sizes):
        if pid == 0:
            continue  # background
        score = min(1.0, size / min_patch_size)
        scores[labeled == pid] = score

    return scores, labeled

def change_alert (previous, current, aoi, change_thresh=0.2, prev_thresh=0.5):
    """
    Generate change alert map based on connectivity scores.

    Parameters
    ----------
    before : 2D np.ndarray
        Connectivity scores for the 'before' period.
    current : 2D np.ndarray
        Connectivity scores for the 'current' period.
    aoi : 2D np.ndarray
        Area of interest mask (1 = inside AOI, 0 = outside).
    change_thresh : float, default=0.2
        Threshold for significant decrease in connectivity.
    prev_thresh : float, default=0.5
        Minimum previous connectivity score to consider.

    Returns
    -------
    alert_map : 2D np.ndarray of int
        Alert map (1 = alert, 0 = no alert).
    """
    # --- Read AOI ---
    aoi = gpd.read_file(aoi)

    # --- Open rasters ---
    with rasterio.open(current) as src, rasterio.open(previous) as src_prev:
        # --- Reproject AOI to match raster CRS ---
        if aoi.crs != src.crs:
            aoi = aoi.to_crs(src.crs)

        profile = src.profile
        nodata = src.nodata

        # --- Clip both rasters using AOI ---
        clipped_data, clipped_transform = mask(src, aoi.geometry, crop=True)
        clipped_prev, _ = mask(src_prev, aoi.geometry, crop=True)

        data = clipped_data[0]
        prev_data = clipped_prev[0]

        # create mask for entire area (no nodata)
        # Rasterize AOI geometry to match raster extent
        aoi_mask = rasterize(
            [(geom, 1) for geom in aoi.geometry],
            out_shape=data.shape,
            transform=clipped_transform,
            fill=0,
            dtype=np.uint8
        )

    # Compute change data (ΔNDVI)
    delta = prev_data - data
    print("Delta stats:", np.nanmin(delta), np.nanmax(delta), np.nanmean(delta))
    print("Previous stats:", np.nanmin(prev_data), np.nanmax(prev_data), np.nanmean(prev_data))

    print("Δ stats:", np.nanmin(delta), np.nanmax(delta), np.nanmean(delta))
    print("Δ > thresh:", np.sum(delta > change_thresh))
    print("Prev > thresh:", np.sum(prev_data > prev_thresh))
    print("Combined:", np.sum((delta > change_thresh) & (prev_data > prev_thresh)))

    # Compute mask for significant loss (e.g., ΔNDVI < 0.2 and previous NDVI > 0.6)
    loss_mask = (delta > change_thresh) & (prev_data > prev_thresh)
    loss_mask = loss_mask.astype(np.uint8)

    print("loss mask number of pixels:", np.sum(loss_mask))

    # Compute connectivity scores
    scores, labeled = connectivity_score(loss_mask, class_value=1, min_patch_size=20, connectivity=8)

    significant_mask = (scores >= 1.0).astype(np.uint8)
    polygons = []
    for geom, val in shapes(significant_mask, transform=clipped_transform):
        if val == 1:
            polygons.append(shape(geom).centroid)


    gdf_points = gpd.GeoDataFrame(geometry=polygons, crs=src.crs)
    gdf_points = gdf_points.to_crs("EPSG:4326")  # ensure WGS84 for export
    gdf_points = gdf_points.reset_index(drop=True)
    gdf_points["x"] = gdf_points.geometry.x
    gdf_points["y"] = gdf_points.geometry.y

    return gdf_points, delta, profile, aoi_mask, data


if __name__ == "__main__":
    previous = "data2/averages/Abergavenny_NDVI_Mean_Summer_2018-2024_harm_cog.tif"
    current = "data2/averages/Abergavenny_NDVI_Mean_Winter_2018-2024_harm_cog.tif"
    aoi = "data2/TR0001_01_TR0001_01_boundary.geojson"

    alerts, delta, profile, aoi_mask, current = change_alert(previous, current, aoi, change_thresh=0.2, prev_thresh=0.6)
    print("Generated alert points:", len(alerts))
    print("Delta stats:", np.nanmin(delta), np.nanmax(delta), np.nanmean(delta))
