import os
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from github import Github
import tempfile
import leafmap.foliumap as leafmap
import streamlit as st
import requests

os.environ["TITILER_ENDPOINT"] = "https://giswqs-titiler-endpoint.hf.space"

# ---------------------------
# 1. URLs for before/after COGs
# ---------------------------
before_url = "https://raw.githubusercontent.com/M-hennen/streamlit_PCI/main/data2/seasons/Abergavenny_NDVI_2021_Winter_harm_cog.tif"
after_url  = "https://raw.githubusercontent.com/M-hennen/streamlit_PCI/main/data2/seasons/Abergavenny_NDVI_2022_Autumn_harm_cog.tif"

# ---------------------------
# 2. Load COGs into arrays
# ---------------------------
with rasterio.open(before_url) as src_before:
    before = src_before.read(1)
    profile = src_before.profile
    nodata_before = src_before.nodata

with rasterio.open(after_url) as src_after:
    after = src_after.read(1)
    nodata_after = src_after.nodata

# ---------------------------
# 3. Compute delta NDVI with masking
# ---------------------------
delta = after - before

# Build a valid mask based on input nodata
valid_mask = np.ones_like(delta, dtype=bool)
if nodata_before is not None:
    valid_mask &= before != nodata_before
if nodata_after is not None:
    valid_mask &= after != nodata_after

# Set invalid pixels to nodata
nodata_val = -9999
delta[~valid_mask] = nodata_val

st.write(f"ΔNDVI mean (valid pixels only): {delta[valid_mask].mean():.4f}")

# ---------------------------
# 4. Save delta as temporary GeoTIFF
# ---------------------------
with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
    delta_path = tmp.name

profile_out = profile.copy()
profile_out.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=nodata_val)

with rasterio.open(delta_path, 'w', **profile_out) as dst:
    dst.write(delta.astype(rasterio.float32), 1)

# ---------------------------
# 5. Convert to proper COG
# ---------------------------
cog_tmp_path = delta_path.replace(".tif", "_cog.tif")
dst_kwargs = cog_profiles.get("deflate")

cog_translate(
    source=delta_path,
    dst_path=cog_tmp_path,
    dst_kwargs=dst_kwargs,
    add_mask=True,        # preserve masked pixels
    in_memory=False,
    web_optimized=True
)

# ---------------------------
# 6. Push to GitHub
# ---------------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = "M-hennen/streamlit_PCI"  # replace with your repo
BRANCH = "main"
GITHUB_FILE_PATH = "data2/seasons/delta_ndvi_cog.tif"

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

with open(cog_tmp_path, "rb") as f:
    content = f.read()

try:
    contents = repo.get_contents(GITHUB_FILE_PATH, ref=BRANCH)
    repo.update_file(contents.path, "Update ΔNDVI COG", content, contents.sha, branch=BRANCH)
except:
    repo.create_file(GITHUB_FILE_PATH, "Add ΔNDVI COG", content, branch=BRANCH)

cog_url = f"https://raw.githubusercontent.com/{REPO_NAME}/{BRANCH}/{GITHUB_FILE_PATH}"
st.write(f"COG URL: {cog_url}")

# ---------------------------
# 7. Visualize in Leafmap
# ---------------------------
region = "data2/TR0001_01_TR0001_01_boundary.geojson"
m = leafmap.Map(center=[51.787, -3.023], basemap="Esri.WorldImagery")
m.add_cog_layer(before_url, bands=[1], name="NDVI Before", colormap_name="rdylgn")
m.add_cog_layer(after_url, bands=[1], name="NDVI After", colormap_name="rdylgn")
m.add_cog_layer(cog_url, bands=[1], name="ΔNDVI Change", colormap_name="rdylgn", opacity=0.7)
m.add_geojson(region, layer_name="AOI")
m.add_layer_control()
m.zoom_to_bounds("NDVI After")
m.to_streamlit(height=500)