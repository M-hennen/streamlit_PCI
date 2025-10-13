import rasterio
import numpy as np
import tempfile
import leafmap.foliumap as leafmap
import streamlit as st
import requests
import os
from localtileserver import get_leaflet_tile_layer
# from localtileserver.tiler.palettes import ColorMap




os.environ["TITILER_ENDPOINT"] = "https://giswqs-titiler-endpoint.hf.space"
# URLs
before_url = "https://raw.githubusercontent.com/M-hennen/streamlit_PCI/main/data2/seasons/Abergavenny_NDVI_2021_Winter_harm_cog.tif"
after_url  = "https://raw.githubusercontent.com/M-hennen/streamlit_PCI/main/data2/seasons/Abergavenny_NDVI_2022_Autumn_harm_cog.tif"

# Check if files exist
for url in [before_url, after_url]:
    r = requests.head(url)
    if r.status_code != 200:
        st.error(f"Raster not found: {url}")

# Load COGs into arrays
with rasterio.open(before_url) as src_before:
    before = src_before.read(1)
    profile = src_before.profile


with rasterio.open(after_url) as src_after:
    after = src_after.read(1)
    profile = src_after.profile
# Compute delta
delta = after - before
delta[np.isnan(delta)] = 0  # Optional: remove NaNs

print("delta mean", delta.mean())

# Save delta to temporary GeoTIFF
tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
profile_out = profile.copy()
profile_out.update(dtype=rasterio.float32, count=1, compress='lzw')
with rasterio.open(tmpfile.name, 'w', **profile_out) as dst:
    dst.write(delta.astype(rasterio.float32), 1)

delta_layer = get_leaflet_tile_layer(tmpfile.name, name="ΔNDVI Change")

# r = requests.head(tmpfile.name)
# if r.status_code != 200:
#     st.error(f"Raster not found: {tmpfile.name}")

# Display in Leafmap
m = leafmap.Map(center=[51.787, -3.023], basemap="Esri.WorldImagery")
m.add_cog_layer(before_url, bands=[1], name="NDVI Before", colormap_name="rdylgn")
m.add_cog_layer(after_url, bands=[1], name="NDVI After", colormap_name="rdylgn")

delta_min, delta_max = np.nanmin(delta), np.nanmax(delta)
m.add_local_tile(
    tmpfile.name,
    layer_name="ΔNDVI Change",
    colormap_name="RdYlGn",
    vmin=delta_min,   # <- ensure proper scaling
    vmax=delta_max,
    opacity=1
)
# m.add_cog_layer(tmpfile.name, bands=[0], name="ΔNDVI Change", colormap_name="RdYlGn", opacity=0.7)
m.add_layer_control()

m.to_streamlit(height=500)