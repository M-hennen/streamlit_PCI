import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

st.title("Eligibility vs Suitability")

left_layer = "data/classification/Kilifi_0_eligible_areas_cog.tif"
right_layer = "data/suitability/Kilifi_0_suitability_index_cog.tif"

with st.expander("See source code"):
    with st.echo():
        m = leafmap.Map(basemaps="Esri.WorldImagery")  # Satellite basemap

        # Esri World Imagery tile service
        esri_satellite = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

        m.add_tile_layer(
            url=esri_satellite,
            name="Esri Satellite",
            attribution="Tiles © Esri"
        )

        # Add rasters as base layers (mutually exclusive toggle)
        m.add_raster(
            left_layer,
            bands=[1],
            vmin=0,
            vmax=1,
            colormap="viridis",
            layer_name="Eligibility",
            opacity=0.9,
            # overlay=True,   # makes it a base layer (radio toggle)
        )

        m.add_raster(
            right_layer,
            bands=[1],
            vmin=0,
            vmax=1,
            colormap="plasma",
            layer_name="Suitability",
            opacity=0.9,
            # overlay=True,   # base layer toggle
        )

        # Add a layer control with radio-style base layers
        m.add_layer_control()

        # Zoom to raster extent
        m.zoom_to_bounds(left_layer)

m.to_streamlit(height=700)