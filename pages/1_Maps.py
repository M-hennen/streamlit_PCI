import streamlit as st
import folium
from folium import GeoJson, Marker
from folium.plugins import MarkerCluster
import leafmap.foliumap as leafmap
import tempfile
import numpy as np
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import storage
import matplotlib as mpl
import rasterio
from plotting import add_styled_colorbar
from weather_functions import get_season
from change_detection_alert import change_alert
# -----------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------
st.set_page_config(layout="wide")

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------
def load_css(file_path: str) -> str:
    """Load an external CSS file and return it as a style block."""
    with open(file_path) as f:
        return f"<style>{f.read()}</style>"

def handle_click_wo_button(state: str):
    """Update session state when a radio selection changes."""
    if st.session_state.selection:
        st.session_state[state] = st.session_state.selection

def load_ndvi(path: str):
    """Load single-band NDVI raster as numpy array + profile.
    Assumes nodata encoded as <= -9990 or np.nan.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    arr[arr <= -9990] = np.nan
    return arr, profile

def save_temp_raster(array: np.ndarray, profile: dict, suffix: str) -> str:
    """Save numpy array as a temporary GeoTIFF and return the file path."""
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    profile_out = profile.copy()
    profile_out.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(tmpfile.name, 'w', **profile_out) as dst:
        dst.write(array.astype(rasterio.float32), 1)
    return tmpfile.name

def custom_colourbar(cmap_name, vmin, vmax, label="Value", label_fontsize=20, tick_fontsize=18):
    """Create and display a custom colorbar in Streamlit using Matplotlib."""
    # Create a Matplotlib figure and an Axes object
    fig, ax = plt.subplots(figsize=(6, 1))  # Small, wide figure for a horizontal colorbar
    fig.subplots_adjust(bottom=0.5)  # Adjust layout to make space for label

    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Create normalization for the value range
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create the colorbar
    cb = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal'
    )

    # Style the label and tick fonts
    cb.set_label(label, fontsize=label_fontsize, labelpad=10)
    cb.ax.tick_params(labelsize=tick_fontsize)

    return fig

# -----------------------------------------------------------
# Sidebar Setup
# -----------------------------------------------------------
css = load_css("sidebar_styles.css")
st.markdown(css, unsafe_allow_html=True)

logo = "https://www.treeconomy.co/images/treeconomy-logo-white.svg"
st.sidebar.image(logo)
gr_logo = "https://images.squarespace-cdn.com/content/v1/62c40b98c82e671febe1629c/9ef7cd12-08c1-4172-94be-e00403447166/tgr-logo-white-rgb-01-01.png?format=1500w"
st.sidebar.image(gr_logo)

# -----------------------------------------------------------
# Rasters Config
# -----------------------------------------------------------
url_path = "https://storage.googleapis.com/pci_rasters/"
ndvi_rasters = {
    i: [f'{year}_{season}', f'{url_path}seasons/Abergavenny_NDVI_{year}_{season}_harm_cog.tif']
    for i, (year, season) in enumerate(
        ( (year, season) for year in range(2017, 2026) for season in ["Spring", "Summer", "Autumn", "Winter"] )
    )
}
ndvi_rasters[34] = ['Latest_NDVI', f'{url_path}latest/Abergavenny_NDVI_2025-10-15_latest_harm_cog.tif']

s1_rasters = {
    i: [f'{year}_{season}', f'{url_path}seasons/Abergavenny_S1_{year}_{season}_harm_cog.tif']
    for i, (year, season) in enumerate(
        ( (year, season) for year in range(2017, 2026) for season in ["Spring", "Summer", "Autumn", "Winter"] )
    )
}
s1_rasters[34] = ['Latest_Radar', f'{url_path}latest/Abergavenny_S1_VV_2025-10-15_latest_harm_cog.tif']
# print(ndvi_rasters)
symb_dict = {
    "NDVI": ["rdylgn", (0.2, 0.8), (-0.4, 0.4)],
    "Radar": ["gray_r", (-18, -4), (-8, 8)],
}
# Load AOI
aoi = f"data2/TR0001_01_TR0001_01_boundary.geojson"


# -----------------------------------------------------------
# UI Layout
# -----------------------------------------------------------
st.title("Forest Health Time Series 🌱")

if "change_detection" not in st.session_state:
    st.session_state.change_detection = False  # default
if "seasonal_mean" not in st.session_state:
    st.session_state.seasonal_mean = False  # default
if "comapare_period" not in st.session_state:
    st.session_state.compare_period = "2025_Summer"  # default

col1, col2, col3 = st.columns([4,1, 1]) 

with col3:
    st.button(
        "Change detection", key="change_detection_btn", 
        on_click=lambda: st.session_state.update(
            change_detection=not st.session_state.change_detection
        )
    )
if st.session_state.change_detection:
    with col2: 
        st.button(
            "Seasonal mean", key="seasonal_mean_btn", 
            on_click=lambda: st.session_state.update(
                seasonal_mean=not st.session_state.seasonal_mean
            )
        )

with col1:
    st.session_state["data_select"] = "NDVI"
    choice = st.radio(
        "Select Data",
        ["NDVI", "Radar"],
        on_change=handle_click_wo_button,
        args=("data_select",),
        key="selection",
        index=["NDVI", "Radar"].index(st.session_state.data_select),
        horizontal=True
    )
    dataset = s1_rasters if choice == "Radar" else ndvi_rasters
    rescale = symb_dict[choice][1]
    rescale_cd = symb_dict[choice][2]

col1, col2 = st.columns([1, 2])

# --- Date Selector ---
with col1:
    # --- Initialize session state ---
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = 34  # default to the latest
    if "compared_date" not in st.session_state:
        st.session_state.compared_date = 33
    if "selected_period" not in st.session_state:
        st.session_state.selected_period = "Latest"  # default to the latest
    if "compared_period" not in st.session_state:
        st.session_state.compared_period = "2025_Summer"  # default to the latest
    
    if st.session_state.change_detection:
        st.write(f"**Target period (after)**:")

    # --- Navigation buttons ---
    scol1, scol2, scol3 = st.columns(3)
    

    with scol1:
        if st.button("Latest", key="latest_image_button"):
            st.session_state.selected_date = 34

    with scol2:
        if st.button("◀ Prev"):
            st.session_state.selected_date = max(0, st.session_state.selected_date - 1)

    with scol3:
        if st.button("Next ▶"):
            st.session_state.selected_date = min(34, st.session_state.selected_date + 1)
    print("Selected season:", dataset[st.session_state.selected_date][0].split("_")[1])
# --- Year/Season dropdowns ---
# col1, _ = st.columns([2, 1])  # layout for dropdowns
# with col1:
    c1, c2 = st.columns(2)
    
    with c1:
        select_year = st.selectbox("Select Year", list(range(2017, 2026)), index=8)
        
    with c2:
        select_season = st.selectbox("Select Season", ["Spring", "Summer", "Autumn", "Winter"], index=1)

    selected_period = f"{select_year}_{select_season}"

    # Update only if the selection changed
    if selected_period != st.session_state.selected_period:
        st.session_state.selected_period = selected_period

        if any(v[0] == selected_period for v in dataset.values()):
            st.session_state.selected_date = next(
                k for k, v in dataset.items() if v[0] == selected_period
            )

    # st.write("Selected date index:", st.session_state.selected_date)
    selected_date = st.session_state.selected_date


    #=== Comparison period selection ===#
    if st.session_state.change_detection:
        # if seasonal not selected
        if st.session_state.seasonal_mean == False:
            st.write(f"**Compared period (before)**")
            c1, c2 = st.columns(2)
            with c1:
                compare_year = st.selectbox("Comparison Year", list(range(2017, 2026)), index=8)
            with c2:
                compare_season = st.selectbox("Comparison Season", ["Spring", "Summer", "Autumn", "Winter"], index=1)        

            compare_period = f"{compare_year}_{compare_season}"

            # Update only if the selection changed
            if compare_period != st.session_state.compared_period:
                st.session_state.compared_period = compare_period

                if any(v[0] == compare_period for v in dataset.values()):
                    st.session_state.compared_date = next(
                        k for k, v in dataset.items() if v[0] == compare_period
                    )

            # st.write("Selected date index:", st.session_state.selected_date)
            compared_date = st.session_state.compared_date 
        else:
            if selected_period == "Latest":
                selected_season = get_season()
                st.write(f"**Compared period (before)**: {selected_season}")
            else:
                selected_season = dataset[st.session_state.selected_date][0].split("_")[1]
            compared_date = 99
            compare_period = selected_season


    

with col1:
    # os.environ["TITILER_ENDPOINT"] = "https://giswqs-titiler-endpoint.hf.space"
    titiler = os.environ["TITILER_ENDPOINT"] = "https://titiler.xyz"
    if st.session_state.change_detection:
        if selected_date == compared_date:
            st.warning("Pick two different dates.")
        else:
            previous_path = dataset[selected_date][1]
            if st.session_state.seasonal_mean:
                if choice == "NDVI":
                    current_path = f'{url_path}averages/Abergavenny_{choice}_Mean_{selected_season}_2018-2024_harm_cog.tif'
                else:
                    current_path = f'{url_path}averages/Abergavenny_S1_Mean_{selected_season}_2018-2024_harm_cog.tif'
            else:
                current_path = dataset[compared_date][1]

            # with rasterio.open(before_path) as src_before:
            #     before = src_before.read(1)
            #     profile = src_before.profile
            #     nodata_before = src_before.nodata

            # with rasterio.open(after_path) as src_after:
            #     after = src_after.read(1)
            #     nodata_after = src_after.nodata

            # # Compute difference
            # delta = before - after

            alerts, delta, profile, aoi_mask, current = change_alert(
                previous_path, 
                current_path, 
                aoi, 
                change_thresh=0.2, 
                prev_thresh=0.6
                )
            
            alert_points_wgs = alerts.to_crs(epsg=4326)
            print ("Number of alert points: ", len(alert_points_wgs))

            # # Build a valid mask based on input nodata
            # valid_mask = np.ones_like(delta, dtype=bool)
            # if nodata_val is not None:
            #     valid_mask &= current != nodata_val

            # Set invalid pixels to nodata
            nodata_val = -9999
            delta[aoi_mask == 0] = nodata_val

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
            # 6. Push to GCP
            # ---------------------------
            # Load GCP credentials from Streamlit secrets
            gcp_info = st.secrets["gcp"]
            creds = storage.Client.from_service_account_info(gcp_info)

            bucket_name = gcp_info["bucket_name"]
            bucket = creds.bucket(bucket_name)

            # Define destination path inside the bucket
            blob_name = f"seasons/delta_{choice}_cog_{selected_date}_vs_{compared_date}.tif"
            blob = bucket.blob(blob_name)

            # Upload the COG file
            blob.upload_from_filename(cog_tmp_path, content_type="image/tiff")

            # Make it public (optional)
            # blob.make_public()

            # Construct the public URL
            cog_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"

            
            print("COG URL:", cog_url)


# -----------------------------------------------------------
# Map Display
# -----------------------------------------------------------
with col2:
    if st.session_state.change_detection:
        # Add tick emoji to change detection title
        st.markdown("**Change Detection** ✅")
        if st.session_state.seasonal_mean:
            st.markdown("**Seasonal Mean** ✅")
        st.markdown(f"Comparing **{dataset[selected_date][0]}** to **{compare_period}**")
        if selected_date == compared_date:
            st.warning("Pick two different dates.")
    else:
        st.markdown(f"Displaying **{dataset[selected_date][0]}**")

    date_id = ndvi_rasters[selected_date][0]
    
    
    gdf = gpd.read_file(aoi)
    total_bounds = gdf.total_bounds
    lon_min, lat_min, lon_max, lat_max = total_bounds
    FIXED_BOUNDS = [
        [lat_min, lon_min],  # Southwest Corner
        [lat_max, lon_max]   # Northeast Corner
    ]
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    ZOOM_LEVEL = 13 

    BBOX_EXTENT = [lon_min, lat_min, lon_max, lat_max] # [minx, miny, maxx, maxy]

    # m = leafmap.Map(center=[51.787, -3.023], basemap="Esri.WorldImagery")
    m = leafmap.Map(
        center=[center_lat, center_lon], 
        zoom_start=ZOOM_LEVEL,
        basemap="Esri.WorldImagery"
    )

    # # Add basemap
    esri_satellite = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    m.add_tile_layer(url=esri_satellite, name="Esri Satellite", attribution="Tiles © Esri")
    import requests
    resp = requests.head(dataset[selected_date][1])
    if resp.status_code == 200:
        if st.session_state.change_detection and selected_date != compared_date:
            # Add to map in Streamlit using leafmap
            m.add_cog_layer(
                current_path, 
                bands=[1], 
                name=f"{compare_period}", 
                rescale=f"{rescale[0]}, {rescale[1]}",
                colormap_name=symb_dict[choice][0]
                )
            m.add_cog_layer(
                previous_path, 
                bands=[1], 
                name=f"{dataset[selected_date][0]}", 
                rescale=f"{rescale[0]}, {rescale[1]}",
                colormap_name=symb_dict[choice][0]
                )
            m.add_cog_layer(
                cog_url, 
                bands=[1], 
                titiler_endpoint=titiler, 
                name="Δ Change", 
                rescale=f"{rescale_cd[0]}, {rescale_cd[1]}",
                colormap_name="coolwarm_r", 
                opacity=1
                )
                
            # --- Add alert points ---
            m.add_points_from_xy(
                alerts,
                x="x",
                y="y",
                spin=True,
            )

        else:
            m.add_cog_layer(
            dataset[selected_date][1],
            bands=[1],
            # titiler_endpoint=titiler,
            name=f"{dataset[selected_date][0]}",
            rescale=f"{rescale[0]}, {rescale[1]}",
            colormap_name=symb_dict[choice][0]
            )

    else: 
        st.error(f"{choice} data not available for {date_id}")
 
    m.fit_bounds(FIXED_BOUNDS)
    m.add_geojson(aoi, layer_name="AOI")
    m.add_layer_control()

    m.to_streamlit(height=500)

with col1:
    if st.session_state.change_detection and selected_date != compared_date:
        st.pyplot(custom_colourbar('coolwarm_r', symb_dict[choice][2][0], symb_dict[choice][2][1], label=f"{choice} Δ Change"))

    if choice == "Radar":
        st.pyplot(
            custom_colourbar(
                'gray', 
                symb_dict[choice][1][0], 
                symb_dict[choice][1][1], 
                label="Radar Backscatter ($\sigma^0$ in dB)")
        )
    else:
        st.pyplot(custom_colourbar('RdYlGn', symb_dict[choice][1][0], symb_dict[choice][1][1], label=f"{choice}"))
    
    if st.session_state.change_detection and selected_date != compared_date:
        st.write(f"✅ Uploaded Δ Delta COG to Google Cloud Storage:")
        st.write(cog_url)


# --- Time Series Expander ---
with col1.expander("📈 Show NDVI Time Series", expanded=False):
    df = pd.read_csv("data2/ndvi_time_series.csv")

    # Compute mean NDVI by year + season
    seasons_df = df.groupby(['year','season'])['value'].mean().reset_index()
    seasons_df['id'] = seasons_df.apply(lambda row: f"{row['year']}_{row['season']}", axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(seasons_df["id"], seasons_df["value"], marker="o", linestyle="-")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean NDVI")
    ax.set_title("NDVI Seasonal Time Series")
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::3])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.markdown(
        """
    This example shows how to use a Streamlit slider to switch between raster layers across time.
    The rasters are displayed as overlays on a satellite basemap.
    """
    )



  

    