import streamlit as st
import leafmap.foliumap as leafmap
import tempfile
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

st.set_page_config(layout="wide")

# Load external CSS file
def load_css(file_path):
    with open(file_path) as f:
        return f"<style>{f.read()}</style>"

# Inject the CSS into the app
css = load_css("sidebar_styles.css")
st.markdown(css, unsafe_allow_html=True)
logo = "https://www.treeconomy.co/images/treeconomy-logo-white.svg"
st.sidebar.image(logo)
gr_logo = "https://images.squarespace-cdn.com/content/v1/62c40b98c82e671febe1629c/9ef7cd12-08c1-4172-94be-e00403447166/tgr-logo-white-rgb-01-01.png?format=1500w"
st.sidebar.image(gr_logo)

# Define the callback function
def handle_click_wo_button(state):
    if st.session_state.selection:
        st.session_state[state] = st.session_state.selection

# --- helper to load single-band NDVI tiff ---
def load_ndvi(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    # assume nodata encoded as <= -9999 or np.nan
    arr[arr <= -9990] = np.nan
    return arr, profile

def save_temp_raster(array, profile, suffix):
    """Save numpy array as a temporary GeoTIFF and return the file path"""
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    profile_out = profile.copy()
    profile_out.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(tmpfile.name, 'w', **profile_out) as dst:
        dst.write(array.astype(rasterio.float32), 1)
    return tmpfile.name

# Define your rasters for each year (local COGs)
# Make sure the order matches the slider (e.g., 2020 → index 0)
ndvi_rasters = {i:[f'{year}_{season}',f'data2/NDVI/seasons/Abergavenny_NDVI_{year}_{season}_harm_cog.tif']
                for i, (year, season) in enumerate(
        ( (year, season) for year in range(2017, 2026) for season in ["Spring", "Summer", "Autumn", "Winter"] )
    )
}
ndvi_rasters[34] = ['Latest_NDVI', 'data2/NDVI/latest/Abergavenny_NDVI_2025-09-30_latest_harm_cog.tif']
s1_rasters = {i:[f'{year}_{season}',f'data2/NDVI/seasons/Abergavenny_S1_{year}_{season}_harm_cog.tif']
                for i, (year, season) in enumerate(
        ( (year, season) for year in range(2017, 2026) for season in ["Spring", "Summer", "Autumn", "Winter"] )
    )
}
s1_rasters[34] = ['Latest_Radar', 'data2/NDVI/latest/Abergavenny_S1_VV_2025-10-02_latest_harm_cog.tif']


symb_dict = {
    "NDVI": ["RdYlGn", (0.2, 0.8)],
    "Radar": ["Greys_r", (-18, -4)],}



st.title("Forest Health Time Series 🌱")
if "change_detection" not in st.session_state:
    st.session_state.change_detection = False  # default
col1, col2 = st.columns([5,1]) 

with col2:
    # Make button to change session state to change detection on/off

    st.button(
        "Change detection", key="change_detection_btn", 
        on_click=lambda: st.session_state.update(
            change_detection=not st.session_state.change_detection
            )
        )

with col1:
    st.session_state["data_select"] = "NDVI"
    choice = st.radio(
        "Select Data",
        ["NDVI", "Radar"],
        on_change=handle_click_wo_button,  # Pass the function as a callback
        args=("data_select",),  # Pass the "status" argument to the callback
        key="selection",
        index=["NDVI", "Radar"].index(st.session_state.data_select),
        horizontal=True
    )

    if choice == "Radar":
        dataset = s1_rasters
    else:
        dataset = ndvi_rasters


col1, col2 = st.columns([1, 2])

with col1:
    


#     selected_date = st.number_input(
#     "Time series (2017–2025)", 
#     min_value=0, 
#     max_value=34, 
#     value=34, 
#     step=1
# )

    # Radio button for user to select a new status
    if "selected_date" not in st.session_state:
        st.session_state.selected_date = 34  # default
    
    scol1,scol2,scol3 = st.columns([0.5,6,0.5])
    with scol1:
        if st.button("◀"):
            st.session_state.selected_date = max(0, st.session_state.selected_date - 1)
    with scol2:
    # Streamlit slider to pick year
        selected_date = st.slider(
            "Time series (2017-2025)", 
            min_value=0, 
            max_value=34, 
            value=st.session_state.selected_date, 
            format="%d"
            )
    with scol3:    
        if st.button("▶"):
            st.session_state.selected_date = min(34, st.session_state.selected_date + 1)
    
with col1.expander("Change Detection Settings", expanded=False):

    scol1, scol2 = st.columns(2)
    with scol1:
        compare_year = st.selectbox("Comparison Year", list(range(2017, 2026)), index=1)

    with scol2:
        compare_season = st.selectbox("Comparison Season", ["Spring", "Summer", "Autumn", "Winter"], index=2)
    compare_period = f"{compare_year}_{compare_season}"
    print(compare_period)
    matches = [k for k, v in dataset.items() if v[0] == compare_period]
    if matches:
        compare_date = matches[0]
    else:
        compare_date = None      


    

    # Read CSV
    df = pd.read_csv("data2/NDVI/ndvi_time_series.csv")

    # Compute mean NDVI by season
    season_means = df.groupby('season')['value'].mean()

    # Compute mean NDVI by year and season
    seasons_df = df.groupby(['year','season'])['value'].mean().reset_index()
    seasons_df['id'] = seasons_df.apply(lambda row: f"{row['year']}_{row['season']}", axis=1)

    # Convert to dictionary
    seasons_dict = seasons_df.set_index('id')['value'].to_dict()  # just the NDVI value

    
with col1.expander("📈 Show NDVI Time Series", expanded=False):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(seasons_df["id"], seasons_df["value"], marker="o", linestyle="-")
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean NDVI")
    ax.set_title("NDVI Seasonal Time Series")
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::3])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    

with col2:
    if st.session_state.change_detection:
        st.markdown(f"Comparing **{dataset[selected_date][0]}** to **{compare_period}**")

        if selected_date == compare_date:
            st.warning("Pick two different dates.")
        else:
            before_path = dataset[selected_date][1]
            after_path  = dataset[compare_date][1]

            ndvi_before, prof = load_ndvi(before_path)
            ndvi_after, _ = load_ndvi(after_path)

            # compute delta and relative
            delta = ndvi_after - ndvi_before
            rel_change = (delta) / (np.abs(ndvi_before) + 1e-6)

            # compute stats from the before NDVI
            mean_before = np.nanmean(ndvi_before)
            std_before = np.nanstd(ndvi_before)

            # thresholds = ±1 std around the mean of BEFORE
            delta_thresh_loss = -std_before
            delta_thresh_gain = std_before

            # compute categories
            cat = np.full(delta.shape, np.nan, dtype=np.float32)  # 0 = no change
            cat[(delta <= delta_thresh_loss)] = 1   # loss
            cat[(delta >= delta_thresh_gain)] = 2   # gain

            delta_path = save_temp_raster(delta, prof, "_delta.tif")
            profile_out = prof.copy()
            profile_out.update(dtype=rasterio.float32, nodata=np.nan)

            # with rasterio.open(cat_path, 'w', **profile_out) as dst:
            #     dst.write(cat, 1)
            cat_path   = save_temp_raster(cat, profile_out, "_cat.tif")

        
    date_id = ndvi_rasters[selected_date][0]

    st.write(f"Displaying rasters for: {date_id}")
       

    # Create the map with satellite basemap
    m = leafmap.Map(basemap="Esri.WorldImagery")

    # Esri World Imagery tile service
    esri_satellite = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

    m.add_tile_layer(
            url=esri_satellite,
            name="Esri Satellite",
            attribution="Tiles © Esri"
        )


    if not os.path.exists(dataset[selected_date][1]):
        st.error(f"Raster file not found: {dataset[selected_date][1]}")
    else:
        # Add overlays for the selected year
        m.add_raster(
            # ndvi_rasters[selected_date][1],
            dataset[selected_date][1],
            bands=[1],
            vmin=symb_dict[choice][1][0], 
            vmax=symb_dict[choice][1][1], 
            colormap=symb_dict[choice][0], 
            layer_name=f"{choice} {date_id}",
            opacity=0.7,
            )
        if st.session_state.change_detection and selected_date != compare_date:
                # Delta NDVI continuous
            m.add_raster(
                delta_path,
                bands=[1],
                vmin=-np.nanmax(np.abs(delta)),
                vmax=np.nanmax(np.abs(delta)),
                colormap="RdYlGn",
                layer_name=f"ΔNDVI Change",
                opacity=1,
            )

    
    region = "data2/NDVI/TR0001_01_TR0001_01_boundary.geojson"

    m.add_geojson(region, layer_name="AOI")
    # Add layer control (checkboxes to toggle overlays)
    m.add_layer_control()

    # Zoom to the first raster
    m.zoom_to_bounds(ndvi_rasters[selected_date][1])

    # Display in Streamlit
    m.to_streamlit(height=500)

    st.markdown(
        """
    This example shows how to use a Streamlit slider to switch between different raster layers representing different years.
    The rasters are displayed as overlays on a satellite basemap.
    """
    )


    
