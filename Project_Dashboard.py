import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import feedparser
import datetime, calendar

from weather_functions import (
    weather, forecast, get_season, 
    render_feed_alerts, render_mock_alerts
)
from plotting import (
    plot_spei_gauge, classify_spei,
    classify_rainfall, plot_rainfall_gauge,
    classify_ndvi, plot_ndvi_gauge
)

# -----------------------------
# Helper functions & constants
# -----------------------------
def load_css(file_path: str) -> str:
    """Load external CSS file and return <style> block."""
    with open(file_path) as f:
        return f"<style>{f.read()}</style>"

def handle_click_wo_button(state: str):
    """Generic callback: update a session-state key from st.session_state.selection."""
    if st.session_state.get("selection") is not None:
        st.session_state[state] = st.session_state.selection

def render_forecast_expander(fc_data: pd.DataFrame, clim_data: pd.DataFrame, today: pd.Timestamp, symbols: dict):
    """Render the 5-day forecast expander with a radio to switch views."""
    days = fc_data.day.unique()

    with st.expander("5 day forecast"):
        if "status" not in st.session_state:
            st.session_state.status = "Condition"

        # Use the global helper for callback and the same "selection" key as original
        choice = st.radio(
            "Select forecast:",
            ["Condition", "Temperature", "Wind", "Rainfall"],
            on_change=handle_click_wo_button,
            args=("status",),
            key="selection",
            index=["Condition", "Temperature", "Wind", "Rainfall"].index(st.session_state.status),
            horizontal=True
        )

        # Create up to 5 columns (match original UI)
        n = min(5, max(0, len(days)-1))
        cols = st.columns(5)
        # iterate days excluding today (original used days[1::])
        for i, day in enumerate(days[1:1+n]):
            with cols[i]:
                st.header(f"{day}/{fc_data[fc_data.day == day].month.min()}")
                subset = fc_data[fc_data.day == day]
                if st.session_state.status == "Temperature":
                    st.metric(label="Max", value=f"{subset.temp.max():.1f} °C")
                    st.metric(label="Min", value=f"{subset.temp.min():.1f} °C")
                elif st.session_state.status == "Wind":
                    st.metric(label="Max", value=f"{subset.wind_speed.max():.1f} m/s")
                    # guard against empty mode
                    try:
                        dir_mode = subset.wind_compass.mode()[0]
                    except Exception:
                        dir_mode = ""
                    st.metric(label="Direction", value=f"{dir_mode}°")
                elif st.session_state.status == "Rainfall":
                    st.metric(label="Probability", value=f"{subset.precip_prob.max():.1f} %")
                else:
                    # Condition display (description + emoji)
                    try:
                        desc = subset.description.mode()[0]
                        main = subset.main.mode()[0]
                    except Exception:
                        desc = ""
                        main = ""
                    st.metric(label=f"{desc}", value=f"{symbols.get(main, '')}")

def render_seasonal_trends_expander(df: pd.DataFrame, column: str, title: str, current_color: str = "darkred",
                                    y_label: str = None, threshold: float = None, threshold_label: str = None):
    """
    Render a 'Seasonal Trends' expander for a given column in a climate dataset.
    - df must contain columns 'month' and 'year' and the target column.
    """
    if y_label is None:
        y_label = column
    with st.expander("Seasonal Trends"):
        fig, ax = plt.subplots(figsize=(8, 4))
        monthly_stats = df.groupby("month")[column].agg(["mean", "std"]).reset_index()
        current_year = df["year"].max()
        current_data = df[df["year"] == current_year]

        ax.plot(monthly_stats["month"], monthly_stats["mean"], color="k", linewidth=1, label="Climatology mean")
        ax.fill_between(
            monthly_stats["month"],
            monthly_stats["mean"] - monthly_stats["std"],
            monthly_stats["mean"] + monthly_stats["std"],
            color="lightgray", alpha=0.5, label="±1 Std. Dev."
        )
        ax.plot(current_data["month"], current_data[column], color=current_color, linewidth=2, label=f"{current_year}")
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        if threshold is not None:
            ax.axhline(threshold, color='orange', linestyle='--', linewidth=1, label=(threshold_label or "Threshold"))
        ax.set_xlabel("Month")
        ax.set_ylabel(y_label)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_title(title)
        plt.legend()
        st.pyplot(fig)

def compute_drought_duration(df: pd.DataFrame, col: str = "SPEI_3", threshold: float = -1.5) -> int:
    """
    Compute the count of consecutive months up to and including latest where df[col] < threshold.
    Returns number of months (0 if not currently below threshold).
    """
    vals = df[col].dropna().values
    if len(vals) == 0:
        return 0
    count = 0
    # iterate backwards
    for v in vals[::-1]:
        if v < threshold:
            count += 1
        else:
            break
    return count

# CONSTANTS (kept from original)
SYMBOLS = {
    "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️", "Thunderstorm": "⛈️",
    "Snow": "❄️", "Mist": "🌫️", "Fog": "🌫️", "Haze": "🌫️", "Smoke": "🌫️", "Dust": "🌫️",
    "Ash": "🌫️", "Squall": "🌫️", "Tornado": "🌪️"
}

SEVERITY_COLORS = {
    "Red": "#ff4c4c",
    "Amber": "#ffb84c",
    "Yellow": "#ffd966"
}

MOCK_FEED_ENTRIES = [
    {
        "title": "Severe Thunderstorm Warning",
        "published": "2025-09-24 12:00:00",
        "summary": "A severe thunderstorm is expected to affect southern England this afternoon.",
        "severity": "Red"
    },
    {
        "title": "Flood Warning",
        "published": "2025-09-24 09:00:00",
        "summary": "Heavy rain has caused rivers in central England to rise; flooding is likely.",
        "severity": "Amber"
    }
]

# -----------------------------
# Page config + styling
# -----------------------------
st.set_page_config(layout="wide")
css = load_css("sidebar_styles.css")
st.markdown(css, unsafe_allow_html=True)

# Sidebar branding (kept)
markdown = """
Planting 1500 Giant Sequoias
<https://www.thegreatreserve.org/abergavenny>
"""
st.sidebar.image("https://www.treeconomy.co/images/treeconomy-logo-white.svg")
st.sidebar.image("https://images.squarespace-cdn.com/content/v1/62c40b98c82e671febe1629c/9ef7cd12-08c1-4172-94be-e00403447166/tgr-logo-white-rgb-01-01.png?format=1500w")
st.sidebar.write(markdown)

# -----------------------------
# Load external data
# -----------------------------
# Warnings feed
url = "https://www.metoffice.gov.uk/public/data/PWSCache/WarningsRSS/Region/wl"
feed = feedparser.parse(url)

# Climate / NDVI CSVs
clim_data = pd.read_csv("data2/Abergavenny_climate_data.csv", parse_dates=["date"])
ndvi_data = pd.read_csv("data2/ndvi_time_series.csv")

# Today's date / season
today = pd.Timestamp.now().normalize()
todays_season = get_season()

# Debug prints (kept)
print(ndvi_data.head())
print("Today's date:", today)
print("Feed title:", feed.feed.get("title"))
print("Number of entries:", len(feed.entries))

# -----------------------------
# Top header: title + current weather + forecast + alerts
# -----------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.title("Abergavenny Grove 🌲")

with col2:
    # Get current weather + forecast (kept coords from original)
    data = weather(51.787, -3.021)
    fc_data = forecast(51.87, -3.021)

    # Current weather header (emoji + description)
    st.header(f"{SYMBOLS.get(data['main'], '')} {data['description'].capitalize()} ")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        monthly_mean_temp = clim_data[clim_data['month'] == today.month].tempC.mean()
        st.metric(
            label="🌡️ Temperature",
            value=f"{data['temp_max']:.1f} °C",
            delta="{:+.1f} °C vs monthly avg".format(data['temp_max'] - monthly_mean_temp)
        )
    with sc2:
        monthly_mean_wind = clim_data[clim_data['month'] == today.month].wind_speed.mean()
        st.metric(
            label="༄ Wind Speed",
            value=f"{data['wind_speed']:.1f} m/s",
            delta="{:+.1f} m/s vs monthly avg".format(data['wind_speed'] - monthly_mean_wind)
        )
    with sc3:
        # daily average approx used in original (monthly / 30)
        monthly_mean_precip = clim_data[clim_data['month'] == today.month].precip_mm.mean() / 30
        st.metric(
            label="⛆ Rainfall (last 1h)",
            value=f"{data['rain']:.1f} mm",
            delta=f"{data['rain'] - monthly_mean_precip:+.1f} mm vs daily avg"
        )

    # Render forecast expander
    render_forecast_expander(fc_data, clim_data, today, SYMBOLS)

    # Render alerts (feed or mock)
    if len(feed.entries) == 0:
        with st.expander(f"Mock Weather Alerts ({len(MOCK_FEED_ENTRIES)})"):
            render_mock_alerts(MOCK_FEED_ENTRIES, SEVERITY_COLORS)
    else:
        with st.expander(f"Weather Alerts ({len(feed.entries)})"):
            render_feed_alerts(feed, SEVERITY_COLORS)

# -----------------------------
# Main panels: Vegetation, Drought, Rainfall
# -----------------------------
col1, col2, col3 = st.columns([1,1,1])

# Vegetation / NDVI panel
with col1:
    st.header("Vegetation Health")
    st.markdown("<p style='margin-top:-10px; font-size:16px;'>Seasonal NDVI</p>", unsafe_allow_html=True)

    # aggregated seasonal NDVI (kept original logic)
    seasons_grouped = ndvi_data.groupby(['year','season'])['value'].mean().reset_index()
    ndvi_current = seasons_grouped['value'].iloc[-1]
    ndvi_prev = seasons_grouped['value'].iloc[-2]
    ndvi_seasonal = ndvi_data[ndvi_data['season'] == todays_season].value.mean()

    category, colour = classify_ndvi(ndvi_current)
    fig = plot_ndvi_gauge(ndvi_current, ndvi_prev, ndvi_seasonal)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=200, width=300)
    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        label=f"🌳 {todays_season} Average",
        value=f"{ndvi_seasonal:.2f}",
        delta="{:+.2f} vs Seasonal Average".format(ndvi_current - ndvi_seasonal)
    )

    # Timeseries expander
    with st.expander("Show timeseries"):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(seasons_grouped.apply(lambda row: f"{row['year']}_{row['season']}", axis=1), seasons_grouped["value"], marker="o", linestyle="-")
        ax.set_xlabel("Season")
        ax.set_ylabel("Mean NDVI")
        ax.set_title("NDVI Seasonal Time Series")
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::3])
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Drought panel
with col2:
    st.header("Drought")
    current_month_name = calendar.month_name[today.month]
    st.markdown(f"<p style='margin-top:-10px; font-size:16px;'>{current_month_name} SPEI-3</p>", unsafe_allow_html=True)

    spei_value = clim_data.SPEI_3.iloc[-1]
    spei_prev = clim_data.SPEI_3.iloc[-2]
    category, colour = classify_spei(spei_value)

    fig = plot_spei_gauge(spei_value, previous_val=spei_prev)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=200, width=300)
    st.plotly_chart(fig, use_container_width=True)

    seasonal_spei = clim_data[clim_data['month'] == today.month].SPEI_3.mean()
    st.metric(
        label=f"{current_month_name} average",
        value=f"{seasonal_spei:.2f}",
        delta=f"{spei_value - seasonal_spei:+.2f} SPEI-3"
    )

    # Reusable seasonal trends expander for SPEI_3
    render_seasonal_trends_expander(
        clim_data, column="SPEI_3",
        title="Average drought (SPEI-3)",
        current_color="darkred",
        y_label="SPEI-3",
        threshold=-1.5,
        threshold_label="Drought threshold"
    )

    # Compute drought duration in months with values below -1.5
    drought_months = compute_drought_duration(clim_data, col="SPEI_3", threshold=-1.5)
    st.markdown(f"Drought has lasted for {drought_months} months" if drought_months > 0 else "No sustained drought (SPEI-3 >= -1.5)")

# Rainfall panel
with col3:
    st.header("Rainfall")
    st.markdown(f"<p style='margin-top:-10px; font-size:16px;'>{current_month_name} Total (mm)</p>", unsafe_allow_html=True)

    rainfall_value = clim_data.precip_mm.iloc[-1]
    rainfall_prev = clim_data.precip_mm.iloc[-2]
    category, colour = classify_rainfall(rainfall_value)
    fig = plot_rainfall_gauge(rainfall_value, previous_val=rainfall_prev)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=200, width=300)
    st.plotly_chart(fig, use_container_width=True)

    seasonal_rain = clim_data[clim_data['month'] == today.month].precip_mm.mean()
    st.metric(
        label=f"{current_month_name} average",
        value=f"{seasonal_rain:.1f} mm",
        delta=f"{rainfall_value - seasonal_rain:+.2f} mm"
    )

    # Reuse the seasonal trends expander for rainfall
    render_seasonal_trends_expander(
        clim_data, column="precip_mm",
        title="Average rainfall (mm)",
        current_color="darkblue",
        y_label="Monthly Rainfall (mm)"
    )