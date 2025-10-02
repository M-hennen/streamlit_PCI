import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import feedparser
import datetime, calendar
from weather_functions import weather, forecast, get_season
from plotting import plot_spei_gauge, classify_spei, classify_rainfall, plot_rainfall_gauge, classify_ndvi, plot_ndvi_gauge

st.set_page_config(layout="wide")

# Load external CSS file
def load_css(file_path):
    with open(file_path) as f:
        return f"<style>{f.read()}</style>"
    
def handle_click_wo_button(state):
    if st.session_state.selection:
        st.session_state[state] = st.session_state.selection

# Inject the CSS into the app
css = load_css("sidebar_styles.css")
st.markdown(css, unsafe_allow_html=True)

# Customize the sidebar
markdown = """
Planting 1500 Giant Sequoias
<https://www.thegreatreserve.org/abergavenny>
"""
logo = "https://www.treeconomy.co/images/treeconomy-logo-white.svg"
st.sidebar.image(logo)
gr_logo = "https://images.squarespace-cdn.com/content/v1/62c40b98c82e671febe1629c/9ef7cd12-08c1-4172-94be-e00403447166/tgr-logo-white-rgb-01-01.png?format=1500w"
st.sidebar.image(gr_logo)
st.sidebar.write(markdown)

# === Load list of classified images ===
url = "https://www.metoffice.gov.uk/public/data/PWSCache/WarningsRSS/Region/UK"
feed = feedparser.parse(url)
# Load Climate Data
clim_data = pd.read_csv("data2/climate_data/Abergavenny_climate_data.csv", parse_dates=["date"])
# Load NDVI data

ndvi_data = pd.read_csv("data2/NDVI/ndvi_time_series.csv")
print(ndvi_data.head())
# Todays date
today = pd.Timestamp.now().normalize()
todays_season = get_season()
print("Today's date:", today)
print("Feed title:", feed.feed.get("title"))
print("Number of entries:", len(feed.entries))

symbols = {"Clear": "☀️","Clouds": "☁️","Rain": "🌧️","Drizzle": "🌦️","Thunderstorm": "⛈️",
           "Snow": "❄️","Mist": "🌫️","Fog": "🌫️","Haze": "🌫️","Smoke": "🌫️","Dust": "🌫️",
           "Ash": "🌫️","Squall": "🌫️","Tornado": "🌪️"
           }

severity_colors = {
    "Severe": "#ff4c4c",   # Red
    "Moderate": "#ffb84c", # Orange
    "Mild": "#ffd966"      # Yellow
}
# Mock feed entries
mock_feed_entries = [
    {
        "title": "Severe Thunderstorm Warning",
        "published": "2025-09-24 12:00:00",
        "summary": "A severe thunderstorm is expected to affect southern England this afternoon.",
        "severity": "Severe"
    },
    {
        "title": "Flood Warning",
        "published": "2025-09-24 09:00:00",
        "summary": "Heavy rain has caused rivers in central England to rise; flooding is likely.",
        "severity": "Moderate"
    }
]
col1, col2 = st.columns([1,2])
with col1:
    st.title("Abergavenny Grove 🌲")
with col2:
    # st.header("Today's Weather")
    data = weather(51.787, -3.021)
    fc_data = forecast(51.87, -3.021)
    days = fc_data.day.unique()

    # Current weather conditions
    st.header(f"{symbols[data['main']]} {data['description'].capitalize()} ")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        monthly_mean = clim_data[clim_data['month'] == today.month].tempC.mean()
        st.metric(label="🌡️ Temperature",
                value=f"{data['temp_max']:.1f} °C",
                delta="{:+.1f} °C vs monthly avg".format(data['temp_max'] - monthly_mean))
    with sc2:
        monthly_mean = clim_data[clim_data['month'] == today.month].wind_speed.mean()
        st.metric(label="༄ Wind Speed",
                value=f"{data['wind_speed']:.1f} m/s",
                delta="{:+.1f} m/s vs monthly avg".format(data['wind_speed'] - monthly_mean))
    with sc3:
        monthly_mean = clim_data[clim_data['month'] == today.month].precip_mm.mean()/30
        st.metric(label="⛆ Rainfull (last 1h)",
                value=f"{data['rain']:.1f} mm",
                delta=f"{data['rain'] - monthly_mean:+.1f} mm vs daily avg")
    with st.expander("5 day forecast"):
        # Initialize session state if it doesn't exist yet
        if "status" not in st.session_state:
            st.session_state.status = "Condition"

        # Define the callback function
        def handle_click_wo_button(state):
            if st.session_state.selection:
                st.session_state[state] = st.session_state.selection

        # Radio button for user to select a new status
        choice = st.radio(
            "Select forecast:",
            ["Condition", "Temperature", "Wind", "Rainfall"],
            on_change=handle_click_wo_button,  # Pass the function as a callback
            args=("status",),  # Pass the "status" argument to the callback
            key="selection",
            index=["Condition", "Temperature", "Wind", "Rainfall"].index(st.session_state.status),
            horizontal=True
        )
        
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        columns = [sc1,sc2,sc3,sc4,sc5]
        for i, day in enumerate(days[1::]):
            with columns[i]:
                st.header(f"{day}/{fc_data[fc_data.day == day].month.min()}")
                if st.session_state.status == "Temperature":
                    st.metric(label="Max", value=f"{fc_data[fc_data.day == day].temp.max():.1f} °C")
                    st.metric(label="Min", value=f"{fc_data[fc_data.day == day].temp.min():.1f} °C")
                elif st.session_state.status == "Wind":
                    st.metric(label="Max", value=f"{fc_data[fc_data.day == day].wind_speed.max():.1f} m/s")
                    st.metric(label="Direction", value=f"{fc_data[fc_data.day == day].wind_compass.mode()[0]}°")
                elif st.session_state.status == "Rainfall":
                    st.metric(label="Probability", value=f"{fc_data[fc_data.day == day].precip_prob.max():.1f} %")
                else:
                    st.metric(label=f"{fc_data[fc_data.day == day].description.mode()[0]}", value=f"{symbols[fc_data[fc_data.day == day].main.mode()[0]]}")
    if len(feed.entries) == 0:
        with st.expander("Weather Alerts"):    
            st.write("No alerts currently.")
            feed = mock_feed_entries 
            with st.expander("Show mock alerts"):
                for entry in feed:  # show top 5
                    print(entry)
                    color = severity_colors.get(entry["severity"], "#ffffff")  # default white
                    with st.container():
                        st.markdown(
                            f"""
                            <div style="border:2px solid {color}; border-radius:10px; padding:15px; margin-bottom:10px; background-color:#f9f9f9">
                                <h4 style="color:{color}; margin:0">{entry['title']}</h4>
                                <small>{entry['published']}</small>
                                <p>{entry['summary']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    else:
        # Convert feedparser entries into a uniform list of dicts
        entries = []
        for e in feed.entries:
            entries.append({
                "title": e.get("title", ""),
                "published": e.get("published", e.get("pubDate", "")),
                "summary": e.get("summary", e.get("description", "")),
                "link": e.get("link", "")   
            })
        with st.expander(f"Show alerts ({len(feed.entries)})"):
            for entry in entries:
                # Pick color based on warning level
                if "Yellow" in entry["title"]:
                    color = "#FFD700"
                elif "Amber" in entry["title"]:
                    color = "#FF8C00"
                elif "Red" in entry["title"]:
                    color = "#FF0000"
                else:
                    color = "#888888"

                st.markdown(
                    f"""
                    <div style="border:2px solid {color}; border-radius:10px; padding:15px; margin-bottom:10px; background-color:#f9f9f9">
                        <h4 style="color:{color}; margin:0">{entry['title']}</h4>
                        <p>{entry['summary']}</p>
                        <p><a href="{entry['link']}" target="_blank">More details</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
# First level columns
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.header("Vegetation Health")
    st.markdown(
        "<p style='margin-top:-10px; font-size:16px;'>Seasonal NDVI</p>",
        unsafe_allow_html=True
    )
    ndvi_current = ndvi_data.groupby(['year','season'])['value'].mean().iloc[-1]
    ndvi_prev = ndvi_data.groupby(['year','season'])['value'].mean().iloc[-2]
    ndvi_seasonal = ndvi_data[ndvi_data['season'] == todays_season].value.mean()

    category, colour = classify_ndvi(ndvi_current)
    fig = plot_ndvi_gauge(ndvi_current, ndvi_prev, ndvi_seasonal)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),  # left, right, top, bottom
        height=200,  # smaller vertical size
        width=300    # smaller horizontal size
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric(label=f"🌳 {todays_season} Average",
        value=f"{ndvi_seasonal:.2f}",
        delta="{:+.2f} vs Seasonal Average".format(ndvi_current - ndvi_seasonal))
    
    with st.expander("Show timeseries"):
        fig, ax = plt.subplots(figsize=(6, 3))
        seasons_df = ndvi_data.groupby(['year','season'])['value'].mean().reset_index()
        ax.plot(seasons_df.apply(lambda row: f"{row['year']}_{row['season']}", axis=1), seasons_df["value"], marker="o", linestyle="-")
        ax.set_xlabel("Season")
        ax.set_ylabel("Mean NDVI")
        ax.set_title("NDVI Seasonal Time Series")
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::3])
        plt.xticks(rotation=45)
        st.pyplot(fig)

    
with col2:
    st.header("Drought")
    st.markdown(
        f"<p style='margin-top:-10px; font-size:16px;'>{calendar.month_name[9]} SPEI-3</p>",
        unsafe_allow_html=True
    )
    spei_value = clim_data.SPEI_3.iloc[-1]
    spei_prev = clim_data.SPEI_3.iloc[-2]

    category, colour = classify_spei(spei_value)

    fig = plot_spei_gauge(spei_value, previous_val=spei_prev )
    # Adjust layout to reduce whitespace
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),  # left, right, top, bottom
        height=200,  # smaller vertical size
        width=300    # smaller horizontal size
    )
    st.plotly_chart(fig, use_container_width=True)

    seasonal_spei = clim_data[clim_data['month'] == today.month].SPEI_3.mean()
    st.metric(
        label=f"{calendar.month_name[9]} average", 
        value= f"{seasonal_spei:.2f}",
        delta= f"{spei_value - seasonal_spei:+.2f} SPEI-3")

    with st.expander("Seasonal Trends"):
        fig, ax = plt.subplots(figsize=(8, 4))
        # Group by month for long-term climatology
        monthly_stats = clim_data.groupby("month")["SPEI_3"].agg(["mean", "std"]).reset_index()
        # Select current year data
        current_year = clim_data["year"].max()
        current_data = clim_data[clim_data["year"] == current_year]
        ax.plot(monthly_stats["month"], monthly_stats["mean"], color="k", linewidth=1,label="Climatology mean")
        ax.fill_between(
            monthly_stats["month"],
            monthly_stats["mean"] - monthly_stats["std"],
            monthly_stats["mean"] + monthly_stats["std"],
            color="lightgray", alpha=0.5, label="±1 Std. Dev."
        )
        ax.plot(current_data["month"], current_data["SPEI_3"], color="darkred", linewidth=2, label=f"{current_year}")
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(-1.5, color='orange', linestyle='--', linewidth=1, label="Drought threshold")
        ax.set_xlabel("Month")
        ax.set_ylabel("SPEI-3")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        ax.set_title("Average drought (SPEI-3)")
        plt.legend()
        st.pyplot(fig)

    if clim_data['SPEI_3'].iloc[-1] <= -1.5:
        count = 1
        while count < len(clim_data) and clim_data['SPEI_3'].iloc[-count] < -1.5:
            count += 1
        print()

    st.markdown(f"Drought has lasted for {count-1} months")

with col3:
    st.header("Rainfall")
    # st.markdown("## Rainfall Summary")  # smaller than st.header
    st.markdown(
        f"<p style='margin-top:-10px; font-size:16px;'>{calendar.month_name[9]} Total (mm)</p>",
        unsafe_allow_html=True
    )
    rainfall_value = clim_data.precip_mm.iloc[-1]
    rainfall_prev = clim_data.precip_mm.iloc[-2]
    category, colour = classify_rainfall(rainfall_value)
    fig = plot_rainfall_gauge(rainfall_value, previous_val=rainfall_prev)
    # Adjust layout to reduce whitespace
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),  # left, right, top, bottom
        height=200,  # smaller vertical size
        width=300    # smaller horizontal size
    )
    st.plotly_chart(fig, use_container_width=True)
    seasonal_rain = clim_data[clim_data['month'] == today.month].precip_mm.mean()

    st.metric(
        label=f"{calendar.month_name[9]} average", 
        value= f"{seasonal_rain:.1f} mm",
        delta= f"{rainfall_value - seasonal_rain:+.2f} mm")

    with st.expander("Seasonal Trends"):
        fig, ax = plt.subplots(figsize=(8, 4))
        # Group by month for long-term climatology
        monthly_stats = clim_data.groupby("month")["precip_mm"].agg(["mean", "std"]).reset_index()
        # Select current year data
        current_year = clim_data["year"].max()
        current_data = clim_data[clim_data["year"] == current_year]
        ax.plot(monthly_stats["month"], monthly_stats["mean"], color="k", linewidth=1,label="Climatology mean")
        ax.fill_between(
            monthly_stats["month"],
            monthly_stats["mean"] - monthly_stats["std"],
            monthly_stats["mean"] + monthly_stats["std"],
            color="lightgray", alpha=0.5, label="±1 Std. Dev."
        )
        ax.plot(current_data["month"], current_data["precip_mm"], color="darkblue", linewidth=2, label=f"{current_year}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Rainfall (mm)")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        ax.set_title("Average rainfall (mm)")
        plt.legend()
        st.pyplot(fig)
