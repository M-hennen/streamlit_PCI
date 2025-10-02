import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import glob
import feedparser
from weather_functions import weather

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

# === Load list of classified images ===
url = "https://www.metoffice.gov.uk/public/data/PWSCache/WarningsRSS/Region/UK"
feed = feedparser.parse(url)

print("Feed title:", feed.feed.get("title"))
print("Number of entries:", len(feed.entries))

symbols = {"clear sky": "☀️", "few clouds": "🌤️",
    "scattered clouds": "☁️", "broken clouds": "☁️", 
    "shower rain": "🌧️", "rain": "🌧️", "thunderstorm": "⛈️",
    "snow": "❄️", "mist": "🌫️"}

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


# st.markdown("""---""")

# First level columns
col1, col2 = st.columns(2)

with col1:
    st.title("Abergavenny: Project Check-In Dashboard")

with col2:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwL-gVcud2IlQGCKMBVC4dO1oJM3dq7LZm1Q&s")
    # location, temp, condition = weather(51.787, -3.021)
   
# First level columns
col1, col2 = st.columns(2)
with col1:
    data = weather(51.787, -3.021)
    # Capatilise first letter of description
    st.header(f"{symbols[data['description']]} {data['description'].capitalize()} ")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric(label="🌡️ Temperature",
              value=f"{data['temp']:.1f} °C",
              delta="1.2 °C vs yesterday")
    with sc2:
        st.metric(label="༄ Wind Speed",
              value=f"{data['wind_speed']:.1f} m/s",
              delta="1.2 m/s vs yesterday")
    with sc3:
        st.metric(label="⛆ Rainfull (last 1h)",
              value=f"{data['rain']:.1f} mm",
              delta="1.2 m/s vs yesterday")
    with st.expander("5 day forecast"):
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        with sc1:
            st.metric(label="Day 1",
                    value="22 °C",
                    delta="1.2 °C vs yesterday")
        with sc2:
            st.metric(label="Day 2",
                    value="21 °C",
                    delta="0.5 °C vs yesterday")
        with sc3:
            st.metric(label="Day 3",
                    value="19 °C",
                    delta="-0.5 °C vs yesterday")
        with sc4:
            st.metric(label="Day 4",
                    value="18 °C",
                    delta="-1.0 °C vs yesterday")
        with sc5:   
            st.metric(label="Day 5",
                    value="20 °C",
                    delta="0.5 °C vs yesterday")           

with col2:
    st.header("Weather Alerts")
    if len(feed.entries) == 0:
        st.write("No alerts currently.")
        feed = mock_feed_entries 
        with col2.expander("Show mock alerts"):
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
                # "severity": infer_severity(e.get("title", ""))
            })

st.markdown("""---""")
st.header("Project Condition Summary")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.metric(label="🌳 Trees Planted",
          value="1,250",
          delta="50 vs last month")
with sc2:
    st.metric(label="🪵 Biomass Growth",
          value="3.5 t/ha",
          delta="0.2 t/ha vs last month")
with sc3:
    st.metric(label="💧 Soil Moisture",
          value="22.5 %",
          delta="1.5 % vs last month")
st.markdown("""---""")

