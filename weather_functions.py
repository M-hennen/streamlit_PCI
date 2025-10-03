import requests
import pandas as pd
import datetime
import streamlit as st
import os
import re

API_KEY = st.secrets["API_KEY"] 
# API_KEY = os.environ.get("OPEN_WEATHER_API")

# Convert wind direction deg into compass direction
def deg_to_compass(deg):
    val = int((deg / 22.5) + 0.5)
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return directions[(val % 16)]

def weather(lat, lon):
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
    params = {
        "lat": lat,  # Latitude
        "lon": lon,  # Longitude
        "appid": API_KEY,  # API key
        "units": "metric"  # Temperature in Celsius
    }
    res = requests.get(BASE_URL, params=params)
    data = res.json()

    if res.status_code == 200:
        conditions = {}
        conditions['location'] = data["name"]
        conditions['temp'] = data["main"]["temp"]
        conditions['temp_max'] = data["main"]["temp_max"]
        conditions['main'] = data["weather"][0]["main"]
        conditions['description'] = data["weather"][0]["description"]
        conditions['wind_speed'] = data["wind"]["speed"]
        conditions['wind_direction'] = data["wind"]["deg"]
        conditions['wind_gust'] = data["wind"]["gust"] if "gust" in data["wind"] else "N/A"
        conditions['rain'] = data['rain']['1h'] if 'rain' in data and '1h' in data['rain'] else 0

        print(f"{conditions['location']}: {conditions['temp']}°C, {conditions['description']}")

        return conditions #location, temp, description, 
    else:
        print("City not found!")


def forecast(lat, lon):
    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,  # Latitude
        "lon": lon,  # Longitude
        "appid": API_KEY,  # API key
        "units": "metric"  # Temperature in Celsius
    }
    res = requests.get(BASE_URL, params=params)
    data = res.json()
    
    forecast_df = {}
    if res.status_code == 200:
        print(f"Weather forecast for coordinates ({lat}, {lon}):\n")
        for item in data["list"]:
            # Extract forecast details
            timestamp = item["dt_txt"]
            temp = item["main"]["temp"]
            temp_min = item["main"]["temp_min"]
            temp_max = item["main"]["temp_max"]
            wind_speed = item["wind"]["speed"]
            wind_direction = item["wind"]["deg"]
            precip_prob = item.get("pop", 0) * 100  # Probability of precipitation
            main = item["weather"][0]["main"]
            description = item["weather"][0]["description"]
            forecast_df[timestamp] = [temp, temp_min, temp_max, wind_speed, wind_direction, precip_prob, main, description]
            # print(f"{timestamp}: {temp}°C, {description}")
        
        df = pd.DataFrame.from_dict(forecast_df, orient='index', columns=['temp', 'temp_min', 'temp_max', 'wind_speed', 'wind_direction', 'precip_prob', 'main', 'description'])
        df.index = pd.to_datetime(df.index)
        
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['hour'] = df.index.hour
        df['wind_compass'] = df['wind_direction'].apply(deg_to_compass)
            
        return df
    else:
        # Print the error message for debugging
        print(f"Error: {data.get('message', 'Unable to fetch forecast!')}")


def get_season(date=None):
    if date is None:
        date = datetime.date.today()

    year = date.year

    # Define seasonal boundaries (approximate astronomical dates)
    spring = (datetime.date(year, 3, 20), datetime.date(year, 6, 20))
    summer = (datetime.date(year, 6, 21), datetime.date(year, 9, 22))
    autumn = (datetime.date(year, 9, 23), datetime.date(year, 12, 20))
    winter1 = (datetime.date(year, 1, 1), datetime.date(year, 3, 19))
    winter2 = (datetime.date(year, 12, 21), datetime.date(year, 12, 31))

    if spring[0] <= date <= spring[1]:
        return "Spring"
    elif summer[0] <= date <= summer[1]:
        return "Summer"
    elif autumn[0] <= date <= autumn[1]:
        return "Autumn"
    elif winter1[0] <= date <= winter1[1] or winter2[0] <= date <= winter2[1]:
        return "Winter"
    else:
        return "Unknown"


def render_mock_alerts(mock_feed_entries, severity_colors):
    """Render mock alerts if no live alerts are available."""
    for entry in mock_feed_entries:
        color = severity_colors.get(entry["severity"], "#ffffff")
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:10px; 
                        padding:15px; margin-bottom:10px; background-color:#f9f9f9">
                <h4 style="color:{color}; margin:0">{entry['title']}</h4>
                <small>{entry['published']}</small>
                <p>{entry['summary']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def _extract_severity(title: str) -> str:
    """Extract severity level from a warning title string."""
    match = re.match(r"(Yellow|Amber|Red)", title)
    return match.group(1) if match else "Unknown"

def _process_feed_alerts(feed):
    entries = [
        {
            "title": e.get("title", ""),
            "published": e.get("published", e.get("pubDate", "")),
            "summary": e.get("summary", e.get("description", "")),
            "link": e.get("link", ""),
            "severity": _extract_severity(e.get("title", ""))
        }
        for e in feed.entries
    ]
    return entries


def render_feed_alerts(feed, severity_colors):
    """Render actual feed alerts from RSS feed."""
    entries = _process_feed_alerts(feed)
    for entry in entries:
        color = severity_colors.get(entry["severity"], "#ffffff")
        st.markdown(
            f"""
            <div style="border:2px solid {color}; border-radius:10px; 
                        padding:15px; margin-bottom:10px; background-color:#f9f9f9">
                <h4 style="color:{color}; margin:0">{entry['title']}</h4>
                <p>{entry['summary']}</p>
                <p><a href="{entry['link']}" target="_blank">More details</a></p>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_alerts(feed, mock_feed_entries, severity_colors):
    """Decide which alerts to render: mock or real feed."""
    if len(feed.entries) == 0:
        render_mock_alerts(mock_feed_entries, severity_colors)
    else:
        render_feed_alerts(feed, severity_colors)

if __name__ == "__main__":
    latitude = 51.787 #51.5074  # Latitude for London51.787, -3.021
    longitude = -3.021 #-0.1278  # Longitude for London
    # latitude = 51.5074  # Latitude for London51.787, -3.021
    # longitude = -0.1278 
    weather(latitude, longitude)
    forecast(latitude, longitude)