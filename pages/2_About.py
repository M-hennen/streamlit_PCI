import streamlit as st
import leafmap.foliumap as leafmap
import os
import pandas as pd
import matplotlib.pyplot as plt
from todays_weather import weather, forecast

st.set_page_config(layout="wide")

# Load external CSS file
def load_css(file_path):
    with open(file_path) as f:
        return f"<style>{f.read()}</style>"

# Inject the CSS into the app
css = load_css("sidebar_styles.css")
st.markdown(css, unsafe_allow_html=True)
# Customize the sidebar
markdown = """
Project Check-In dashboard 
<https://github.com/opengeos/streamlit-map-template>
"""
logo = "https://www.treeconomy.co/images/treeconomy-logo-white.svg"
st.sidebar.image(logo)
gr_logo = "https://images.squarespace-cdn.com/content/v1/62c40b98c82e671febe1629c/9ef7cd12-08c1-4172-94be-e00403447166/tgr-logo-white-rgb-01-01.png?format=1500w"
st.sidebar.image(gr_logo)

st.title("About 🌱")
