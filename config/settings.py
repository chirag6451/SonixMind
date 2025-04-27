"""
Application settings and configuration values.
"""
import streamlit as st

# App Configuration
APP_TITLE = "Audio Processing Studio"
APP_ICON = "🎵"
PAGE_ICON = "🎵"
LAYOUT = "wide"

# Audio Processing Settings
DEFAULT_SR = 44100
DEFAULT_BIT_DEPTH = 16
DEFAULT_CHANNELS = 2

# YouTube Download Settings
DEFAULT_YOUTUBE_SEGMENT = 60  # seconds
DEFAULT_YOUTUBE_QUALITY = "192k"  # audio quality

# Visualization Settings
WAVEFORM_HEIGHT = 140
WAVEFORM_COLOR = "#1DB954"
SPECTROGRAM_HEIGHT = 192

# UI Theme Settings
DARK_THEME = True
ACCENT_COLOR = "#1DB954"

# Initialize app settings
def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stSlider > div > div {
        background-color: #333333;
    }
    .stSlider > div > div > div {
        background-color: #1DB954 !important;
    }
    .stProgress > div > div {
        background-color: #1DB954;
    }
    button {
        background-color: #1DB954 !important;
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #333333;
        border-radius: 4px 4px 0px 0px;
        border: none;
        color: white;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1DB954 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True) 