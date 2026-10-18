import os
import pandas as pd
import streamlit as st

def _resolve_path(rel_path):
    candidates = [
        rel_path,
        os.path.join("..", rel_path),
        os.path.join(os.path.dirname(__file__), "..", rel_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return rel_path

@st.cache_data(show_spinner=False)
def load_regression_data():
    """Loads and caches UCI Bike Sharing dataset."""
    path = _resolve_path("Data/hour.csv")
    if not os.path.exists(path):
        path = _resolve_path("data/hour.csv")
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=False)
def load_classification_data():
    """Loads and caches INN Hotels Group booking cancellation dataset."""
    path = _resolve_path("Data/INNHotelsGroup.csv")
    if not os.path.exists(path):
        path = _resolve_path("data/INNHotelsGroup.csv")
    df = pd.read_csv(path)
    return df

@st.cache_data(show_spinner=False)
def load_clustering_data():
    """Loads and caches Spotify Tracks audio dataset."""
    path = _resolve_path("Data/spotify_tracks.csv")
    if not os.path.exists(path):
        path = _resolve_path("data/spotify_tracks.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()
