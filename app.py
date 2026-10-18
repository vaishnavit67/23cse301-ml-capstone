import sys
import os

# Add root directory and src to sys.path so modules resolve cleanly in local and cloud deployment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
PAGES_DIR = os.path.join(CURRENT_DIR, "pages")
for p in [CURRENT_DIR, SRC_DIR, PAGES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# Streamlit Page Config
st.set_page_config(
    page_title="23CSE301 | Machine Learning Capstone",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global Styling
st.markdown("""
<style>
    /* Global Typography & Backgrounds */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        letter-spacing: -0.02em;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #0072B2;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600;
        color: #555555;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    /* Status Badge */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Import Page Renderers
from home import render_home_page
from regression import render_regression_page
from classification import render_classification_page
from clustering import render_clustering_page
from model_comparison import render_comparison_page
from datasets_methodology import render_methodology_page
from about import render_about_page

# Navigation State Handling
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Home"

PAGES = {
    "Home": {
        "title": "🏠 Home Dashboard",
        "render": render_home_page
    },
    "Regression": {
        "title": "🚴 Regression (10 Models)",
        "render": render_regression_page
    },
    "Classification": {
        "title": "🏨 Classification (10 Models)",
        "render": render_classification_page
    },
    "Clustering": {
        "title": "🎵 Clustering (2 Models)",
        "render": render_clustering_page
    },
    "Comparison": {
        "title": "📊 Model Comparison",
        "render": render_comparison_page
    },
    "Methodology": {
        "title": "📖 Datasets & Methodology",
        "render": render_methodology_page
    },
    "About": {
        "title": "ℹ️ About & Team",
        "render": render_about_page
    }
}

# Sidebar Content
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 15px 0;">
        <h2 style="margin: 0; color: #0072B2; font-weight: 800;">23CSE301</h2>
        <span style="font-size: 0.85rem; color: #666; font-weight: 600; text-transform: uppercase;">ML Capstone Dashboard</span><br>
        <span style="background-color: #e6f4ea; color: #137333; font-size: 0.75rem; font-weight: bold; padding: 2px 8px; border-radius: 10px; border: 1px solid #ceead6; display: inline-block; margin-top: 5px;">
            ● Review 2 Verified (+2 Bonus)
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    page_keys = list(PAGES.keys())
    page_titles = [PAGES[k]["title"] for k in page_keys]
    
    current_idx = page_keys.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in page_keys else 0
    
    selected_title = st.radio(
        "Navigation Menu:",
        page_titles,
        index=current_idx,
        label_visibility="collapsed"
    )
    
    # Map back selected title to key
    selected_key = page_keys[page_titles.index(selected_title)]
    st.session_state["nav_selection"] = selected_key
    
    st.markdown("---")
    
    # Quick Summary Box in Sidebar
    st.markdown("""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 0.85rem;">
        <b style="color: #333;">Master Model Count: 22</b><br>
        <span style="color: #666;">• Regression: 10 Models</span><br>
        <span style="color: #666;">• Classification: 10 Models</span><br>
        <span style="color: #666;">• Clustering: 2 Models</span><br>
        <hr style="margin: 8px 0;">
        <span style="color: #555;"><b>Team 17:</b> Amrita CSE</span><br>
        <span style="font-size: 0.8rem; color: #888;">Himakesh • Vaishnavi • Manesh</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 20px; text-align: center; font-size: 0.75rem; color: #999;">
        Built for 23CSE301 Machine Learning Capstone Review 2 Presentation.
    </div>
    """, unsafe_allow_html=True)

# Main Page Routing
render_fn = PAGES[st.session_state["nav_selection"]]["render"]
render_fn()
