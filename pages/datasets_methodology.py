import streamlit as st
import pandas as pd
from data_loader import load_regression_data, load_classification_data, load_clustering_data

def render_methodology_page():
    st.title("📖 Datasets, Preprocessing & Academic Methodology")
    st.markdown("""
    Detailed documentation of dataset origins, exploratory data analysis (EDA), feature engineering rationale, data leakage prevention protocols, and evaluation standards.
    """)
    
    t_tab1, t_tab2, t_tab3, t_tab4 = st.tabs([
        "🚴 1. Bike Sharing (Regression)",
        "🏨 2. INN Hotels (Classification)",
        "🎵 3. Spotify Tracks (Clustering)",
        "🛡️ 4. Data Leakage & Protocols"
    ])
    
    # Tab 1: Regression Dataset
    with t_tab1:
        st.subheader("UCI Bike Sharing Dataset — Hourly Track")
        st.markdown("""
        - **Source:** Capital Bikeshare system, Washington D.C. (UCI Machine Learning Repository)
        - **Total Records:** 17,379 hours across 2011–2012
        - **Target Variable:** `cnt` (Total hourly count of rental bikes, `cnt = casual + registered`)
        - **Prediction Nature:** Continuous numerical regression
        """)
        
        with st.expander("🔍 View Raw Dataset Sample (First 5 Rows)", expanded=False):
            df_reg = load_regression_data()
            st.dataframe(df_reg.head(), use_container_width=True)
            st.caption(f"Dimensions: {df_reg.shape[0]:,} rows × {df_reg.shape[1]} columns")
            
        st.markdown("""
        #### Preprocessing & Feature Engineering Steps:
        1. **Feature Engineering:** Extracted `day = pd.to_datetime(df['dteday']).dt.day` to incorporate within-month billing and activity cycles.
        2. **Target Leakage Prevention:** Strictly dropped `casual`, `registered`, `instant`, and raw `dteday`. `cnt` is the exact mathematical sum of `casual + registered`; retaining either would cause direct target leakage.
        3. **Outlier Treatment:** Applied IQR capping ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$) on `temp`, `atemp`, `hum`, and `windspeed` using **training set thresholds only**.
        4. **Categorical Encoding:** One-Hot Encoded `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, and `weathersit`.
        5. **Numerical Scaling:** Standard scaled continuous features (`StandardScaler`).
        """)

    # Tab 2: Classification Dataset
    with t_tab2:
        st.subheader("INN Hotels Group Dataset — Booking Cancellation Track")
        st.markdown("""
        - **Total Records:** 36,275 reservation records
        - **Target Variable:** `booking_status` (`0 = Not_Canceled`, `1 = Canceled`)
        - **Class Distribution:** 24,390 Not_Canceled (67.24%) vs. 11,885 Canceled (32.76%)
        - **Task:** Binary classification under moderate class imbalance
        """)
        
        with st.expander("🔍 View Raw Dataset Sample (First 5 Rows)", expanded=False):
            df_class = load_classification_data()
            st.dataframe(df_class.head(), use_container_width=True)
            st.caption(f"Dimensions: {df_class.shape[0]:,} rows × {df_class.shape[1]} columns")
            
        st.markdown(r"""
        #### The 7 Engineered Features (Notebooks/Classification.ipynb):
        - `total_nights`: `no_of_weekend_nights + no_of_week_nights` (measures length of stay).
        - `total_guests`: `no_of_adults + no_of_children` (measures travel party size).
        - `total_previous_bookings`: `no_of_previous_cancellations + no_of_previous_bookings_not_canceled` (historical customer fidelity).
        - `is_family`: Binary indicator $(1 \text{ if } \text{no\_of\_children} > 0 \text{ else } 0)$.
        - `price_per_guest`: `avg_price_per_room / total_guests` (individual cost burden).
        - `arrival_month_sin` & `arrival_month_cos`: Cyclic trigonometric encoding of month $(2\pi \cdot m / 12)$ preserving continuous calendar circularity (December $\leftrightarrow$ January).
        
        #### Preprocessing & Outlier Treatment:
        - Checked for missing and duplicate values (0 missing, 0 duplicate records).
        - Applied IQR capping on `lead_time` (upper bound: 292 days) and `avg_price_per_room` (bounds: €20.75 to €179.55) using **training set quantiles only**.
        - Categoricals (`type_of_meal_plan`, `room_type_reserved`, `market_segment_type`) transformed via `OneHotEncoder(handle_unknown='ignore')`.
        - Continuous numerical attributes standardized via `StandardScaler()`.
        """)

    # Tab 3: Clustering Dataset
    with t_tab3:
        st.subheader("Spotify Audio Tracks Dataset — Unsupervised Track")
        st.markdown("""
        - **Total Records:** 50,000 track records
        - **Target:** **None** (Unsupervised Learning)
        - **Core Feature Representation:** 9 continuous audio physics metrics:
          `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
        - **Ground-Truth Label Isolation:** `genre` was strictly excluded from distance matrices and model fitting; referenced only post-hoc for validation (ARI = 0.0256, NMI = 0.1757).
        """)
        
        with st.expander("🔍 View Raw Dataset Sample (First 5 Rows)", expanded=False):
            df_clust = load_clustering_data()
            st.dataframe(df_clust.head(), use_container_width=True)
            st.caption(f"Dimensions: {df_clust.shape[0]:,} rows × {df_clust.shape[1]} columns")
            
        st.markdown(r"""
        #### Systematic Optimization Experiments:
        - **Baseline (15 Features, StandardScaler):** $R^2$ equivalent Silhouette = `0.2416` (K-Means) and `0.2871` (Complete Linkage).
        - **Experiment A (2D Benchmark):** Loudness + Acousticness (`MinMaxScaler`, Average Linkage, $k=2$) $\\rightarrow$ **`0.6054`** on sample ($0.5759 \pm 0.0256$ multi-seed mean).
        - **Experiment C (Primary Capstone Model):** Core 9 Audio Features (`RobustScaler`, Complete Linkage, $k=2$) $\\rightarrow$ **`0.5026`** on sample ($0.5034 \pm 0.0060$ multi-seed mean). Doubles baseline separation while retaining all 9 perceptual dimensions.
        - **Feasibility on 0.70:** 387 grid combinations confirmed that $0.70$ is unachievable on continuous audio because acoustic properties form dense overlapping multivariate continua without void margins.
        """)

    # Tab 4: Protocols
    with t_tab4:
        st.subheader("🛡️ Academic Integrity & Data Leakage Prevention Protocols")
        st.markdown("""
        To comply with capstone rubric guidelines and academic standards:
        
        1. **Strict Train-Test Isolation:**
           - All train/test splits performed using an 80:20 ratio with fixed random seed `random_state=42`.
           - In Classification, `stratify=y` was applied to preserve class proportions (67.2% vs 32.8%).
           
        2. **Transformer Fitting Discipline:**
           - Scalers (`StandardScaler`, `RobustScaler`, `MinMaxScaler`), imputers (`SimpleImputer`), and encoders (`OneHotEncoder`) were **fitted exclusively on the training set**.
           - Test sets and live GUI user inputs are strictly transformed using the already-fitted training parameters.
           
        3. **Outlier Threshold Derivation:**
           - IQR capping bounds ($Q_1, Q_3, \text{IQR}$) were calculated solely on $X_{\text{train}}$. The test set was clipped using these training boundaries, preventing distribution leakage.
           
        4. **Zero Ground-Truth Guidance in Clustering:**
           - Ground-truth genre labels were quarantined in a separate dataframe before distance calculation, ensuring strictly unsupervised geometric clustering.
           
        5. **Multi-Seed Stability Verification:**
           - Models were evaluated across 5 independent random sample seeds (42, 100, 2024, 7, 999) to verify that single-run metrics reflect genuine distributions rather than isolated sample artifacts.
        """)
