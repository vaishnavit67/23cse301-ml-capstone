import streamlit as st
import pandas as pd
from model_registry import REGRESSION_ALGORITHMS, load_regression_results
from prediction_service import predict_regression
from visualization import plot_actual_vs_predicted, plot_residuals, plot_regression_comparison_bar

def render_regression_page():
    st.title("🚴 Regression Workspace — Bike Sharing Demand")
    st.markdown("""
    **Problem Statement:** Predict the total hourly count of rental bikes (`cnt`) in the capital bikeshare system based on meteorological variables (temperature, humidity, windspeed) and seasonal/temporal indicators.
    """)
    
    results = load_regression_results()
    
    # Algorithm Selector
    model_keys = list(REGRESSION_ALGORITHMS.keys())
    model_names = [REGRESSION_ALGORITHMS[k]["name"] for k in model_keys]
    
    col_sel, col_rank = st.columns([0.7, 0.3])
    with col_sel:
        selected_name = st.selectbox(
            "Select Regression Algorithm (10 Available):",
            model_names,
            index=0
        )
    
    selected_key = [k for k, v in REGRESSION_ALGORITHMS.items() if v["name"] == selected_name][0]
    meta = REGRESSION_ALGORITHMS[selected_key]
    model_res = results.get(selected_key, {})
    
    with col_rank:
        st.markdown(f"""
        <div style="background-color: #f0f7ff; border-radius: 8px; padding: 10px; border-left: 4px solid #0072B2; margin-top: 25px;">
            <span style="font-size: 0.85rem; color: #555;">Performance Rank</span><br>
            <span style="font-size: 1.3rem; font-weight: bold; color: #0072B2;">#{meta['rank']} of 10</span>
        </div>
        """, unsafe_allow_html=True)
        
    # Overview & Metrics Section
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Test R² Score", f"{model_res.get('r2', 0.0):.4f}")
    with m_col2:
        st.metric("RMSE", f"{model_res.get('rmse', 0.0):.2f}", help="Root Mean Squared Error in bikes/hour")
    with m_col3:
        st.metric("MAE", f"{model_res.get('mae', 0.0):.2f}", help="Mean Absolute Error in bikes/hour")
    with m_col4:
        st.metric("5-Fold CV R²", f"{model_res.get('cv_mean', 0.0):.4f}", f"±{model_res.get('cv_std', 0.0):.4f}")
    with m_col5:
        st.metric("Training Time", f"{model_res.get('train_time', 0.0):.2f}s")
        
    with st.expander("ℹ️ Model Architecture, Hyperparameters & Formula", expanded=True):
        st.markdown(f"**Algorithm Description:** {meta['desc']}")
        st.markdown(f"**Hyperparameters:** `{meta['hyperparameters']}`")
        st.markdown(f"**Mathematical Formulation:**")
        st.markdown(meta["math"])
        st.markdown(f"**Dataset & Target:** `{meta['dataset']}` $\\rightarrow$ `{meta['target']}`")

    st.markdown("---")
    
    # Interactive Prediction Form
    st.subheader("🔮 Interactive Demand Inference Form")
    st.caption("Adjust the sliders and dropdowns below to simulate specific weather conditions and hour of the day, then click **Predict Bike Demand**.")
    
    with st.form(key="reg_predict_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            st.markdown("##### 📅 Temporal Attributes")
            season_map = {"Spring (1)": 1, "Summer (2)": 2, "Fall (3)": 3, "Winter (4)": 4}
            season = st.selectbox("Season", list(season_map.keys()), index=2)
            yr = st.selectbox("Year", ["2011 (Year 0)", "2012 (Year 1)"], index=1)
            mnth = st.slider("Month", min_value=1, max_value=12, value=7)
            day = st.slider("Day of the Month", min_value=1, max_value=31, value=15)
            hr = st.slider("Hour of the Day (0-23)", min_value=0, max_value=23, value=17, help="5:00 PM evening peak commute")
            
        with f_col2:
            st.markdown("##### 🏢 Calendar Attributes")
            holiday = st.selectbox("Holiday", ["No (0)", "Yes (1)"], index=0)
            weekday_map = {
                "Sunday (0)": 0, "Monday (1)": 1, "Tuesday (2)": 2, "Wednesday (3)": 3,
                "Thursday (4)": 4, "Friday (5)": 5, "Saturday (6)": 6
            }
            weekday = st.selectbox("Day of Week", list(weekday_map.keys()), index=2)
            workingday = st.selectbox("Working Day (Neither weekend nor holiday)", ["Yes (1)", "No (0)"], index=0)
            weathersit_map = {
                "Clear / Few Clouds (1)": 1,
                "Mist / Cloudy (2)": 2,
                "Light Snow / Rain (3)": 3,
                "Heavy Rain / Ice Pellets (4)": 4
            }
            weathersit = st.selectbox("Weather Situation", list(weathersit_map.keys()), index=0)
            
        with f_col3:
            st.markdown("##### 🌤️ Meteorological Sensors")
            temp = st.slider("Normalized Temperature (0 to 1)", min_value=0.02, max_value=1.0, value=0.68, step=0.02, help="0.68 corresponds to approx 28°C")
            atemp = st.slider("Normalized Feeling Temp (0 to 1)", min_value=0.0, max_value=1.0, value=0.65, step=0.02)
            hum = st.slider("Normalized Humidity (0 to 1)", min_value=0.0, max_value=1.0, value=0.45, step=0.02, help="0.45 corresponds to 45% relative humidity")
            windspeed = st.slider("Normalized Windspeed (0 to 1)", min_value=0.0, max_value=0.85, value=0.18, step=0.02)
            
        submit_btn = st.form_submit_button("⚡ Predict Bike Demand", use_container_width=True, type="primary")
        
    if submit_btn:
        input_data = {
            "season": season_map[season],
            "yr": 0 if "2011" in yr else 1,
            "mnth": mnth,
            "hr": hr,
            "holiday": 1 if "Yes" in holiday else 0,
            "weekday": weekday_map[weekday],
            "workingday": 1 if "Yes" in workingday else 0,
            "weathersit": weathersit_map[weathersit],
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed,
            "day": day
        }
        
        pred_res = predict_regression(selected_key, input_data)
        
        st.markdown("""
        <div style="background-color: #eaf5ff; border: 2px solid #0072B2; border-radius: 10px; padding: 20px; margin-top: 15px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.95rem; color: #555; text-transform: uppercase; font-weight: 600;">Predicted Hourly Rental Demand</span>
                    <h1 style="color: #0072B2; margin: 5px 0 0 0; font-size: 2.8rem;">{} {}</h1>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: #0072B2; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.9rem;">{}</span><br>
                    <span style="font-size: 0.85rem; color: #444; margin-top: 8px; display: inline-block;">Baseline Mean: ~189.5 bikes ({:+.1f}%)</span>
                </div>
            </div>
        </div>
        """.format(
            pred_res["prediction"],
            pred_res["units"],
            pred_res["status"],
            pred_res["percent_diff"]
        ), unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Model Visualizations
    st.subheader(f"📊 Diagnostic Visualizations: {selected_name}")
    v_col1, v_col2 = st.columns(2)
    
    sample_actuals = model_res.get("sample_actuals", [])
    sample_preds = model_res.get("sample_preds", [])
    
    with v_col1:
        if sample_actuals and sample_preds:
            st.plotly_chart(plot_actual_vs_predicted(sample_actuals, sample_preds, selected_name), use_container_width=True)
        else:
            st.info("Evaluation sample not available for scatter plot.")
            
    with v_col2:
        if sample_actuals and sample_preds:
            st.plotly_chart(plot_residuals(sample_actuals, sample_preds, selected_name), use_container_width=True)
        else:
            st.info("Evaluation sample not available for residual plot.")

    st.markdown("---")
    
    # Full 10 Algorithm Comparison Section
    st.subheader("📈 Consolidated 10-Algorithm Regression Comparison")
    
    comp_data = []
    for k, v in results.items():
        comp_data.append({
            "Rank": REGRESSION_ALGORITHMS[k]["rank"],
            "Algorithm": v["name"],
            "R² Score": v["r2"],
            "RMSE (bikes/hr)": v["rmse"],
            "MAE (bikes/hr)": v["mae"],
            "5-Fold CV R²": f"{v['cv_mean']:.4f} ± {v['cv_std']:.4f}",
            "Training Time": f"{v['train_time']:.2f}s"
        })
    df_comp = pd.DataFrame(comp_data).sort_values("Rank")
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    st.plotly_chart(plot_regression_comparison_bar(results, sort_metric="r2"), use_container_width=True)
    
    st.markdown("""
    **Analytical Takeaway:**
    Tree-based and non-linear kernel ensembles (**Random Forest** with $R^2 = 0.9303$ and **SVR** with $R^2 = 0.9258$) substantially outperform linear baselines ($R^2 \approx 0.68$). This proves that bicycle rental dynamics are governed by non-linear interactions between the hour of the day (morning/evening rush peaks), workday status, and weather comfort thresholds.
    """)
