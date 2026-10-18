import streamlit as st
import pandas as pd
from model_registry import CLASSIFICATION_ALGORITHMS, load_classification_results
from prediction_service import predict_classification
from visualization import plot_confusion_matrix, plot_roc_curve, plot_probability_breakdown, plot_classification_comparison_bar

def render_classification_page():
    st.title("🏨 Classification Workspace — Hotel Booking Cancellation")
    st.markdown("""
    **Problem Statement:** Predict whether an upcoming hotel reservation will be **Cancelled** or **Honored (Not_Canceled)** based on guest demographics, room specifications, lead time, and booking channel attributes.
    """)
    
    results = load_classification_results()
    
    # Model Selector
    model_keys = list(CLASSIFICATION_ALGORITHMS.keys())
    model_names = [CLASSIFICATION_ALGORITHMS[k]["name"] for k in model_keys]
    
    col_sel, col_rank = st.columns([0.7, 0.3])
    with col_sel:
        selected_name = st.selectbox(
            "Select Classification Algorithm (10 Available):",
            model_names,
            index=0
        )
        
    selected_key = [k for k, v in CLASSIFICATION_ALGORITHMS.items() if v["name"] == selected_name][0]
    meta = CLASSIFICATION_ALGORITHMS[selected_key]
    model_res = results.get(selected_key, {})
    
    with col_rank:
        badge_bg = "#eaf5ff" if "Part A" in meta["track_part"] else "#fff4ea"
        badge_color = "#0072B2" if "Part A" in meta["track_part"] else "#D55E00"
        st.markdown(f"""
        <div style="background-color: {badge_bg}; border-radius: 8px; padding: 10px; border-left: 4px solid {badge_color}; margin-top: 25px;">
            <span style="font-size: 0.85rem; color: #555;">Scope: <b>{meta['track_part']}</b></span><br>
            <span style="font-size: 1.2rem; font-weight: bold; color: {badge_color};">Rank #{meta['rank']} of 10</span>
        </div>
        """, unsafe_allow_html=True)

    # Key Evaluation Metrics
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric("Accuracy", f"{model_res.get('accuracy', 0.0):.2f}%")
    with m_col2:
        st.metric("Weighted F1", f"{model_res.get('weighted_f1', 0.0):.2f}%")
    with m_col3:
        st.metric("ROC-AUC", f"{model_res.get('roc_auc', 0.0):.2f}%")
    with m_col4:
        st.metric("Precision", f"{model_res.get('precision', 0.0):.2f}%")
    with m_col5:
        st.metric("Recall", f"{model_res.get('recall', 0.0):.2f}%")
    with m_col6:
        st.metric("Training Time", f"{model_res.get('train_time', 0.0):.2f}s")
        
    with st.expander("ℹ️ Model Architecture, Hyperparameters & Details", expanded=True):
        st.markdown(f"**Algorithm Description:** {meta['desc']}")
        st.markdown(f"**Family & Scope:** `{meta['family']}` | `{meta['track_part']}`")
        st.markdown(f"**Hyperparameters:** `{meta['hyperparameters']}`")
        st.markdown(f"**Target Class Mapping:** `0 = Not_Canceled`, `1 = Canceled`")

    st.markdown("---")
    
    # Interactive Prediction Form
    st.subheader("🔮 Interactive Booking Cancellation Inference")
    st.caption("Enter reservation parameters to calculate real-time cancellation probability, class assignment, and risk categorization.")
    
    with st.form(key="class_predict_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        
        with f_col1:
            st.markdown("##### ⏳ Booking Horizon & Rates")
            lead_time = st.slider("Lead Time (Days between booking and arrival)", min_value=0, max_value=450, value=180, help="Longer lead times strongly correlate with cancellations.")
            avg_price_per_room = st.slider("Average Price per Room (€)", min_value=0.0, max_value=350.0, value=125.0, step=5.0)
            market_segment_type = st.selectbox("Market Segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"], index=0)
            type_of_meal_plan = st.selectbox("Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Not Selected", "Meal Plan 3"], index=0)
            room_type_reserved = st.selectbox("Room Type", [f"Room_Type {i}" for i in range(1, 8)], index=0)
            
        with f_col2:
            st.markdown("##### 👥 Party & Stay Duration")
            no_of_adults = st.slider("Number of Adults", min_value=1, max_value=4, value=2)
            no_of_children = st.slider("Number of Children", min_value=0, max_value=3, value=0)
            no_of_weekend_nights = st.slider("Weekend Nights (Sat/Sun)", min_value=0, max_value=5, value=1)
            no_of_week_nights = st.slider("Weekday Nights (Mon-Fri)", min_value=0, max_value=10, value=2)
            no_of_special_requests = st.slider("Special Requests (e.g. high floor, extra bed)", min_value=0, max_value=5, value=0, help="Special requests signify high guest commitment.")
            
        with f_col3:
            st.markdown("##### 📜 Historical Profile & Calendar")
            repeated_guest = st.selectbox("Repeated Guest", ["No (0)", "Yes (1)"], index=0)
            required_car_parking_space = st.selectbox("Parking Space Required", ["No (0)", "Yes (1)"], index=0)
            no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=15, value=0)
            no_of_previous_bookings_not_canceled = st.number_input("Previous Successful Stays", min_value=0, max_value=25, value=0)
            arrival_year = st.selectbox("Arrival Year", [2017, 2018], index=1)
            arrival_month = st.slider("Arrival Month", min_value=1, max_value=12, value=10)
            arrival_date = st.slider("Arrival Date of Month", min_value=1, max_value=31, value=18)
            
        submit_btn = st.form_submit_button("⚡ Predict Booking Cancellation", use_container_width=True, type="primary")
        
    if submit_btn:
        input_data = {
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": 1 if "Yes" in required_car_parking_space else 0,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "repeated_guest": 1 if "Yes" in repeated_guest else 0,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests
        }
        
        pred_res = predict_classification(selected_key, input_data)
        
        border_col = pred_res["risk_color"]
        bg_col = "#fff5f5" if pred_res["prediction_class"] == 1 else "#f3fdf6"
        
        st.markdown(f"""
        <div style="background-color: {bg_col}; border: 2px solid {border_col}; border-radius: 10px; padding: 20px; margin-top: 15px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.95rem; color: #555; text-transform: uppercase; font-weight: 600;">Predicted Booking Outcome</span>
                    <h1 style="color: {border_col}; margin: 5px 0 0 0; font-size: 2.6rem;">{pred_res['predicted_label']}</h1>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: {border_col}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.9rem;">{pred_res['risk_tier']}</span><br>
                    <span style="font-size: 0.9rem; color: #444; margin-top: 8px; display: inline-block;">P(Canceled) = <b>{pred_res['prob_canceled']}%</b> | P(Not_Canceled) = <b>{pred_res['prob_not_canceled']}%</b></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(plot_probability_breakdown(pred_res["prob_not_canceled"], pred_res["prob_canceled"]), use_container_width=True)

    st.markdown("---")
    
    # Model Visualizations
    st.subheader(f"📊 Evaluation Visualizations: {selected_name}")
    v_col1, v_col2 = st.columns(2)
    
    cm = model_res.get("confusion_matrix")
    roc_pts = model_res.get("roc_curve")
    auc_score = model_res.get("roc_auc", 0.0)
    
    with v_col1:
        if cm:
            st.plotly_chart(plot_confusion_matrix(cm, classes=["Not_Canceled", "Canceled"]), use_container_width=True)
            
    with v_col2:
        if roc_pts:
            st.plotly_chart(plot_roc_curve(roc_pts, auc_score, selected_name), use_container_width=True)

    st.markdown("---")
    
    # Full 10 Algorithm Comparison Table
    st.subheader("📈 Consolidated 10-Algorithm Classification Comparison")
    st.caption("Consolidating Review 1 (Part A) and Review 2 (Part B) algorithms on the unified held-out test split.")
    
    comp_data = []
    for k, v in results.items():
        comp_data.append({
            "Rank": CLASSIFICATION_ALGORITHMS[k]["rank"],
            "Algorithm": v["name"],
            "Scope": v["track_part"],
            "Accuracy (%)": v["accuracy"],
            "Weighted F1 (%)": v["weighted_f1"],
            "ROC-AUC (%)": v["roc_auc"],
            "Precision (%)": v["precision"],
            "Recall (%)": v["recall"],
            "Training Time": f"{v['train_time']:.2f}s"
        })
    df_comp = pd.DataFrame(comp_data).sort_values("Rank")
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    st.plotly_chart(plot_classification_comparison_bar(results, sort_metric="accuracy"), use_container_width=True)
    
    st.markdown("""
    **Analytical Takeaway:**
    Ensemble methods evaluated in **Review 2 Part B** (**Random Forest** with 90.03% accuracy and 95.53% ROC-AUC; **Bagging** with 89.00% accuracy) improve substantially over classical Part A baselines (**Logistic Regression** at 81.28%). Decision Tree ensembles effectively capture high-order non-linear interactions between `lead_time`, `avg_price_per_room`, and `no_of_special_requests` without being biased by feature correlation.
    """)
