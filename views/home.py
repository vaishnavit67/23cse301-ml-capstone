import streamlit as st
import pandas as pd
from model_registry import (
    load_regression_results, load_classification_results, load_clustering_results,
    REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS, CLUSTERING_ALGORITHMS
)

def render_home_page():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding: 30px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <h1 style="color: #ffffff; margin-bottom: 5px; font-weight: 800;">23CSE301 | Machine Learning Capstone</h1>
        <h4 style="color: #64b5f6; margin-top: 0; font-weight: 400;">Comprehensive Three-Track Machine Learning Engineering Dashboard</h4>
        <p style="color: #e0e0e0; font-size: 1.05rem; margin-bottom: 0;">
            An interactive, full-stack deployment uniting <b>all 22 algorithms</b> across Regression, Classification, and Clustering. Built for rigorous academic evaluation, live inferencing, and comparative performance analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Algorithms", value="22 Models", delta="100% Implemented")
    with col2:
        st.metric(label="Regression Track", value="10 Algorithms", delta="UCI Bike Sharing")
    with col3:
        st.metric(label="Classification Track", value="10 Algorithms", delta="Part A (5) + Part B (5)")
    with col4:
        st.metric(label="Clustering Track", value="2 + Benchmarks", delta="Spotify Audio Catalog")

    st.markdown("---")

    # Three Track Summaries
    st.subheader("📌 Three Core Machine Learning Problem Tracks")
    
    t_col1, t_col2, t_col3 = st.columns(3)
    
    # Regression Card
    with t_col1:
        st.markdown("""
        <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; background-color: #f8fafc; height: 100%;">
            <h3 style="color: #0072B2; margin-top: 0;">🚴 Track 1: Regression</h3>
            <p><b>Problem Statement:</b> Predict hourly bicycle rental demand (<code>cnt</code>) to optimize fleet logistics and station balancing.</p>
            <p><b>Dataset:</b> UCI Bike Sharing (17,379 hourly records)</p>
            <p><b>Target:</b> Continuous numerical demand (<code>cnt</code>)</p>
            <p><b>Algorithms:</b> 10 Models (Linear, Ridge, Lasso, ElasticNet, Poly, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN)</p>
            <div style="background-color: #e8f4f8; padding: 10px; border-radius: 6px; border-left: 4px solid #0072B2;">
                <b>Best Model:</b> Random Forest Regressor<br>
                <b>Test R²:</b> <code>0.9303</code> | <b>RMSE:</b> <code>47.00</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Regression Workspace →", key="btn_nav_reg", use_container_width=True):
            st.session_state["nav_selection"] = "Regression"
            st.rerun()

    # Classification Card
    with t_col2:
        st.markdown("""
        <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; background-color: #f8fafc; height: 100%;">
            <h3 style="color: #D55E00; margin-top: 0;">🏨 Track 2: Classification</h3>
            <p><b>Problem Statement:</b> Predict whether a hotel booking will be cancelled (<code>Canceled</code> vs <code>Not_Canceled</code>) to mitigate revenue loss.</p>
            <p><b>Dataset:</b> INN Hotels Group (36,275 reservation records)</p>
            <p><b>Target:</b> Binary status (<code>booking_status</code>)</p>
            <p><b>Algorithms:</b> 10 Models (Review 1 Part A: 5 Baseline; Review 2 Part B: 5 Ensembles/MLP)</p>
            <div style="background-color: #fcf0eb; padding: 10px; border-radius: 6px; border-left: 4px solid #D55E00;">
                <b>Best Model:</b> Random Forest Classifier<br>
                <b>Accuracy:</b> <code>90.03%</code> | <b>ROC-AUC:</b> <code>95.53%</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Classification Workspace →", key="btn_nav_class", use_container_width=True):
            st.session_state["nav_selection"] = "Classification"
            st.rerun()

    # Clustering Card
    with t_col3:
        st.markdown("""
        <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; background-color: #f8fafc; height: 100%;">
            <h3 style="color: #009E73; margin-top: 0;">🎵 Track 3: Clustering</h3>
            <p><b>Problem Statement:</b> Discover natural, data-driven acoustic music archetypes from intrinsic continuous audio physics without genre guidance.</p>
            <p><b>Dataset:</b> Spotify Tracks Catalog (50,000 tracks)</p>
            <p><b>Learning Type:</b> Unsupervised (Zero ground-truth leakage)</p>
            <p><b>Algorithms:</b> Agglomerative Hierarchical & K-Means + 2D High-Separation Benchmark</p>
            <div style="background-color: #edf7f2; padding: 10px; border-radius: 6px; border-left: 4px solid #009E73;">
                <b>Primary Model:</b> Core 9 RobustScaler (Complete Linkage)<br>
                <b>Silhouette:</b> <code>0.5026</code> (Stability: <code>0.5034 ± 0.0060</code>)
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Clustering Workspace →", key="btn_nav_clust", use_container_width=True):
            st.session_state["nav_selection"] = "Clustering"
            st.rerun()

    st.markdown("---")

    # Complete 22 Algorithm Inventory Table
    st.subheader("📋 Verified 22-Algorithm Master Inventory")
    st.caption("All models trained with identical preprocessed pipelines, reproducible random seeds (`random_state=42`), and strict train-only leakage prevention.")

    reg_res = load_regression_results()
    class_res = load_classification_results()
    clust_res = load_clustering_results()

    inventory_data = []

    # Regression rows
    for k, v in reg_res.items():
        meta = REGRESSION_ALGORITHMS.get(k, {})
        inventory_data.append({
            "Track": "1. Regression",
            "Review Scope": "Review 1",
            "Algorithm Name": v["name"],
            "Dataset & Target": "hour.csv | cnt (demand)",
            "Primary Metric": f"R² = {v['r2']:.4f}",
            "Secondary Metric": f"RMSE = {v['rmse']:.2f}",
            "Status": "✅ Verified"
        })

    # Classification rows
    for k, v in class_res.items():
        meta = CLASSIFICATION_ALGORITHMS.get(k, {})
        inventory_data.append({
            "Track": "2. Classification",
            "Review Scope": f"{v['track_part']}",
            "Algorithm Name": v["name"],
            "Dataset & Target": "INNHotelsGroup.csv | booking_status",
            "Primary Metric": f"Accuracy = {v['accuracy']:.2f}%",
            "Secondary Metric": f"ROC-AUC = {v['roc_auc']:.2f}%",
            "Status": "✅ Verified"
        })

    # Clustering rows
    inventory_data.append({
        "Track": "3. Clustering",
        "Review Scope": "Review 2",
        "Algorithm Name": "Agglomerative Hierarchical (Primary Model)",
        "Dataset & Target": "spotify_tracks.csv | Unsupervised",
        "Primary Metric": f"Silhouette = {clust_res['primary_model']['silhouette']:.4f}",
        "Secondary Metric": f"Davies-Bouldin = {clust_res['primary_model']['davies_bouldin']:.4f}",
        "Status": "✅ Verified"
    })
    inventory_data.append({
        "Track": "3. Clustering",
        "Review Scope": "Review 2",
        "Algorithm Name": "K-Means Clustering",
        "Dataset & Target": "spotify_tracks.csv | Unsupervised",
        "Primary Metric": f"Silhouette = {clust_res['kmeans_model']['silhouette']:.4f}",
        "Secondary Metric": f"Davies-Bouldin = {clust_res['kmeans_model']['davies_bouldin']:.4f}",
        "Status": "✅ Verified"
    })

    df_inv = pd.DataFrame(inventory_data)
    st.dataframe(df_inv, use_container_width=True, hide_index=True)

    # Course & Capstone metadata footer
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; font-size: 0.9rem;">
        <b style="color: #0072B2;">23CSE301 — Machine Learning Capstone</b> | 
        <span>Amrita Vishwa Vidyapeetham • Department of Computer Science and Engineering</span>
    </div>
    """, unsafe_allow_html=True)
