import streamlit as st
import pandas as pd
from model_registry import (
    load_regression_results, load_classification_results, load_clustering_results,
    REGRESSION_ALGORITHMS, CLASSIFICATION_ALGORITHMS, CLUSTERING_ALGORITHMS
)
from visualization import plot_regression_comparison_bar, plot_classification_comparison_bar

def render_comparison_page():
    st.title("📊 Model Comparison Hub")
    st.markdown("""
    Compare algorithm performance within each problem track using verified test-set metrics, cross-validation distributions, and interactive rankings.
    """)
    
    track_choice = st.radio(
        "Select Machine Learning Track to Compare:",
        ["1. Regression (10 Algorithms)", "2. Classification (10 Algorithms)", "3. Clustering Track"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if "Regression" in track_choice:
        st.subheader("🚴 Regression Track: All 10 Algorithms Ranked")
        results = load_regression_results()
        
        sort_by = st.selectbox("Sort Table & Chart By:", ["R² Score (Higher is better)", "RMSE (Lower is better)", "MAE (Lower is better)", "5-Fold CV R² (Higher is better)"])
        metric_key = "r2" if "R² Score" in sort_by else ("rmse" if "RMSE" in sort_by else ("mae" if "MAE" in sort_by else "cv_mean"))
        
        table_rows = []
        for k, v in results.items():
            meta = REGRESSION_ALGORITHMS.get(k, {})
            table_rows.append({
                "Rank": meta["rank"],
                "Algorithm": v["name"],
                "Family": meta.get("family", "Regression"),
                "R² Score": v["r2"],
                "RMSE (bikes/hr)": v["rmse"],
                "MAE (bikes/hr)": v["mae"],
                "5-Fold CV R²": f"{v['cv_mean']:.4f} ± {v['cv_std']:.4f}",
                "Training Time": f"{v['train_time']:.2f}s"
            })
            
        df_reg = pd.DataFrame(table_rows)
        if metric_key in ["rmse", "mae"]:
            df_reg = df_reg.sort_values(by="RMSE (bikes/hr)" if metric_key == "rmse" else "MAE (bikes/hr)", ascending=True)
        elif metric_key == "r2":
            df_reg = df_reg.sort_values(by="R² Score", ascending=False)
            
        st.dataframe(df_reg, use_container_width=True, hide_index=True)
        st.plotly_chart(plot_regression_comparison_bar(results, sort_metric=metric_key), use_container_width=True)
        
        # Best model highlight
        st.markdown("""
        <div style="background-color: #f0f7ff; border-left: 5px solid #0072B2; padding: 15px; border-radius: 6px;">
            <h4 style="margin: 0; color: #0072B2;">🏆 Best Regression Model: Random Forest Regressor</h4>
            <p style="margin: 5px 0 0 0; color: #333;">
                Achieved highest <b>Test R² = 0.9303</b> and lowest <b>RMSE = 46.99 bikes/hour</b>. The 5-fold cross-validation score (<b>0.9180 ± 0.0025</b>) confirms low variance and high generalizability across arbitrary seasonal subsets.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    elif "Classification" in track_choice:
        st.subheader("🏨 Classification Track: All 10 Algorithms Ranked")
        st.caption("Unites Review 1 (Part A: 5 Baselines) and Review 2 (Part B: 5 Ensembles/MLP) on the identical held-out test split.")
        
        results = load_classification_results()
        
        sort_by = st.selectbox("Sort Table & Chart By:", ["Accuracy (%)", "Weighted F1 (%)", "ROC-AUC (%)", "Precision (%)", "Recall (%)"])
        metric_key = "accuracy" if "Accuracy" in sort_by else ("weighted_f1" if "Weighted F1" in sort_by else "roc_auc")
        
        table_rows = []
        for k, v in results.items():
            meta = CLASSIFICATION_ALGORITHMS.get(k, {})
            table_rows.append({
                "Rank": meta["rank"],
                "Algorithm": v["name"],
                "Scope": v["track_part"],
                "Family": meta.get("family", "Classification"),
                "Accuracy (%)": v["accuracy"],
                "Weighted F1 (%)": v["weighted_f1"],
                "ROC-AUC (%)": v["roc_auc"],
                "Precision (%)": v["precision"],
                "Recall (%)": v["recall"],
                "Training Time": f"{v['train_time']:.2f}s"
            })
            
        df_class = pd.DataFrame(table_rows).sort_values(by=sort_by, ascending=False)
        st.dataframe(df_class, use_container_width=True, hide_index=True)
        
        st.plotly_chart(plot_classification_comparison_bar(results, sort_metric=metric_key), use_container_width=True)
        
        # Best model highlight
        st.markdown("""
        <div style="background-color: #fdf5f0; border-left: 5px solid #D55E00; padding: 15px; border-radius: 6px;">
            <h4 style="margin: 0; color: #D55E00;">🏆 Best Classification Model: Random Forest Classifier (Part B)</h4>
            <p style="margin: 5px 0 0 0; color: #333;">
                Achieved top performance across all dimensions: <b>Accuracy = 90.03%</b>, <b>Weighted F1 = 89.89%</b>, and <b>ROC-AUC = 95.53%</b>. Outperforms classical Part A algorithms (Logistic Regression at 81.28%) by effectively learning high-order guest booking decision boundaries.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.subheader("🎵 Clustering Track: Evaluated Unsupervised Configurations")
        st.caption("Clustering models cannot be ranked by accuracy or R² because ground-truth labels are absent during training. Comparison relies strictly on geometric cohesion and separation trade-offs.")
        
        results = load_clustering_results()
        
        clust_table = [
            {
                "Configuration": "Exp C: Core 9 Audio Features (PRIMARY MODEL)",
                "Algorithm": "Agglomerative Hierarchical",
                "Linkage": "Complete",
                "Features": "9 Core Perceptual Dimensions",
                "Scaler": "RobustScaler",
                "k": 2,
                "Silhouette Score": results["primary_model"]["silhouette"],
                "Davies-Bouldin": results["primary_model"]["davies_bouldin"],
                "Calinski-Harabasz": results["primary_model"]["calinski_harabasz"],
                "Stability (5 Seeds)": results["primary_model"]["multi_seed_mean"],
                "Role in Project": "PRIMARY CAPSTONE MODEL"
            },
            {
                "Configuration": "Exp A: 2D Loudness + Acousticness (BENCHMARK)",
                "Algorithm": "Agglomerative Hierarchical",
                "Linkage": "Average",
                "Features": "2D (Loudness, Acousticness)",
                "Scaler": "MinMaxScaler",
                "k": 2,
                "Silhouette Score": results["experiment_a_benchmark"]["silhouette"],
                "Davies-Bouldin": results["experiment_a_benchmark"]["davies_bouldin"],
                "Calinski-Harabasz": results["experiment_a_benchmark"]["calinski_harabasz"],
                "Stability (5 Seeds)": results["experiment_a_benchmark"]["multi_seed_mean"],
                "Role in Project": "Ablation Study (Discards 7 features)"
            },
            {
                "Configuration": "Exp B: 2D Loudness + Acousticness (K-Means)",
                "Algorithm": "K-Means",
                "Linkage": "N/A",
                "Features": "2D (Loudness, Acousticness)",
                "Scaler": "MinMaxScaler",
                "k": 2,
                "Silhouette Score": 0.5952,
                "Davies-Bouldin": 0.6324,
                "Calinski-Harabasz": 2730.22,
                "Stability (5 Seeds)": "N/A",
                "Role in Project": "K-Means 2D Alternative"
            },
            {
                "Configuration": "Exp D: PCA 2-Component Model",
                "Algorithm": "Agglomerative Hierarchical",
                "Linkage": "Complete",
                "Features": "PCA 2 Components (Core 9)",
                "Scaler": "StandardScaler -> PCA",
                "k": 2,
                "Silhouette Score": 0.5777,
                "Davies-Bouldin": 0.4432,
                "Calinski-Harabasz": 1017.07,
                "Stability (5 Seeds)": "0.5113 ± 0.1233",
                "Role in Project": "Dimensionality Reduction Benchmark"
            },
            {
                "Configuration": "Exp E: Single Linkage Chaining Check",
                "Algorithm": "Agglomerative Hierarchical",
                "Linkage": "Single",
                "Features": "2D Contrast",
                "Scaler": "MinMaxScaler",
                "k": 2,
                "Silhouette Score": 0.3174,
                "Davies-Bouldin": 0.6321,
                "Calinski-Harabasz": 4.44,
                "Stability (5 Seeds)": "Degenerate Chaining",
                "Role in Project": "REJECTED ({0: 1498, 1: 2})"
            }
        ]
        
        df_c = pd.DataFrame(clust_table)
        st.dataframe(df_c, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div style="background-color: #f0faf4; border-left: 5px solid #009E73; padding: 15px; border-radius: 6px;">
            <h4 style="margin: 0; color: #009E73;">🏆 Recommended Capstone Presentation: Experiment C (Core 9 Audio)</h4>
            <p style="margin: 5px 0 0 0; color: #333;">
                While Experiment A achieves a higher mathematical Silhouette score (0.6054 on sample) by aggressively discarding 7 audio dimensions, <b>Experiment C</b> preserves all nine perceptual audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo) and has the lowest observed score variability across tested seeds (<b>0.5034 ± 0.0060</b>).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Educational Metric Guide
    with st.expander("📚 Guide to Evaluation Metrics Used in this Capstone", expanded=False):
        st.markdown(r"""
        ### Regression Metrics
        - **$R^2$ (Coefficient of Determination):** Proportion of target variance explained by predictors. Range $(-\infty, 1]$. Higher is better. $1.0$ is perfect fit.
        - **RMSE (Root Mean Squared Error):** Square root of the average squared errors. Heavily penalizes large prediction outliers. Expressed in original target units (bikes/hour). Lower is better.
        - **MAE (Mean Absolute Error):** Average magnitude of absolute errors. Linear and robust to extreme outliers. Expressed in bikes/hour. Lower is better.
        
        ### Classification Metrics
        - **Accuracy:** Proportion of correct classifications among all predictions. Useful when classes are balanced.
        - **Precision:** $TP / (TP + FP)$. Measures reliability when predicting positive cancellations (minimizing false alarms).
        - **Recall (Sensitivity):** $TP / (TP + FN)$. Measures completeness in catching actual cancellations.
        - **Weighted F1-Score:** Harmonic mean of precision and recall weighted by class support, balancing both false positives and false negatives.
        - **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures discrimination capability across all classification probability thresholds. Range $[0.5, 1.0]$.
        
        ### Clustering Validation Metrics
        - **Silhouette Score:** Measures cohesion vs. separation: $(b - a) / \max(a, b)$. Range $[-1, 1]$. Values above 0.5 indicate strong, defensible cluster separation. Higher is better.
        - **Davies–Bouldin Index (DB):** Ratio of within-cluster distances to between-cluster distances. Lower values indicate tighter, better-separated clusters. Zero is optimal.
        - **Calinski–Harabasz Index (CH):** Ratio of between-cluster dispersion to within-cluster dispersion (variance ratio criterion). Higher values indicate denser, better-separated clusters.
        """)
