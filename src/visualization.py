import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Colorblind-safe palette (Wong 2011 / Seaborn colorblind)
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "amber": "#F0E442",
    "dark_blue": "#004949",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "bg_dark": "#0E1117",
    "card_bg": "#1E222A"
}

def plot_actual_vs_predicted(actuals, preds, model_name="Selected Model"):
    """
    Renders actual vs. predicted scatter with ideal y=x reference line.
    """
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=actuals,
        y=preds,
        mode="markers",
        marker=dict(
            color=PALETTE["blue"],
            size=6,
            opacity=0.6,
            line=dict(width=0.5, color="white")
        ),
        name="Test Observations",
        hovertemplate="Actual: %{x}<br>Predicted: %{y:.1f}<extra></extra>"
    ))
    
    # Ideal line
    min_val = min(min(actuals), min(preds))
    max_val = max(max(actuals), max(preds))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=PALETTE["vermilion"], width=2, dash="dash"),
        name="Ideal Fit (y = x)"
    ))
    
    fig.update_layout(
        title=f"Actual vs. Predicted Rental Demand — {model_name}",
        xaxis_title="Actual Demand (cnt)",
        yaxis_title="Predicted Demand (cnt)",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
    )
    return fig


def plot_residuals(actuals, preds, model_name="Selected Model"):
    """
    Renders residual scatter plot and distribution histogram.
    """
    residuals = np.array(actuals) - np.array(preds)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Residuals vs. Predicted Values", "Residual Error Distribution"),
        column_widths=[0.65, 0.35]
    )
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=preds,
        y=residuals,
        mode="markers",
        marker=dict(color=PALETTE["green"], size=5, opacity=0.6),
        name="Residuals",
        hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>"
    ), row=1, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["vermilion"], row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(
        y=residuals,
        nbinsy=30,
        marker=dict(color=PALETTE["blue"], opacity=0.7),
        name="Distribution",
        showlegend=False
    ), row=1, col=2)
    
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    fig.update_xaxes(title_text="Predicted Demand", row=1, col=1)
    fig.update_yaxes(title_text="Residual Error (y - y_hat)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency", row=1, col=2)
    return fig


def plot_confusion_matrix(cm, classes=None):
    """
    Interactive annotated confusion matrix heatmap.
    """
    if classes is None:
        classes = ["Not_Canceled", "Canceled"]
        
    cm_arr = np.array(cm)
    total = np.sum(cm_arr)
    annotations = []
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_arr[i, j]
            pct = (val / total) * 100
            annotations.append(
                f"<b>{val:,}</b><br>({pct:.1f}%)"
            )
            
    text_matrix = np.array(annotations).reshape(len(classes), len(classes))
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=classes,
        y=classes,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 14},
        colorscale=[[0, "#EBF3FB"], [1, PALETTE["blue"]]],
        showscale=False
    ))
    
    fig.update_layout(
        title="Test Set Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        yaxis=dict(autorange="reversed")
    )
    return fig


def plot_roc_curve(roc_pts, auc_score, model_name="Selected Classifier"):
    """
    Interactive ROC Curve with AUC shading.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=roc_pts["fpr"],
        y=roc_pts["tpr"],
        mode="lines",
        line=dict(color=PALETTE["blue"], width=3),
        name=f"ROC Curve (AUC = {auc_score:.2f}%)",
        fill="tozeroy",
        fillcolor="rgba(0, 114, 178, 0.15)"
    ))
    
    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color=PALETTE["grey"], width=2, dash="dash"),
        name="Random Classifier (AUC = 50.0%)"
    ))
    
    fig.update_layout(
        title=f"Receiver Operating Characteristic (ROC) — {model_name}",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95)
    )
    return fig


def plot_probability_breakdown(prob_not_canceled, prob_canceled):
    """
    Visual stacked confidence bar for classification inferences.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[prob_not_canceled],
        name="Not_Canceled",
        orientation="h",
        marker=dict(color=PALETTE["green"]),
        text=f"Not_Canceled: {prob_not_canceled:.1f}%",
        textposition="inside",
        insidetextanchor="middle"
    ))
    
    fig.add_trace(go.Bar(
        y=["Probability"],
        x=[prob_canceled],
        name="Canceled",
        orientation="h",
        marker=dict(color=PALETTE["vermilion"]),
        text=f"Canceled: {prob_canceled:.1f}%",
        textposition="inside",
        insidetextanchor="middle"
    ))
    
    fig.update_layout(
        barmode="stack",
        height=100,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showticklabels=False, range=[0, 100], showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        showlegend=False,
        template="plotly_white"
    )
    return fig


def plot_clustering_scatter(sample_data, proj="pca", algorithm="hierarchical", user_point=None):
    """
    Renders 2D cluster scatter plot with optional user track projection.
    """
    x_col = f"{proj}_x"
    y_col = f"{proj}_y"
    lbl_col = "hier_labels" if algorithm == "hierarchical" else "kmeans_labels"
    
    df_pts = pd.DataFrame({
        "x": sample_data[x_col],
        "y": sample_data[y_col],
        "Cluster": [f"Cluster {l}" for l in sample_data[lbl_col]],
        "Track": sample_data["track_name"],
        "Artist": sample_data["artist_name"],
        "Genre": sample_data["genre"]
    })
    
    color_map = {
        "Cluster 0": PALETTE["blue"],
        "Cluster 1": PALETTE["orange"]
    }
    
    fig = px.scatter(
        df_pts,
        x="x",
        y="y",
        color="Cluster",
        color_discrete_map=color_map,
        hover_data=["Track", "Artist", "Genre"],
        title=f"2D {proj.upper()} Manifold Projection ({algorithm.capitalize()} Clustering)"
    )
    
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    
    # Overlay user custom track if provided
    if user_point and proj == "pca":
        fig.add_trace(go.Scatter(
            x=[user_point["pca_x"]],
            y=[user_point["pca_y"]],
            mode="markers+text",
            marker=dict(
                size=18,
                color="#F0E442",
                symbol="star",
                line=dict(color="black", width=2)
            ),
            text=["YOUR TRACK"],
            textposition="top center",
            name="Your Custom Track"
        ))
        
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title=f"{proj.upper()} Component 1",
        yaxis_title=f"{proj.upper()} Component 2",
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
    )
    return fig


def plot_elbow_curve(k_range, inertias, silhouettes):
    """
    Dual-axis Plotly Elbow Curve & Silhouette score trajectory.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Inertia curve
    fig.add_trace(go.Scatter(
        x=k_range,
        y=inertias,
        mode="lines+markers",
        line=dict(color=PALETTE["blue"], width=3),
        marker=dict(size=8),
        name="Inertia (WCSS)"
    ), secondary_y=False)
    
    # Silhouette curve
    fig.add_trace(go.Scatter(
        x=k_range,
        y=silhouettes,
        mode="lines+markers",
        line=dict(color=PALETTE["orange"], width=3, dash="dot"),
        marker=dict(size=8),
        name="Silhouette Score"
    ), secondary_y=True)
    
    fig.update_layout(
        title="K-Means Cluster Count Selection: Elbow Method & Silhouette Curve",
        xaxis_title="Number of Clusters (k)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95)
    )
    fig.update_yaxes(title_text="Inertia (Within-Cluster Sum of Squares)", secondary_y=False)
    fig.update_yaxes(title_text="Average Silhouette Width", secondary_y=True)
    return fig


def plot_dendrogram_fig(linkage_matrix):
    """
    Generates Matplotlib figure of hierarchical dendrogram.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=12,
        leaf_rotation=45.,
        leaf_font_size=10.,
        show_contracted=True,
        ax=ax,
        color_threshold=linkage_matrix[-1, 2] * 0.7
    )
    ax.set_title("Agglomerative Hierarchical Clustering Dendrogram (Truncated)", fontweight="bold")
    ax.set_xlabel("Cluster Sample / Sub-tree Index")
    ax.set_ylabel("Ward Linkage Euclidean Distance")
    plt.tight_layout()
    return fig


def plot_regression_comparison_bar(results_dict, sort_metric="r2"):
    """
    Comparative bar chart for all 10 regression algorithms.
    """
    data = []
    for k, v in results_dict.items():
        data.append({
            "Algorithm": v["name"],
            "R2": v["r2"],
            "RMSE": v["rmse"],
            "MAE": v["mae"],
            "CV_R2": v["cv_mean"]
        })
    df = pd.DataFrame(data).sort_values(by=sort_metric.upper() if sort_metric != "cv_mean" else "CV_R2", ascending=(sort_metric == "rmse" or sort_metric == "mae"))
    
    fig = px.bar(
        df,
        x="Algorithm",
        y=sort_metric.upper() if sort_metric != "cv_mean" else "CV_R2",
        color=sort_metric.upper() if sort_metric != "cv_mean" else "CV_R2",
        color_continuous_scale="Blues",
        text_auto=".3f",
        title=f"All 10 Regression Algorithms Ranked by {sort_metric.upper()}"
    )
    fig.update_layout(
        template="plotly_white",
        height=450,
        xaxis_tickangle=-35,
        margin=dict(l=40, r=40, t=50, b=100)
    )
    return fig


def plot_classification_comparison_bar(results_dict, sort_metric="accuracy"):
    """
    Comparative bar chart for all 10 classification algorithms.
    """
    data = []
    for k, v in results_dict.items():
        data.append({
            "Algorithm": v["name"],
            "Track Part": v["track_part"],
            "Accuracy": v["accuracy"],
            "Weighted F1": v["weighted_f1"],
            "ROC-AUC": v["roc_auc"]
        })
    metric_label = "Accuracy" if sort_metric == "accuracy" else ("Weighted F1" if sort_metric == "weighted_f1" else "ROC-AUC")
    df = pd.DataFrame(data).sort_values(by=metric_label, ascending=False)
    
    fig = px.bar(
        df,
        x="Algorithm",
        y=metric_label,
        color="Track Part",
        color_discrete_map={"Part A": PALETTE["blue"], "Part B": PALETTE["orange"]},
        text_auto=".2f",
        title=f"All 10 Classification Algorithms Ranked by {metric_label} (%)"
    )
    fig.update_layout(
        template="plotly_white",
        height=450,
        xaxis_tickangle=-35,
        margin=dict(l=40, r=40, t=50, b=100),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
    )
    return fig
