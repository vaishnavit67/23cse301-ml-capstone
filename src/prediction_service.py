import numpy as np
import pandas as pd
from model_registry import (
    load_regression_model, load_regression_metadata,
    load_classification_model, load_classification_metadata,
    load_clustering_artifacts
)
from preprocessing import (
    engineer_classification_features,
    REG_NUMERICAL, REG_CATEGORICAL,
    CORE_AUDIO_FEATURES
)

def predict_regression(model_id, input_dict):
    """
    Executes real-time regression prediction on bike demand.
    """
    meta = load_regression_metadata()
    model = load_regression_model(model_id)
    
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Apply train-derived IQR outlier bounds
    iqr_bounds = meta.get("iqr_bounds", {})
    for col, (low, high) in iqr_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
            
    # Apply ColumnTransformer
    preprocessor = meta["preprocessor"]
    X_proc = preprocessor.transform(df)
    
    # If polynomial regression, apply poly transformer
    if model_id == "polynomial_regression" and "poly_transformer" in meta:
        X_proc = meta["poly_transformer"].transform(X_proc)
        
    raw_pred = float(model.predict(X_proc)[0])
    clipped_pred = max(0.0, raw_pred)
    rounded_pred = int(round(clipped_pred))
    
    # Baseline comparison (UCI dataset overall mean is ~189.5 bikes/hr)
    dataset_mean = 189.5
    percent_diff = ((clipped_pred - dataset_mean) / dataset_mean) * 100
    
    return {
        "prediction_raw": raw_pred,
        "prediction": rounded_pred,
        "units": "bikes / hour",
        "dataset_mean": dataset_mean,
        "percent_diff": round(percent_diff, 1),
        "status": "High Demand" if rounded_pred > 250 else ("Moderate Demand" if rounded_pred >= 100 else "Low Demand")
    }


def predict_classification(model_id, input_dict):
    """
    Executes real-time hotel booking cancellation prediction.
    """
    meta = load_classification_metadata()
    model = load_classification_model(model_id)
    
    # Convert input dictionary to DataFrame
    raw_df = pd.DataFrame([input_dict])
    
    # Apply feature engineering
    df = engineer_classification_features(raw_df)
    
    # Apply train-derived IQR bounds
    iqr_bounds = meta.get("iqr_bounds", {})
    for col, (low, high) in iqr_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
            
    # Apply ColumnTransformer
    preprocessor = meta["preprocessor"]
    X_proc = preprocessor.transform(df)
    
    # Prediction
    pred_class_idx = int(model.predict(X_proc)[0])
    
    # Probability estimation
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_proc)[0]
        prob_not_canceled = float(probs[0])
        prob_canceled = float(probs[1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X_proc)[0])
        prob_canceled = 1.0 / (1.0 + np.exp(-score))
        prob_not_canceled = 1.0 - prob_canceled
    else:
        prob_canceled = 1.0 if pred_class_idx == 1 else 0.0
        prob_not_canceled = 1.0 - prob_canceled
        
    label_map = {0: "Not_Canceled", 1: "Canceled"}
    predicted_label = label_map.get(pred_class_idx, "Unknown")
    
    # Risk categorization
    if prob_canceled >= 0.65:
        risk_tier = "High Cancellation Risk"
        risk_color = "#D55E00" # Red-Orange
    elif prob_canceled >= 0.35:
        risk_tier = "Moderate Risk (Borderline)"
        risk_color = "#E69F00" # Amber
    else:
        risk_tier = "Low Cancellation Risk (Likely to Honor)"
        risk_color = "#009E73" # Green
        
    return {
        "prediction_class": pred_class_idx,
        "predicted_label": predicted_label,
        "prob_not_canceled": round(prob_not_canceled * 100, 1),
        "prob_canceled": round(prob_canceled * 100, 1),
        "risk_tier": risk_tier,
        "risk_color": risk_color
    }


def predict_clustering(input_dict):
    """
    Assigns user-submitted track audio features to an acoustic cluster archetype.
    """
    artifacts = load_clustering_artifacts()
    scaler_core = artifacts["scaler_core"]
    kmeans_model = artifacts["kmeans"]
    pca_model = artifacts["pca"]
    
    # Extract audio features in consistent order as a DataFrame
    df_raw = pd.DataFrame([[input_dict[f] for f in CORE_AUDIO_FEATURES]], columns=CORE_AUDIO_FEATURES)
    
    # Transform with RobustScaler
    X_scaled = scaler_core.transform(df_raw)
    
    # Assign cluster with K-Means
    cluster_idx = int(kmeans_model.predict(X_scaled)[0])
    
    # Project to 2D PCA space
    pca_coords = pca_model.transform(X_scaled)[0]
    
    archetypes = {
        0: {
            "name": "Modern Electric / High-Energy Archetype",
            "desc": "Characterized by high volume, high acoustic energy, modern production, and punchy rhythmic percussion (Commercial Pop, Rock, Electronic, Hip-Hop).",
            "badge_color": "#0072B2"
        },
        1: {
            "name": "Acoustic / Ambient / Organic Archetype",
            "desc": "Characterized by high organic acoustic resonance, softer dynamic volume, classical instrumentation, or tranquil ambient soundscapes (Classical, Folk, Acoustic, Ambient).",
            "badge_color": "#E69F00"
        }
    }
    
    selected_archetype = archetypes.get(cluster_idx, archetypes[0])
    
    return {
        "cluster_id": cluster_idx,
        "archetype_name": selected_archetype["name"],
        "archetype_desc": selected_archetype["desc"],
        "badge_color": selected_archetype["badge_color"],
        "pca_x": round(float(pca_coords[0]), 3),
        "pca_y": round(float(pca_coords[1]), 3)
    }
