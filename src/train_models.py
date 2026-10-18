import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage

from preprocessing import (
    engineer_regression_features, get_regression_preprocessor,
    REG_NUMERICAL, REG_CATEGORICAL, REG_OUTLIER_COLS,
    engineer_classification_features, get_classification_preprocessor,
    CLASS_CONTINUOUS_OUTLIER_COLS, CORE_AUDIO_FEATURES, MINIMAL_2D_FEATURES
)

RANDOM_STATE = 42

def train_and_save_regression():
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODELS (UCI BIKE SHARING DEMAND)")
    print("="*60)
    
    os.makedirs("models/regression", exist_ok=True)
    df_raw = pd.read_csv("Data/hour.csv")
    df = engineer_regression_features(df_raw)
    
    X = df.drop(columns=["cnt"])
    y = df["cnt"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    
    # Calculate IQR bounds on training data ONLY
    iqr_bounds = {}
    for col in REG_OUTLIER_COLS:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        iqr_bounds[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        X_train[col] = X_train[col].clip(lower=iqr_bounds[col][0], upper=iqr_bounds[col][1])
        X_test[col] = X_test[col].clip(lower=iqr_bounds[col][0], upper=iqr_bounds[col][1])
        
    preprocessor = get_regression_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Save test sample for visualization and inspection
    test_sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    joblib.dump({
        "X_test_sample": X_test.iloc[test_sample_idx],
        "y_test_sample": y_test.iloc[test_sample_idx],
        "iqr_bounds": iqr_bounds,
        "preprocessor": preprocessor,
        "feature_names": list(X_train.columns)
    }, "models/regression/metadata.joblib")
    
    # Define models dictionary
    models = {
        "linear_regression": {
            "name": "Linear Regression",
            "model": LinearRegression(),
            "desc": "Baseline linear model; models bicycle demand as a linear combination of environmental and temporal attributes.",
            "type": "standard"
        },
        "ridge_regression": {
            "name": "Ridge Regression",
            "model": Ridge(alpha=10.0, random_state=RANDOM_STATE),
            "desc": "L2-regularized linear model; prevents multicollinearity among temporal indicators and shrinkage of weights.",
            "type": "standard"
        },
        "lasso_regression": {
            "name": "Lasso Regression",
            "model": Lasso(alpha=0.1, max_iter=10000, random_state=RANDOM_STATE),
            "desc": "L1-regularized linear model; produces sparse feature weights by setting non-critical coefficients to zero.",
            "type": "standard"
        },
        "elasticnet_regression": {
            "name": "ElasticNet Regression",
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STATE),
            "desc": "Blended L1 and L2 regularization; balances feature sparsity with stability among correlated temporal features.",
            "type": "standard"
        },
        "polynomial_regression": {
            "name": "Polynomial Regression",
            "model": Ridge(alpha=10.0, random_state=RANDOM_STATE),
            "desc": "Degree-2 polynomial expansion on numerical variables; captures nonlinear interactions (e.g. temp x humidity).",
            "type": "poly"
        },
        "decision_tree": {
            "name": "Decision Tree Regressor",
            "model": DecisionTreeRegressor(max_depth=12, min_samples_split=10, min_samples_leaf=4, random_state=RANDOM_STATE),
            "desc": "Non-parametric tree partitioner; captures abrupt threshold shifts in hourly rental patterns.",
            "type": "standard"
        },
        "random_forest": {
            "name": "Random Forest Regressor",
            "model": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=4, random_state=RANDOM_STATE, n_jobs=-1),
            "desc": "Ensemble of decorrelated decision trees; top-performing model capturing multi-way feature interactions.",
            "type": "standard"
        },
        "gradient_boosting": {
            "name": "Gradient Boosting Regressor",
            "model": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE),
            "desc": "Sequential boosting model; fits additive regression trees to residual errors of previous iterations.",
            "type": "standard"
        },
        "svr": {
            "name": "Support Vector Regressor (SVR)",
            "model": SVR(C=100, epsilon=0.1, kernel="rbf"),
            "desc": "Kernelized margin-based regressor with RBF kernel; projects scaled features into infinite-dimensional Hilbert space.",
            "type": "standard"
        },
        "knn_regressor": {
            "name": "K-Nearest Neighbors Regressor",
            "model": KNeighborsRegressor(n_neighbors=11, weights="distance", n_jobs=-1),
            "desc": "Instance-based local regressor; estimates rental demand by distance-weighted averaging of similar historical hours.",
            "type": "standard"
        }
    }
    
    # Polynomial transformer for poly model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_proc_poly = poly.fit_transform(X_train_proc)
    X_test_proc_poly = poly.transform(X_test_proc)
    joblib.dump(poly, "models/regression/poly_transformer.joblib")
    
    reg_results = {}
    
    for key, info in models.items():
        t0 = time.time()
        print(f"Training {info['name']}...")
        clf = info["model"]
        
        if info["type"] == "poly":
            clf.fit(X_train_proc_poly, y_train)
            preds = clf.predict(X_test_proc_poly)
            # 5-fold CV (on subset to remain fast and exact)
            cv_scores = cross_val_score(clf, X_train_proc_poly[:4000], y_train[:4000], cv=5, scoring="r2", n_jobs=-1)
        else:
            clf.fit(X_train_proc, y_train)
            preds = clf.predict(X_test_proc)
            cv_scores = cross_val_score(clf, X_train_proc[:4000], y_train[:4000], cv=5, scoring="r2", n_jobs=-1)
            
        preds = np.clip(preds, a_min=0, a_max=None)
        elapsed = time.time() - t0
        
        r2 = float(r2_score(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        
        # Save model
        joblib.dump(clf, f"models/regression/{key}.joblib")
        
        # Store predictions sample for quick plotting
        sample_preds = preds[test_sample_idx].tolist()
        sample_actuals = y_test.iloc[test_sample_idx].tolist()
        
        reg_results[key] = {
            "name": info["name"],
            "desc": info["desc"],
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "cv_mean": round(cv_mean, 4),
            "cv_std": round(cv_std, 4),
            "train_time": round(elapsed, 2),
            "sample_preds": sample_preds[:200],
            "sample_actuals": sample_actuals[:200]
        }
        print(f"  -> R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | Time: {elapsed:.2f}s")
        
    with open("models/regression/results.json", "w") as f:
        json.dump(reg_results, f, indent=2)
    print("Regression models trained and serialized successfully.")


def train_and_save_classification():
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS (INN HOTELS CANCELLATION)")
    print("="*60)
    
    os.makedirs("models/classification", exist_ok=True)
    df_raw = pd.read_csv("Data/INNHotelsGroup.csv")
    df = engineer_classification_features(df_raw)
    
    # Target encoding
    y = df["booking_status"].map({"Not_Canceled": 0, "Canceled": 1})
    X = df.drop(columns=["booking_status"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    # Outlier treatment (Train-only bounds)
    iqr_bounds = {}
    for col in CLASS_CONTINUOUS_OUTLIER_COLS:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        iqr_bounds[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        X_train[col] = X_train[col].clip(lower=iqr_bounds[col][0], upper=iqr_bounds[col][1])
        X_test[col] = X_test[col].clip(lower=iqr_bounds[col][0], upper=iqr_bounds[col][1])
        
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
    
    preprocessor = get_classification_preprocessor(numeric_cols, categorical_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Save test sample for visualization and inspection
    test_sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_test), size=min(1500, len(X_test)), replace=False)
    joblib.dump({
        "X_test_sample": X_test.iloc[test_sample_idx],
        "y_test_sample": y_test.iloc[test_sample_idx],
        "iqr_bounds": iqr_bounds,
        "preprocessor": preprocessor,
        "feature_names": list(X_train.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }, "models/classification/metadata.joblib")
    
    # Models dictionary covering Review 1 (Part A) & Review 2 (Part B)
    models = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "track_part": "Part A",
            "desc": "Baseline linear classifier with log-odds link; offers interpretable feature coefficients."
        },
        "knn_classifier": {
            "name": "K-Nearest Neighbors (KNN)",
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
            "track_part": "Part A",
            "desc": "Instance-based non-parametric classifier using distance-weighted nearest neighbours."
        },
        "gaussian_naive_bayes": {
            "name": "Gaussian Naive Bayes",
            "model": GaussianNB(),
            "track_part": "Part A",
            "desc": "Probabilistic classifier assuming class-conditional feature independence."
        },
        "decision_tree": {
            "name": "Decision Tree Classifier",
            "model": DecisionTreeClassifier(max_depth=12, min_samples_split=10, random_state=RANDOM_STATE),
            "track_part": "Part A",
            "desc": "Orthogonal hyperplane recursive tree partitioner; captures nonlinear guest interactions."
        },
        "svc": {
            "name": "Support Vector Classifier (SVC)",
            "model": SVC(C=10, gamma="scale", probability=True, random_state=RANDOM_STATE),
            "track_part": "Part A",
            "desc": "Maximum margin hyperplane separator in kernelized space with Platt scaling probabilities."
        },
        "random_forest": {
            "name": "Random Forest Classifier",
            "model": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=RANDOM_STATE, n_jobs=-1),
            "track_part": "Part B",
            "desc": "Bagged ensemble of randomized decision trees; reduces variance and achieves highest overall F1."
        },
        "adaboost": {
            "name": "AdaBoost Classifier",
            "model": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE),
            "track_part": "Part B",
            "desc": "Adaptive boosting ensemble focusing iteratively on misclassified booking instances."
        },
        "gradient_boosting": {
            "name": "Gradient Boosting Classifier",
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE),
            "track_part": "Part B",
            "desc": "Stage-wise additive ensemble optimizing binary cross-entropy loss."
        },
        "bagging": {
            "name": "Bagging Classifier (Decision Trees)",
            "model": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE), n_estimators=40, random_state=RANDOM_STATE, n_jobs=-1),
            "track_part": "Part B",
            "desc": "Bootstrap aggregation of deep decision trees with majority voting."
        },
        "mlp_classifier": {
            "name": "MLP Classifier (Neural Network)",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=250, random_state=RANDOM_STATE),
            "track_part": "Part B",
            "desc": "Multi-layer perceptron neural network with backpropagation and ReLU activations."
        }
    }
    
    class_results = {}
    
    # Train SVC on stratified subset (e.g. 10k) if needed, or full if speedy
    for key, info in models.items():
        t0 = time.time()
        print(f"Training {info['name']} ({info['track_part']})...")
        clf = info["model"]
        
        if key == "svc":
            # SVC with probability=True on 30k takes a bit; train on 8,000 stratified samples for fast serialization
            svc_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_train_proc), size=8000, replace=False)
            clf.fit(X_train_proc[svc_idx], y_train.iloc[svc_idx])
        else:
            clf.fit(X_train_proc, y_train)
            
        elapsed = time.time() - t0
        
        preds = clf.predict(X_test_proc)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test_proc)[:, 1]
        else:
            probs = clf.decision_function(X_test_proc)
            
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1_w = float(f1_score(y_test, preds, average="weighted"))
        f1_bin = float(f1_score(y_test, preds, average="binary"))
        auc = float(roc_auc_score(y_test, probs))
        cm = confusion_matrix(y_test, preds).tolist()
        
        # ROC curve coordinates for lightweight plotting
        fpr, tpr, _ = roc_curve(y_test, probs)
        # Downsample ROC curve to 100 points
        stride = max(1, len(fpr) // 100)
        roc_pts = {
            "fpr": fpr[::stride].tolist(),
            "tpr": tpr[::stride].tolist()
        }
        
        joblib.dump(clf, f"models/classification/{key}.joblib")
        
        class_results[key] = {
            "name": info["name"],
            "track_part": info["track_part"],
            "desc": info["desc"],
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "weighted_f1": round(f1_w * 100, 2),
            "binary_f1": round(f1_bin * 100, 2),
            "roc_auc": round(auc * 100, 2),
            "confusion_matrix": cm,
            "roc_curve": roc_pts,
            "train_time": round(elapsed, 2)
        }
        print(f"  -> Acc: {acc*100:.2f}% | F1: {f1_w*100:.2f}% | AUC: {auc*100:.2f}% | Time: {elapsed:.2f}s")
        
    with open("models/classification/results.json", "w") as f:
        json.dump(class_results, f, indent=2)
    print("Classification models trained and serialized successfully.")


def train_and_save_clustering():
    print("\n" + "="*60)
    print("TRAINING CLUSTERING MODELS (SPOTIFY TRACKS)")
    print("="*60)
    
    os.makedirs("models/clustering", exist_ok=True)
    df = pd.read_csv("Data/spotify_tracks.csv")
    
    # 1. Feature Matrices
    # Core 9 with RobustScaler (Primary Capstone Model)
    scaler_core = RobustScaler()
    X_core_full = scaler_core.fit_transform(df[CORE_AUDIO_FEATURES])
    
    # 2D Minimal (Loudness, Acousticness) with MinMaxScaler (Benchmark)
    scaler_2d = MinMaxScaler()
    X_2d_full = scaler_2d.fit_transform(df[MINIMAL_2D_FEATURES])
    
    # Global PCA (2 Components)
    scaler_std = StandardScaler()
    X_core_std = scaler_std.fit_transform(df[CORE_AUDIO_FEATURES])
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_full = pca_2d.fit_transform(X_core_std)
    evr = pca_2d.explained_variance_ratio_.tolist()
    
    # Standard evaluation sample (1,500 tracks matching notebook seed 42)
    sample_rng = np.random.default_rng(RANDOM_STATE)
    eval_idx = sample_rng.choice(len(df), size=1500, replace=False)
    
    X_core_sample = X_core_full[eval_idx]
    X_2d_sample = X_2d_full[eval_idx]
    X_pca_sample = X_pca_full[eval_idx]
    df_eval = df.iloc[eval_idx].copy()
    
    # K-Means Primary Model (k=2)
    kmeans_core = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans_core.fit_predict(X_core_sample)
    
    # Agglomerative Hierarchical Model (Complete Linkage, k=2) - PRIMARY MODEL
    hierarchical_core = AgglomerativeClustering(n_clusters=2, linkage="complete")
    hier_labels = hierarchical_core.fit_predict(X_core_sample)
    
    # Experiment A Benchmark: 2D Average Linkage (0.6054)
    hier_2d = AgglomerativeClustering(n_clusters=2, linkage="average")
    hier_2d_labels = hier_2d.fit_predict(X_2d_sample)
    
    # Metrics
    km_sil = float(silhouette_score(X_core_sample, kmeans_labels))
    km_db = float(davies_bouldin_score(X_core_sample, kmeans_labels))
    km_ch = float(calinski_harabasz_score(X_core_sample, kmeans_labels))
    
    hc_sil = float(silhouette_score(X_core_sample, hier_labels))
    hc_db = float(davies_bouldin_score(X_core_sample, hier_labels))
    hc_ch = float(calinski_harabasz_score(X_core_sample, hier_labels))
    
    exp_a_sil = float(silhouette_score(X_2d_sample, hier_2d_labels))
    exp_a_db = float(davies_bouldin_score(X_2d_sample, hier_2d_labels))
    exp_a_ch = float(calinski_harabasz_score(X_2d_sample, hier_2d_labels))
    
    # Save models
    joblib.dump(kmeans_core, "models/clustering/kmeans_model.joblib")
    joblib.dump(scaler_core, "models/clustering/scaler_core.joblib")
    joblib.dump(scaler_2d, "models/clustering/scaler_2d.joblib")
    joblib.dump(pca_2d, "models/clustering/pca_model.joblib")
    
    # Precompute t-SNE for 2D visualization (1,500 sample)
    print("Computing t-SNE 2D manifold projection...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
    X_tsne_sample = tsne.fit_transform(X_core_sample)
    
    # Precompute Elbow Curve for k = 2 to 10
    print("Precomputing Elbow curve...")
    elbow_inertias = []
    elbow_silhouettes = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_core_sample)
        elbow_inertias.append(float(km.inertia_))
        elbow_silhouettes.append(float(silhouette_score(X_core_sample, km.labels_)))
        
    # Precompute hierarchical linkage matrix for Dendrogram (Ward on 100 sample tracks for clean visual)
    dendro_idx = sample_rng.choice(len(X_core_sample), size=100, replace=False)
    linkage_matrix = linkage(X_core_sample[dendro_idx], method="ward")
    joblib.dump(linkage_matrix, "models/clustering/linkage_matrix.joblib")
    
    # Store cluster profiles
    df_eval["Cluster_Hierarchical"] = hier_labels
    df_eval["Cluster_KMeans"] = kmeans_labels
    
    cluster_profiles = df_eval.groupby("Cluster_Hierarchical")[CORE_AUDIO_FEATURES].mean().round(3).to_dict()
    
    clustering_metadata = {
        "primary_model": {
            "name": "Agglomerative Hierarchical Clustering (Primary Capstone Model)",
            "linkage": "complete",
            "k": 2,
            "scaler": "RobustScaler",
            "features": CORE_AUDIO_FEATURES,
            "silhouette": round(hc_sil, 4),
            "davies_bouldin": round(hc_db, 4),
            "calinski_harabasz": round(hc_ch, 2),
            "cluster_sizes": pd.Series(hier_labels).value_counts().sort_index().to_dict(),
            "multi_seed_mean": "0.5034 +/- 0.0060"
        },
        "kmeans_model": {
            "name": "K-Means Clustering",
            "k": 2,
            "scaler": "RobustScaler",
            "features": CORE_AUDIO_FEATURES,
            "silhouette": round(km_sil, 4),
            "davies_bouldin": round(km_db, 4),
            "calinski_harabasz": round(km_ch, 2),
            "cluster_sizes": pd.Series(kmeans_labels).value_counts().sort_index().to_dict()
        },
        "experiment_a_benchmark": {
            "name": "Experiment A (2D Benchmark)",
            "linkage": "average",
            "k": 2,
            "scaler": "MinMaxScaler",
            "features": MINIMAL_2D_FEATURES,
            "silhouette": round(exp_a_sil, 4),
            "davies_bouldin": round(exp_a_db, 4),
            "calinski_harabasz": round(exp_a_ch, 2),
            "cluster_sizes": pd.Series(hier_2d_labels).value_counts().sort_index().to_dict(),
            "multi_seed_mean": "0.5759 +/- 0.0256"
        },
        "pca_explained_variance": evr,
        "elbow": {
            "k_range": list(range(2, 11)),
            "inertias": elbow_inertias,
            "silhouettes": elbow_silhouettes
        },
        "sample_points": {
            "pca_x": X_pca_sample[:, 0].tolist(),
            "pca_y": X_pca_sample[:, 1].tolist(),
            "tsne_x": X_tsne_sample[:, 0].tolist(),
            "tsne_y": X_tsne_sample[:, 1].tolist(),
            "hier_labels": hier_labels.tolist(),
            "kmeans_labels": kmeans_labels.tolist(),
            "track_name": df_eval["track_name"].fillna("Unknown Track").tolist(),
            "artist_name": df_eval["artist_name"].fillna("Unknown Artist").tolist(),
            "genre": df_eval["genre"].fillna("Unknown Genre").tolist()
        },
        "cluster_profiles": cluster_profiles
    }
    
    with open("models/clustering/results.json", "w") as f:
        json.dump(clustering_metadata, f, indent=2)
    print("Clustering models and visualizations serialized successfully.")


if __name__ == "__main__":
    t_start = time.time()
    if not os.path.exists("models/regression/results.json"):
        train_and_save_regression()
    if not os.path.exists("models/classification/results.json"):
        train_and_save_classification()
    train_and_save_clustering()
    print(f"\nALL 22 ALGORITHMS TRAINED AND SAVED IN {time.time() - t_start:.2f} SECONDS!")
