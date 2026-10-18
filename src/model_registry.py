import os
import json
import joblib
import streamlit as st

def _resolve_model_path(rel_path):
    candidates = [
        rel_path,
        os.path.join(os.path.dirname(__file__), "..", rel_path),
        os.path.abspath(rel_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return rel_path

# =====================================================================
# 1. REGRESSION ALGORITHMS (10 MODELS)
# =====================================================================
REGRESSION_ALGORITHMS = {
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest Regressor",
        "rank": 1,
        "family": "Ensemble (Bagging)",
        "desc": "An ensemble of decorrelated decision trees that bootstrap subsets of records and features, averaging outputs to achieve superior generalization on complex nonlinear bike demand.",
        "math": r"$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$",
        "hyperparameters": "n_estimators=100, max_depth=16, min_samples_split=4, random_state=42",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour",
        "best_model": True
    },
    "svr": {
        "id": "svr",
        "name": "Support Vector Regressor (SVR)",
        "rank": 2,
        "family": "Kernel Methods",
        "desc": "Projects scaled temporal and weather features into an infinite-dimensional reproducing kernel Hilbert space using the Radial Basis Function (RBF) kernel with an epsilon-insensitive tube.",
        "math": r"$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum (\xi_i + \xi_i^*), \quad \text{s.t. } |y_i - (w^T \phi(x_i) + b)| \le \epsilon + \xi_i$$",
        "hyperparameters": "C=100, epsilon=0.1, kernel='rbf'",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "polynomial_regression": {
        "id": "polynomial_regression",
        "name": "Polynomial Regression (Degree 2)",
        "rank": 3,
        "family": "Nonlinear Polynomial",
        "desc": "Expands continuous weather variables into second-degree terms and interaction products (e.g. temp x humidity), followed by L2-regularized linear estimation.",
        "math": r"$$\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2], \quad \hat{y} = \phi(x)^T w$$",
        "hyperparameters": "degree=2, include_bias=False, alpha=10.0",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "gradient_boosting": {
        "id": "gradient_boosting",
        "name": "Gradient Boosting Regressor",
        "rank": 4,
        "family": "Ensemble (Boosting)",
        "desc": "Builds shallow decision trees sequentially in a stage-wise greedy manner, fitting each subsequent tree to the negative gradient (pseudo-residuals) of the squared loss.",
        "math": r"$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$",
        "hyperparameters": "n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "decision_tree": {
        "id": "decision_tree",
        "name": "Decision Tree Regressor",
        "rank": 5,
        "family": "Tree-based",
        "desc": "Partitions feature space recursively by greedily selecting feature thresholds that minimize mean squared error within each resulting terminal leaf.",
        "math": r"$$\text{MSE} = \frac{1}{|R|} \sum_{i \in R} (y_i - \bar{y}_R)^2$$",
        "hyperparameters": "max_depth=12, min_samples_split=10, min_samples_leaf=4, random_state=42",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "linear_regression": {
        "id": "linear_regression",
        "name": "Linear Regression (OLS)",
        "rank": 6,
        "family": "Linear Baseline",
        "desc": "Estimates regression weights by minimizing the residual sum of squares via Ordinary Least Squares (OLS) closed-form normal equations.",
        "math": r"$$\hat{\beta} = (X^T X)^{-1} X^T y$$",
        "hyperparameters": "fit_intercept=True, n_jobs=-1",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "ridge_regression": {
        "id": "ridge_regression",
        "name": "Ridge Regression (L2)",
        "rank": 7,
        "family": "Regularized Linear",
        "desc": "Applies L2 penalty on weight magnitudes, shrinking correlated indicator coefficients and mitigating multicollinearity among seasonal variables.",
        "math": r"$$\mathcal{L} = \|y - Xw\|^2 + \alpha \|w\|_2^2$$",
        "hyperparameters": "alpha=10.0, random_state=42",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "lasso_regression": {
        "id": "lasso_regression",
        "name": "Lasso Regression (L1)",
        "rank": 8,
        "family": "Regularized Linear",
        "desc": "Applies L1 penalty on feature weights, enforcing geometric sparsity and automatically performing feature selection by driving minor coefficients to zero.",
        "math": r"$$\mathcal{L} = \frac{1}{2N} \|y - Xw\|^2 + \alpha \|w\|_1$$",
        "hyperparameters": "alpha=0.1, max_iter=10000, random_state=42",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "knn_regressor": {
        "id": "knn_regressor",
        "name": "K-Nearest Neighbors Regressor",
        "rank": 9,
        "family": "Instance-based",
        "desc": "Predicts rental demand by identifying the $k$ most geometrically proximate historical hours in standardized Euclidean space and taking a distance-weighted average.",
        "math": r"$$\hat{y}(x) = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i}, \quad w_i = \frac{1}{d(x, x_i)}$$",
        "hyperparameters": "n_neighbors=11, weights='distance', n_jobs=-1",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    },
    "elasticnet_regression": {
        "id": "elasticnet_regression",
        "name": "ElasticNet Regression",
        "rank": 10,
        "family": "Regularized Linear",
        "desc": "Blends L1 and L2 regularizations via convex combination, overcoming Lasso limitations when dealing with groups of highly correlated features.",
        "math": r"$$\mathcal{L} = \frac{1}{2N} \|y - Xw\|^2 + \alpha \rho \|w\|_1 + \frac{\alpha(1-\rho)}{2} \|w\|_2^2$$",
        "hyperparameters": "alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42",
        "dataset": "UCI Bike Sharing Dataset (Hourly)",
        "target": "cnt (Total Bike Rentals per hour)",
        "units": "bikes / hour"
    }
}

# =====================================================================
# 2. CLASSIFICATION ALGORITHMS (10 MODELS)
# =====================================================================
CLASSIFICATION_ALGORITHMS = {
    "random_forest": {
        "id": "random_forest",
        "name": "Random Forest Classifier",
        "rank": 1,
        "track_part": "Part B (Review 2)",
        "family": "Ensemble (Bagging)",
        "desc": "Ensemble of deep randomized decision trees voting on cancellation probability; achieves highest accuracy and AUC by suppressing tree variance.",
        "hyperparameters": "n_estimators=100, max_depth=16, random_state=42, n_jobs=-1",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"],
        "best_model": True
    },
    "bagging": {
        "id": "bagging",
        "name": "Bagging Classifier (Decision Trees)",
        "rank": 2,
        "track_part": "Part B (Review 2)",
        "family": "Ensemble (Bagging)",
        "desc": "Bootstrap aggregation of unpruned decision trees trained on random sample replacements to generate robust ensemble class predictions.",
        "hyperparameters": "estimator=DecisionTreeClassifier(max_depth=10), n_estimators=40, random_state=42",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "gradient_boosting": {
        "id": "gradient_boosting",
        "name": "Gradient Boosting Classifier",
        "rank": 3,
        "track_part": "Part B (Review 2)",
        "family": "Ensemble (Boosting)",
        "desc": "Sequentially optimizes logistic loss by fitting new shallow trees to negative gradient residuals of prior booking predictions.",
        "hyperparameters": "n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "decision_tree": {
        "id": "decision_tree",
        "name": "Decision Tree Classifier",
        "rank": 4,
        "track_part": "Part A (Review 1)",
        "family": "Tree-based",
        "desc": "Recursive binary splitting using Gini impurity criterion; captures crucial interactions between lead time and room price.",
        "hyperparameters": "max_depth=12, min_samples_split=10, random_state=42",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "mlp_classifier": {
        "id": "mlp_classifier",
        "name": "MLP Classifier (Neural Network)",
        "rank": 5,
        "track_part": "Part B (Review 2)",
        "family": "Deep Learning / Neural Net",
        "desc": "Feedforward artificial neural network with two hidden layers (64, 32) and ReLU activation functions optimized via Adam backpropagation.",
        "hyperparameters": "hidden_layer_sizes=(64, 32), activation='relu', max_iter=250, random_state=42",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "knn_classifier": {
        "id": "knn_classifier",
        "name": "K-Nearest Neighbors (KNN)",
        "rank": 6,
        "track_part": "Part A (Review 1)",
        "family": "Instance-based",
        "desc": "Assigns cancellation status based on the weighted majority vote of the 7 nearest bookings in standardized Euclidean space.",
        "hyperparameters": "n_neighbors=7, weights='distance', n_jobs=-1",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "svc": {
        "id": "svc",
        "name": "Support Vector Classifier (SVC)",
        "rank": 7,
        "track_part": "Part A (Review 1)",
        "family": "Kernel Methods",
        "desc": "Constructs an optimal soft-margin separating hyperplane in non-linear RBF kernel feature space with calibrated Platt probability estimation.",
        "hyperparameters": "C=10, gamma='scale', probability=True, random_state=42",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "adaboost": {
        "id": "adaboost",
        "name": "AdaBoost Classifier",
        "rank": 8,
        "track_part": "Part B (Review 2)",
        "family": "Ensemble (Boosting)",
        "desc": "Adaptive boosting over decision stumps, dynamically reweighting difficult-to-classify reservations across 100 sequential rounds.",
        "hyperparameters": "n_estimators=100, learning_rate=0.5, random_state=42",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "Logistic Regression",
        "rank": 9,
        "track_part": "Part A (Review 1)",
        "family": "Generalized Linear Model",
        "desc": "Baseline linear classifier estimating the log-odds of booking cancellation via standard maximum likelihood estimation.",
        "hyperparameters": "max_iter=1000, random_state=42",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    },
    "gaussian_naive_bayes": {
        "id": "gaussian_naive_bayes",
        "name": "Gaussian Naive Bayes",
        "rank": 10,
        "track_part": "Part A (Review 1)",
        "family": "Probabilistic",
        "desc": "Applies Bayes Theorem under the assumption that all booking features are conditionally independent given the status label.",
        "hyperparameters": "var_smoothing=1e-9",
        "dataset": "INN Hotels Group Dataset (36,275 Bookings)",
        "target": "booking_status (0: Not_Canceled, 1: Canceled)",
        "classes": ["Not_Canceled (0)", "Canceled (1)"]
    }
}

# =====================================================================
# 3. CLUSTERING ALGORITHMS (2 MANDATORY + BENCHMARK)
# =====================================================================
CLUSTERING_ALGORITHMS = {
    "hierarchical_complete": {
        "id": "hierarchical_complete",
        "name": "Agglomerative Hierarchical Clustering (Primary Capstone Model)",
        "algorithm": "Agglomerative Hierarchical",
        "linkage": "Complete",
        "k": 2,
        "scaler": "RobustScaler",
        "feature_set": "Core 9 Audio Perceptual Features",
        "desc": "Bottom-up agglomerative clustering using maximum pairwise diameter (Complete Linkage) on RobustScaler-transformed perceptual audio space. Preserves all 9 perceptual dimensions.",
        "dataset": "Spotify Tracks Audio Dataset (50,000 Tracks)",
        "role": "Primary Capstone Model (Doubles baseline silhouette to 0.5026, lowest score dispersion +/-0.0060)"
    },
    "kmeans": {
        "id": "kmeans",
        "name": "K-Means Clustering",
        "algorithm": "K-Means",
        "linkage": "N/A (Centroid-based)",
        "k": 2,
        "scaler": "RobustScaler / StandardScaler",
        "feature_set": "Core 9 Audio Perceptual Features",
        "desc": "Partitioning algorithm iteratively minimizing within-cluster sum of squares (WCSS / Inertia) to find spherical cluster centroids.",
        "dataset": "Spotify Tracks Audio Dataset (50,000 Tracks)",
        "role": "Baseline Partitioning Model"
    },
    "experiment_a_benchmark": {
        "id": "experiment_a_benchmark",
        "name": "2D High-Separation Benchmark (Experiment A)",
        "algorithm": "Agglomerative Hierarchical",
        "linkage": "Average",
        "k": 2,
        "scaler": "MinMaxScaler",
        "feature_set": "2D (Loudness, Acousticness)",
        "desc": "Ablation benchmark isolating the antagonistic audio axis (electric volume vs acoustic resonance). Achieved 0.6054 on sample (0.5759 +/- 0.0256 multi-seed mean).",
        "dataset": "Spotify Tracks Audio Dataset (50,000 Tracks)",
        "role": "Ablation Study (Demonstrates dimensionality vs. silhouette trade-off)"
    }
}

# =====================================================================
# LOADERS & CACHES
# =====================================================================
@st.cache_data(show_spinner=False)
def load_regression_results():
    path = _resolve_model_path("models/regression/results.json")
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_classification_results():
    path = _resolve_model_path("models/classification/results.json")
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_clustering_results():
    path = _resolve_model_path("models/clustering/results.json")
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_regression_model(model_id):
    path = _resolve_model_path(f"models/regression/{model_id}.joblib")
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_regression_metadata():
    path = _resolve_model_path("models/regression/metadata.joblib")
    poly_path = _resolve_model_path("models/regression/poly_transformer.joblib")
    meta = joblib.load(path)
    if os.path.exists(poly_path):
        meta["poly_transformer"] = joblib.load(poly_path)
    return meta

@st.cache_resource(show_spinner=False)
def load_classification_model(model_id):
    path = _resolve_model_path(f"models/classification/{model_id}.joblib")
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_classification_metadata():
    path = _resolve_model_path("models/classification/metadata.joblib")
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_clustering_artifacts():
    return {
        "kmeans": joblib.load(_resolve_model_path("models/clustering/kmeans_model.joblib")),
        "scaler_core": joblib.load(_resolve_model_path("models/clustering/scaler_core.joblib")),
        "scaler_2d": joblib.load(_resolve_model_path("models/clustering/scaler_2d.joblib")),
        "pca": joblib.load(_resolve_model_path("models/clustering/pca_model.joblib")),
        "linkage_matrix": joblib.load(_resolve_model_path("models/clustering/linkage_matrix.joblib"))
    }
