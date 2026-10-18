import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

# =====================================================================
# 1. OUTLIER HANDLING TRANSFORMER
# =====================================================================
class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Applies IQR-based capping using pre-calculated training bounds to prevent data leakage.
    """
    def __init__(self, bounds=None):
        self.bounds = bounds or {}

    def fit(self, X, y=None):
        # Bounds are fitted on training data only
        if not self.bounds and isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=np.number).columns:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.bounds[col] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, (lower, upper) in self.bounds.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df


# =====================================================================
# 2. REGRESSION PREPROCESSING (BIKE SHARING)
# =====================================================================
REG_CATEGORICAL = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"
]
REG_NUMERICAL = ["temp", "atemp", "hum", "windspeed", "day"]
REG_OUTLIER_COLS = ["temp", "atemp", "hum", "windspeed"]

def engineer_regression_features(df_raw):
    """
    Extracts 'day' from 'dteday' and removes target-leakage columns.
    """
    df = df_raw.copy()
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"])
        df["day"] = df["dteday"].dt.day
    
    # Drop leak columns
    leak_cols = ["instant", "dteday", "casual", "registered"]
    cols_to_drop = [c for c in leak_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df

def get_regression_preprocessor(iqr_bounds=None):
    """
    Builds the ColumnTransformer for regression matching Notebooks/Regression.ipynb.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, REG_NUMERICAL),
            ("cat", cat_pipeline, REG_CATEGORICAL)
        ]
    )
    return preprocessor


# =====================================================================
# 3. CLASSIFICATION PREPROCESSING (HOTEL BOOKING)
# =====================================================================
CLASS_CONTINUOUS_OUTLIER_COLS = ["lead_time", "avg_price_per_room"]

def engineer_classification_features(df_raw):
    """
    Engineers the exact 7 features from Notebooks/Classification.ipynb:
    - total_nights
    - total_guests
    - total_previous_bookings
    - is_family
    - price_per_guest
    - arrival_month_sin
    - arrival_month_cos
    """
    data = df_raw.copy()
    
    # Total nights
    if "no_of_weekend_nights" in data.columns and "no_of_week_nights" in data.columns:
        data["total_nights"] = data["no_of_weekend_nights"] + data["no_of_week_nights"]
    
    # Total guests
    if "no_of_adults" in data.columns and "no_of_children" in data.columns:
        data["total_guests"] = data["no_of_adults"] + data["no_of_children"]
    
    # Total previous bookings
    if "no_of_previous_cancellations" in data.columns and "no_of_previous_bookings_not_canceled" in data.columns:
        data["total_previous_bookings"] = (
            data["no_of_previous_cancellations"] + data["no_of_previous_bookings_not_canceled"]
        )
    
    # Is family
    if "no_of_children" in data.columns:
        data["is_family"] = (data["no_of_children"] > 0).astype(int)
    
    # Price per guest
    if "avg_price_per_room" in data.columns and "total_guests" in data.columns:
        data["price_per_guest"] = data["avg_price_per_room"] / data["total_guests"].replace(0, np.nan)
        data["price_per_guest"] = data["price_per_guest"].fillna(data["avg_price_per_room"])
    
    # Arrival month cyclics
    if "arrival_month" in data.columns:
        data["arrival_month_sin"] = np.sin(2 * np.pi * data["arrival_month"] / 12)
        data["arrival_month_cos"] = np.cos(2 * np.pi * data["arrival_month"] / 12)
        
    # Drop identifiers and target if present in raw
    if "Booking_ID" in data.columns:
        data = data.drop(columns=["Booking_ID"])
        
    return data

def get_classification_preprocessor(numeric_cols, categorical_cols):
    """
    Builds the ColumnTransformer for classification matching Notebooks/Classification.ipynb.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

# =====================================================================
# 4. CLUSTERING PREPROCESSING (SPOTIFY TRACKS)
# =====================================================================
CORE_AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
MINIMAL_2D_FEATURES = ["loudness", "acousticness"]
