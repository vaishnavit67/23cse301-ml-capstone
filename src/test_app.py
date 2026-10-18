import sys
import os

sys.path.insert(0, ".")
sys.path.insert(0, "src")
sys.path.insert(0, "pages")

print("--- TESTING MODULE IMPORTS ---")
import preprocessing
import model_registry
import data_loader
import prediction_service
import visualization
import home
import regression
import classification
import clustering
import model_comparison
import datasets_methodology
import about
print("All modules imported successfully!")

print("\n--- TESTING REGRESSION PREDICTIONS (ALL 10 ALGORITHMS) ---")
sample_reg_input = {
    "season": 3, "yr": 1, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 2,
    "workingday": 1, "weathersit": 1, "temp": 0.68, "atemp": 0.65, "hum": 0.45,
    "windspeed": 0.18, "day": 15
}
for model_key in model_registry.REGRESSION_ALGORITHMS:
    res = prediction_service.predict_regression(model_key, sample_reg_input)
    print(f"  {model_key:23s} -> {res['prediction']} {res['units']} ({res['status']})")

print("\n--- TESTING CLASSIFICATION PREDICTIONS (ALL 10 ALGORITHMS) ---")
sample_class_input = {
    "no_of_adults": 2, "no_of_children": 0, "no_of_weekend_nights": 1, "no_of_week_nights": 2,
    "type_of_meal_plan": "Meal Plan 1", "required_car_parking_space": 0, "room_type_reserved": "Room_Type 1",
    "lead_time": 180, "arrival_year": 2018, "arrival_month": 10, "arrival_date": 18,
    "market_segment_type": "Online", "repeated_guest": 0, "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0, "avg_price_per_room": 125.0, "no_of_special_requests": 0
}
for model_key in model_registry.CLASSIFICATION_ALGORITHMS:
    res = prediction_service.predict_classification(model_key, sample_class_input)
    print(f"  {model_key:23s} -> {res['predicted_label']} (P(Cancel)={res['prob_canceled']}%, {res['risk_tier']})")

print("\n--- TESTING CLUSTERING PREDICTION ---")
sample_audio_input = {
    "danceability": 0.72, "energy": 0.81, "loudness": -5.4, "speechiness": 0.06,
    "acousticness": 0.08, "instrumentalness": 0.02, "liveness": 0.15, "valence": 0.64, "tempo": 124.0
}
clust_res = prediction_service.predict_clustering(sample_audio_input)
print(f"  Clustering -> Cluster {clust_res['cluster_id']}: {clust_res['archetype_name']}")
print(f"  PCA Coordinates -> ({clust_res['pca_x']}, {clust_res['pca_y']})")

print("\nALL 22 PREDICTION FLOWS VERIFIED SUCCESSFULLY WITH ZERO ERRORS!")
