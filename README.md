
# Machine Learning Capstone Project

## 23CSE301 — Machine Learning

## Team - 17

This repository contains our Machine Learning Capstone Project, covering
three problem tracks: Regression, Classification, and Clustering.

The project follows the complete machine learning workflow:

**Data Loading → Exploratory Data Analysis → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Result Comparison**

---

## Project Overview

| Track | Status | Problem |
|---|---|---|
| Regression | Completed | Bike Sharing Demand Prediction |
| Classification | Completed | Hotel Booking Cancellation Prediction |
| Clustering | Planned | To be added |

---

## 1. Regression Track

### Problem Statement

The objective of this regression task is to predict the **total number of bike rentals (`cnt`) per hour** based on temporal, seasonal, weather, and environmental factors.

### Dataset

- **Dataset:** UCI Bike Sharing Dataset — Hourly
- **Target:** `cnt`
- **Prediction Type:** Continuous numerical prediction
- **Task:** Predict hourly bike rental demand

### Exploratory Data Analysis

The dataset was examined through:

- Dataset shape and data types
- Missing-value analysis
- Duplicate detection
- Target distribution
- Numerical feature distributions
- Categorical feature distributions
- Correlation heatmap
- Feature-target scatter plots
- Boxplots and outlier analysis

### Data Preprocessing

The preprocessing pipeline included:

- Missing-value checking
- Duplicate checking
- IQR-based outlier treatment
- Feature engineering
- Categorical feature encoding
- Numerical feature scaling where required
- Train-test splitting
- Training-only fitting of preprocessing transformations

### Regression Results

| Rank | Model | R² | RMSE | MAE |
|---|---|---:|---:|---:|
| 1 | Random Forest | 0.9303 | 46.996 | 29.47 |
| 2 | SVR | 0.9255 | 48.58 | 29.83 |
| 3 | Decision Tree | 0.8760 | 62.67 | 37.60 |
| 4 | Gradient Boosting | 0.8682 | 64.61 | 45.86 |
| 5 | Polynomial Regression | 0.6992 | 97.594 | 72.414 |
| 6 | Lasso Regression | 0.6817 | 100.396 | 74.058 |
| 7 | ElasticNet Regression | 0.6816 | 100.4138 | 74.0587 |
| 8 | Ridge Regression | 0.6816 | 100.4176 | 74.0871 |
| 9 | Linear Regression | 0.6814 | 100.4387 | 74.0996 |
| 10 | KNN Regression | 0.5676 | 117.0197 | 86.1087 |

### Hyperparameter Tuning

Hyperparameter tuning was performed using cross-validation.

The following search techniques were used:

- GridSearchCV
- RandomizedSearchCV

The hyperparameters investigated included:

- Random Forest: `n_estimators`
- SVR: `C`, `kernel`
- Decision Tree: `max_depth`
- Gradient Boosting: `learning_rate`
- KNN: `n_neighbors`

### Regression Analysis

The evaluated models showed different levels of performance on the Bike Sharing dataset. Tree-based and nonlinear models captured relationships in the data more effectively than the linear models evaluated in this experiment.

Random Forest achieved the highest test-set R² among the evaluated models, with an R² of approximately 0.93 on the common test split.

---

## 2. Classification Track

## Hotel Booking Cancellation Prediction

### Problem Statement

The objective of this classification task is to predict whether a hotel booking will be **`Canceled` or `Not_Canceled`** based on booking-related information.

### Dataset

- **Rows:** 36,275
- **Original features:** 19
- **Target:** `booking_status`
- **Classes:**
  - `Not_Canceled`
  - `Canceled`

### Exploratory Data Analysis

The dataset was examined through:

- Dataset shape and data types
- Missing-value analysis
- Duplicate detection
- Target-class distribution
- Numerical feature distributions
- Categorical feature distributions
- Correlation heatmap
- Feature-target scatter plots
- Boxplots and outlier analysis

### Data Preprocessing

The preprocessing pipeline included:

- Missing-value checking
- Duplicate checking
- Feature engineering
- Categorical feature encoding
- Numerical feature scaling where required
- Stratified 80:20 train-test split
- Training-only fitting of preprocessing transformations

### Engineered Features

- `total_nights`
- `total_guests`
- `total_previous_bookings`
- `is_family`
- `price_per_guest`
- `arrival_month_sin`
- `arrival_month_cos`

---

## Classification — Part A

Five classification algorithms were implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Decision Tree Classifier
5. Support Vector Classifier (SVC)

### Results

| Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 81.28% | 75.19% | 63.99% | 80.86% | 87.16% |
| KNN | 85.20% | 79.87% | 73.29% | 85.02% | 91.00% |
| Gaussian Naive Bayes | 40.83% | 35.22% | 96.05% | 33.04% | 78.81% |
| Decision Tree | 88.04% | 82.17% | 81.07% | 88.01% | 91.54% |
| SVC | 86.38% | 81.53% | 75.56% | 86.24% | 92.43% |

---

## Classification — Part B

Five additional classification algorithms were implemented using the same dataset and train-test split.

### Algorithms Implemented

1. Random Forest Classifier
2. AdaBoost Classifier
3. Gradient Boosting Classifier
4. Bagging Classifier
5. Multi-Layer Perceptron (MLP) Classifier

### Model Training and Evaluation

Each algorithm was trained and evaluated using the same classification task. The workflow included:

- Baseline model training
- Prediction on the test set
- Confusion matrix analysis
- Accuracy, precision, recall, and weighted F1-score calculation
- ROC-AUC evaluation
- Hyperparameter tuning
- Evaluation of tuned models
- Comparison of baseline and tuned performance

### Part B Results

| Model | Baseline Accuracy | Tuned Accuracy | Baseline Weighted F1 | Tuned Weighted F1 | Tuned ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 90.17% | 90.05% | 0.9005 | 0.9000 | 0.9575 |
| AdaBoost | 81.67% | 81.92% | 0.8111 | 0.8149 | 0.8775 |
| Gradient Boosting | 86.08% | 86.60% | 0.8582 | 0.8638 | 0.9277 |
| Bagging | 90.02% | 90.24% | 0.8995 | 0.9016 | 0.9555 |
| MLP | 87.20% | 86.81% | 0.8704 | 0.8673 | 0.9340 |

### Hyperparameter Tuning

Hyperparameter tuning was performed to find suitable model configurations.

The parameters investigated included:

- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_leaf`
- **AdaBoost:** `n_estimators`, `learning_rate`
- **Gradient Boosting:** `n_estimators`, `learning_rate`, `max_depth`
- **Bagging:** `n_estimators`, `max_samples`, `max_features`
- **MLP:** `hidden_layer_sizes`, `alpha`, `learning_rate_init`

### Classification Analysis

The results show that the ensemble-based models performed well on the hotel booking cancellation dataset.

- **Bagging** achieved the highest tuned accuracy of 90.24% and weighted F1-score of 0.9016 among the models in the comparison.
- **Random Forest** achieved the highest tuned ROC-AUC of 0.9575.
- **Gradient Boosting** improved its accuracy and weighted F1-score after tuning.
- **AdaBoost** showed a small improvement after tuning.
- **MLP** showed a slight decrease in performance after tuning.

Bagging is a strong choice when prioritizing accuracy and weighted F1-score, while Random Forest is a strong choice when prioritizing ROC-AUC. The final model should also be assessed using cancellation recall, since identifying canceled bookings is an important part of the prediction task.

### Overall Classification Comparison

All ten classification algorithms from Part A and Part B are compared using:

- Accuracy
- Precision
- Recall
- Weighted F1-score
- ROC-AUC

This comparison supports the selection of an appropriate model for predicting hotel booking cancellations.

---

## 3. Clustering Track

To be completed.

Planned work includes:

- K-Means clustering
- Hierarchical clustering
- Silhouette Score evaluation
- Davies-Bouldin Index evaluation
- Calinski-Harabasz Score evaluation
- Elbow curve for K-Means
- Dendrogram for hierarchical clustering
- PCA-based two-dimensional cluster visualization
- t-SNE visualization for at least one clustering algorithm

---

## 4. Repository Structure

```text
Project/
│
├── README.md
├── requirements.txt
│
├── Data/
│   └── INNHotelsGroup.csv
│
├── Notebooks/
|   ├──Regression.ipynb
│   ├── Classification.ipynb
│   └── Clustering.ipynb
│
└── anaconda_projects/
    └── db/
```

*Update the notebook names and folder structure to match the actual files in the repository.*

---

## 5. Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 6. Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Open the relevant Jupyter Notebook and run the cells sequentially.

---

## 7. Project Status

| Component | Status |
|---|---|
| Regression Track | Completed |
| Classification Part A | Completed |
| Classification Part B | Completed |
| Clustering Track | Planned |

---

## 8. Academic Integrity & AI Assistance

Generative AI tools were used as a supporting resource during the development of this project, primarily for code scaffolding, debugging assistance, and understanding implementation approaches.

The team reviewed the code and outputs and validated the model evaluations and interpretations as part of the project workflow. The final analytical decisions and conclusions were reviewed by the team.

AI was used as a development aid to support implementation and learning.

---

## Done By

### Team Members

| S.No. | Name | Register Number |
|---|---|---|
| 1 | Himakesh Puthumbaku | CB.SC.U4CSE24741 |
| 2 | Vaishnavi Tallada | CB.SC.U4CSE24757 |
| 3 | Vuyyala Manesh | CB.SC.U4CSE24759 |

