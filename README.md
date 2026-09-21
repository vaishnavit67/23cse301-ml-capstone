# Machine Learning Capstone Project

## 23CSE301 — Machine Learning

## Team - 17

This repository contains our Machine Learning Capstone Project, covering
three problem tracks: Regression, Classification, and Clustering.

The project follows the complete machine learning workflow:

**Data Loading → Exploratory Data Analysis → Data Cleaning → Feature Engineering → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Result Comparison**

---

## 📌 Project Overview

| Track | Status | Problem |
|---|---|---|
| Regression | Part A and B completed | Bike Sharing Demand Prediction |
| Classification | Part A Completed | Hotel Booking Cancellation Prediction |
| Clustering | Planned | To be added |

---

## 1. Regression Track

## 
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
- IQR-based outlier treatment
- Feature engineering
- Categorical feature encoding
- Numerical feature scaling where required
- Stratified 80:20 train-test split
- Training-only fitting of preprocessing transformations

---

## 2. Classification Track

## Hotel Booking Cancellation Prediction

### Problem Statement

The objective of this classification task is to predict whether a
hotel booking will be **Cancelled** or **Not_Canceled** based on
booking-related information.

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
- IQR-based outlier treatment
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

### Regression Results

| Rank | Model | R² | RMSE | MAE |
|------|-------|----:|-----:|----:|
| 1 | Random Forest | 0.9303 | 46.996 | 29.47 |
| 2 | SVR | 0.9255 | 48.58 | 29.83 |
| 3 | Decision Tree | 0.8760 | 62.67 | 37.60 |
| 4 | Gradient Boosting | 0.8682 | 64.61 | 45.86 |
| 5 | Polynomial Regression | 0.6992 | — | — |
| 6 | Lasso Regression | 0.6817 | — | — |
| 7 | ElasticNet Regression | 0.6816 | — | — |
| 8 | Ridge Regression | 0.6816 | — | — |
| 9 | Linear Regression | 0.6814 | — | — |
| 10 | KNN Regression | 0.5676 | — | — |

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

Random Forest achieved the highest test-set R² among the evaluated models, with an R² of approximately 0.93 on the common test split. Its performance was also supported by its cross-validation results.
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

> **Note:** These results represent Classification **Part A**.
> Classification Part B will be added during the next stage.

---

## 3. Clustering Track

To be completed.

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
│   └── Classification.ipynb
│
└── anaconda_projects/
  └── db/
```

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

## 7. Project Status

| Component | Status |
|---|---|
| Regression Track | To be completed |
| Classification Part A | Completed |
| Classification Part B | To be completed |
| Clustering Track | Planned |

## 8. Academic Integrity & AI Assistance

Generative AI tools were used as a supporting resource during the
development of this project, primarily for code scaffolding,
debugging assistance, and understanding implementation approaches.

However, the data analysis, exploratory observations, feature
engineering decisions, model evaluation, interpretation of results,
and conclusions were carried out and validated by our team.

We did not rely on AI-generated interpretations without verification.
We examined the data, evaluated the models using the required metrics,
interpreted the visualizations and results, and made the final
analytical decisions independently.

AI was therefore used as a development aid, while the analytical
reasoning, interpretation, and conclusions represent our team's work
and understanding.

## Done By
### Team Members

| S.No. | Name | Register Number |
|---|---|---|
| 1 | Himakesh Puthumbaku | CB.SC.U4CSE24741 |
| 2 | Vaishnavi Tallada | CB.SC.U4CSE24757 |
| 3 | Vuyyala Manesh | CB.SC.U4CSE24759 |

