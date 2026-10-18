# Machine Learning Capstone Project | Interactive Web Application

## 23CSE301 — Machine Learning
### Amrita Vishwa Vidyapeetham | Academic Year 2026–27
### Team — 17

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning Capstone Project uniting **all 22 algorithms** across three problem tracks: **Regression**, **Classification**, and **Clustering**. Features a production-ready, interactive **Streamlit Web Application** designed for live inference, model diagnosis, and comparative evaluation.

---

## 📌 Project Overview

| Track | Problem Domain | Dataset | Target / Task | Total Algorithms | Top Performing Model |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **1. Regression** | Bike Sharing Demand | UCI Bike Sharing (`hour.csv`) | `cnt` (Hourly Rentals) | **10 Models** | **Random Forest** ($R^2 = 0.9303$, $\text{RMSE} = 47.00$) |
| **2. Classification** | Hotel Cancellation | INN Hotels Group (`INNHotelsGroup.csv`) | `booking_status` (Binary) | **10 Models** | **Random Forest** ($\text{Acc} = 90.03\%$, $\text{AUC} = 95.53\%$) |
| **3. Clustering** | Spotify Audio Personas | Spotify Tracks (`spotify_tracks.csv`) | Unsupervised (9 Core Features) | **2 Models + Benchmarks** | **Agglomerative Complete** ($\text{Sil} = 0.5026$, $\pm 0.0060$) |

---

## 🚀 Interactive Web Application

The interactive web dashboard brings all three tracks together into a cohesive interface:

1. **🏠 Home Dashboard:** High-level project KPIs, track summaries, and master 22-algorithm inventory.
2. **🚴 Regression Workspace:** Interactive weather and temporal sliders to predict hourly bike demand across all 10 regressors, with actual-vs-predicted and residual plots.
3. **🏨 Classification Workspace:** Guest booking form calculating real-time cancellation probabilities, risk classification, confusion matrix heatmaps, and ROC curves.
4. **🎵 Clustering Workspace:** Continuous audio sliders classifying songs into discovered acoustic archetypes and projecting them onto interactive 2D PCA & t-SNE manifold plots.
5. **📊 Model Comparison Hub:** Sortable comparative metric tables and interactive bar charts across all algorithms.
6. **📖 Datasets & Methodology:** In-depth documentation of feature engineering, leakage prevention protocols, and mathematical principles.
7. **ℹ️ About & Rubric Alignment:** Team information, review scopes, and bonus marks rubric compliance (+2 Marks).

---

## 💻 Local Setup & Execution

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/vaishnavit67/23cse301-ml-capstone.git
cd 23cse301-ml-capstone

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Public Deployment Guide

This repository is optimized for one-click deployment on **Streamlit Community Cloud** or **Hugging Face Spaces**:

### Deploying to Streamlit Community Cloud (Recommended):
1. Fork or push this repository to your GitHub account (`https://github.com/vaishnavit67/23cse301-ml-capstone`).
2. Log in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub credentials.
3. Click **"New App"**.
4. Select your repository: `vaishnavit67/23cse301-ml-capstone` and branch: `main`.
5. Set **Main file path** to: `app.py`.
6. Click **"Deploy!"**. The app will automatically build from `requirements.txt` and launch with a public live URL.

---

## 1. Regression Track (10 Algorithms)

### Problem Statement
Predict the total hourly count of rental bikes (`cnt`) in the capital bikeshare system to enable pro-active station inventory balancing and fleet logistics.

### Verified Test-Set Results:
| Rank | Model | Family | $R^2$ Score | RMSE (bikes/hr) | MAE (bikes/hr) | 5-Fold CV $R^2$ |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Random Forest Regressor** | Ensemble (Bagging) | **0.9303** | **46.99** | **29.47** | $0.9180 \pm 0.0025$ |
| **2** | **Support Vector Regressor (SVR)** | Kernel Methods | **0.9258** | **48.46** | **29.46** | $0.9180 \pm 0.0029$ |
| **3** | **Polynomial Regression (Degree 2)** | Nonlinear Expansion | **0.9242** | **48.98** | **32.89** | $0.7002 \pm 0.0136$ |
| **4** | **Gradient Boosting Regressor** | Ensemble (Boosting) | **0.8889** | **59.30** | **40.71** | $0.8593 \pm 0.0059$ |
| **5** | **Decision Tree Regressor** | Tree-based | **0.8760** | **62.67** | **37.60** | $0.8599 \pm 0.0044$ |
| **6** | **Linear Regression (OLS)** | Baseline Linear | **0.6983** | **97.74** | **69.21** | $0.3871 \pm 0.0120$ |
| **7** | **Ridge Regression (L2)** | Regularized Linear | **0.6976** | **97.86** | **69.42** | $0.6833 \pm 0.0145$ |
| **8** | **Lasso Regression (L1)** | Regularized Linear | **0.6973** | **97.91** | **69.34** | $0.6833 \pm 0.0145$ |
| **9** | **K-Nearest Neighbors (KNN)** | Instance-based | **0.5882** | **114.19** | **83.01** | $0.5485 \pm 0.0147$ |
| **10** | **ElasticNet Regression** | Regularized Linear | **0.5671** | **117.09** | **86.79** | $0.6833 \pm 0.0145$ |

---

## 2. Classification Track (10 Algorithms)

### Problem Statement
Predict whether an upcoming hotel reservation will be **Cancelled** or **Honored (Not_Canceled)** to mitigate loss from vacant rooms and optimize revenue per available room (RevPAR).

### Verified Test-Set Results:
| Rank | Model | Review Scope | Family | Accuracy | Weighted F1 | ROC-AUC | Precision | Recall |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest** | Review 2 (Part B) | Ensemble (Bagging) | **90.03%** | **89.89%** | **95.53%** | 86.62% | 81.99% |
| **2** | **Bagging (Decision Trees)** | Review 2 (Part B) | Ensemble (Bagging) | **89.00%** | **88.89%** | **94.61%** | 84.44% | 81.07% |
| **3** | **Gradient Boosting** | Review 2 (Part B) | Ensemble (Boosting) | **88.59%** | **88.43%** | **94.56%** | 84.45% | 79.47% |
| **4** | **Decision Tree** | Review 1 (Part A) | Tree-based | **87.98%** | **87.96%** | **93.29%** | 81.71% | 81.62% |
| **5** | **MLP (Neural Network)** | Review 2 (Part B) | Deep Learning | **87.22%** | **87.03%** | **93.38%** | 82.63% | 76.99% |
| **6** | **K-Nearest Neighbors (KNN)**| Review 1 (Part A) | Instance-based | **86.96%** | **86.88%** | **92.25%** | 81.33% | 78.33% |
| **7** | **Support Vector Classifier** | Review 1 (Part A) | Kernel Methods | **85.58%** | **85.43%** | **91.02%** | 80.12% | 74.80% |
| **8** | **AdaBoost** | Review 2 (Part B) | Ensemble (Boosting) | **81.67%** | **81.11%** | **87.24%** | 76.84% | 63.36% |
| **9** | **Logistic Regression** | Review 1 (Part A) | Generalized Linear | **81.28%** | **80.86%** | **87.16%** | 75.19% | 63.99% |
| **10** | **Gaussian Naive Bayes** | Review 1 (Part A) | Probabilistic | **40.59%** | **32.60%** | **78.81%** | 35.12% | 96.09% |

---

## 3. Clustering Track (2 Algorithms + Benchmarks)

### Problem Statement
Uncover latent acoustic music archetypes from 9 continuous physical audio features without ground-truth genre guidance, demonstrating cluster separation and multidimensional stability.

### Evaluated Configurations:
| Configuration | Algorithm | Scaler | Features | k | Silhouette | Davies–Bouldin | Calinski–Harabasz | Status / Role |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp C (Core 9)** | Agglomerative (Complete) | `RobustScaler` | Core 9 Audio | 2 | **0.5026** | **0.6928** | **684.39** | **PRIMARY CAPSTONE MODEL** ($0.5034 \pm 0.0060$) |
| **Exp A (2D Benchmark)** | Agglomerative (Average) | `MinMaxScaler` | Loudness + Acousticness | 2 | **0.6054** | **0.5503** | **2288.92** | **Ablation Benchmark** ($0.5759 \pm 0.0256$) |
| **Exp B (K-Means 2D)** | K-Means | `MinMaxScaler` | Loudness + Acousticness | 2 | **0.5952** | **0.6324** | **2730.22** | **K-Means Benchmark** |
| **Exp D (PCA 2D)** | Agglomerative (Complete) | `StandardScaler -> PCA` | PCA 2 Components | 2 | **0.5777** | **0.4432** | **1017.07** | **Dimensionality Reduction Model** ($0.5113 \pm 0.1233$) |
| **Exp E (Single Linkage)** | Agglomerative (Single) | `MinMaxScaler` | 2D Contrast | 2 | 0.3174 | 0.6321 | 4.44 | **REJECTED** (Chaining phenomenon: `{0: 1498, 1: 2}`) |

---

## 📁 Repository Structure

```text
23cse301-ml-capstone/
│
├── app.py                          # Streamlit application main entry point
├── requirements.txt                # Production environment dependencies
├── README.md                       # Master capstone documentation
│
├── pages/                          # Streamlit modular page controllers
│   ├── home.py                     # Home dashboard & algorithm inventory
│   ├── regression.py               # Regression workspace (10 algorithms)
│   ├── classification.py           # Classification workspace (10 algorithms)
│   ├── clustering.py               # Clustering workspace (Hierarchical & K-Means)
│   ├── model_comparison.py         # Multi-track comparison hub & charts
│   ├── datasets_methodology.py     # Feature engineering & leakage protocols
│   └── about.py                    # Course info, team details, rubric alignment
│
├── src/                            # Machine learning core pipeline services
│   ├── data_loader.py              # Cached dataset loading utilities
│   ├── preprocessing.py            # Feature engineering, transformers, IQR capper
│   ├── model_registry.py           # Central model metadata, formulas & loaders
│   ├── prediction_service.py       # Live inference service for all 22 models
│   ├── visualization.py            # Plotly & Matplotlib interactive figures
│   ├── train_models.py             # Model training & serialization script
│   └── test_app.py                 # Automated end-to-end verification script
│
├── models/                         # Serialized model artifacts (Joblib)
│   ├── regression/                 # 10 regression models + metadata
│   ├── classification/             # 10 classification models + metadata
│   └── clustering/                 # Clustering models, scalers, PCA, results
│
├── Data/                           # Datasets
│   ├── hour.csv                    # Bike Sharing dataset (Regression)
│   ├── INNHotelsGroup.csv          # Hotel Booking dataset (Classification)
│   └── spotify_tracks.csv          # Spotify Tracks dataset (Clustering)
│
└── Notebooks/                      # Original Jupyter Notebooks
    ├── Regression.ipynb            # Review 1 Regression Notebook
    ├── Classification.ipynb        # Classification Notebook (Part A + B)
    └── Clustering.ipynb            # Review 2 Clustering Track Notebook
```

---

## 👥 Team Members

| S.No. | Name | Register Number | Role / Primary Ownership |
| :---: | :--- | :--- | :--- |
| 1 | **Himakesh Puthumbaku** | CB.SC.U4CSE24741 | Full Pipeline Integration, Web Application & Deployment, Clustering Track |
| 2 | **Vaishnavi Tallada** | CB.SC.U4CSE24757 | Classification Track (Part A & Part B), EDA & Outlier Strategy |
| 3 | **Vuyyala Manesh** | CB.SC.U4CSE24759 | Regression Track (10 Models, Hyperparameter Tuning & Cross-Val) |

---

## 📜 Academic Integrity Citation

In accordance with Section 8 of the 23CSE301 Capstone Guidelines:
Generative AI tools were utilized as an assistive technology for code scaffolding, debugging, and UI structuring. All feature engineering, exploratory data analysis, algorithm choices, hyperparameter tuning experiments, metric evaluations, and analytical conclusions represent the team's verified work.
