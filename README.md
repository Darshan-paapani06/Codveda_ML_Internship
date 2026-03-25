#  Codveda ML Internship — Level 1 (Basic)

<div align="center">

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Python-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?style=for-the-badge&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-green?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Level%201-Completed-success?style=for-the-badge)

**Internship at [Codveda Technologies](https://www.codveda.com)**
**Intern : Darshan Gowda T S**
**Duration : 20 March 2026 – 20 April 2026**

</div>

---

## About This Repository

This repository contains all the work completed during my Machine Learning Internship at Codveda Technologies. Each level covers a set of ML tasks with increasing complexity — from basic preprocessing all the way to neural networks.

This README covers **Level 1 — Basic**, which focuses on the foundational building blocks of any machine learning pipeline.

---

## Level 1 — Task Overview

| Task | Topic | Dataset | Key Result |
|------|-------|---------|-----------|
| Task 1 | Data Preprocessing | Iris + Churn | Both datasets cleaned and saved |
| Task 2 | Linear Regression | Boston Housing | R² = 0.67, RMSE = $4.93K |
| Task 3 | KNN Classifier | Iris | Accuracy = 96.7% |

---

## Task 1 — Data Preprocessing for Machine Learning

### What was done
Before any model can be trained, raw data needs to be cleaned and prepared. This task covers the complete preprocessing pipeline applied to two real-world datasets.

**Datasets used:**
- `1__iris.csv` — 150 samples, 4 features, 3 flower species
- `churn-bigml-80.csv` — 2666 customers, 20 features, binary churn label

### Techniques Applied

**Missing Value Handling**
- Mean Imputation — fills missing values with the column average
- Drop Strategy — removes rows that contain missing values

**Categorical Encoding**
- Label Encoding — converts categories to integers (0, 1, 2)
- One-Hot Encoding — creates a separate binary column per category
- Manual Mapping — Yes/No columns mapped to 1/0

**Feature Scaling**
- StandardScaler — transforms features to mean=0, std=1
- MinMaxScaler — scales all values to the range [0, 1]

**Train / Test Split**
- 80% training / 20% testing
- `stratify=y` used to maintain class balance in both sets

### Output Files
```
outputs/
├── iris_preprocessed.csv
└── churn_preprocessed.csv
```

### How to Run
```bash
python scripts/L1_T1_Data_Preprocessing.py
```

---

## Task 2 — Linear Regression for House Price Prediction

### What was done
Built a Linear Regression model to predict median house prices using the Boston Housing dataset. The model was trained, evaluated and interpreted fully.

**Dataset used:**
- `4__house_Prediction_Data_Set.csv` — 506 houses, 13 features

### Feature Description

| Feature | Description |
|---------|-------------|
| CRIM | Per capita crime rate |
| RM | Average number of rooms |
| LSTAT | % lower income population |
| PTRATIO | Pupil-teacher ratio |
| NOX | Nitric oxide concentration |
| MEDV | Median house value in $1000s ← Target |

### Model Performance

| Metric | Value |
|--------|-------|
| R² Score (Train) | 0.7509 |
| R² Score (Test) | 0.6688 |
| RMSE | $4.93K |
| MAE | $3.19K |

### Key Findings from Coefficients

| Feature | Coefficient | Effect |
|---------|------------|--------|
| RM | +3.15 | More rooms → higher price |
| LSTAT | -3.61 | More poverty → lower price |
| PTRATIO | -2.04 | Worse schools → lower price |
| DIS | -3.08 | Far from employment → lower price |

### Output Files
```
outputs/
├── L1_T2_actual_vs_predicted.png
├── L1_T2_feature_coefficients.png
├── L1_T2_error_distribution.png
└── L1_T2_predictions.csv
```

### Plots Generated

**Actual vs Predicted**
Shows how close the model predictions are to real house prices. The closer the dots are to the red diagonal line, the better the prediction.

**Feature Coefficients**
Horizontal bar chart showing which features increase (blue) or decrease (red) the house price and by how much.

**Error Distribution**
Histogram of prediction errors — ideally centered around zero, confirming the model is unbiased.

### How to Run
```bash
python scripts/L1_T2_Linear_Regression.py
```

---

## Task 3 — K-Nearest Neighbors (KNN) Classifier

### What was done
Built a KNN classifier to classify iris flowers into 3 species. Tested 8 different values of K and selected the best one based on accuracy.

**Dataset used:**
- `1__iris.csv` — 150 samples, 4 features, 3 species (Setosa, Versicolor, Virginica)

### K Value Comparison

| K | Accuracy | F1 Score |
|---|----------|----------|
| K=1 | 0.9667 | 0.9666 |
| K=3 | 0.9333 | 0.9327 |
| K=5 | 0.9333 | 0.9327 |
| K=7 | 0.9667 | 0.9666 |
| K=9 | 0.9667 | 0.9666 |
| K=11 | 0.9667 | 0.9666 |
| K=13 | 0.9667 | 0.9666 |
| **K=15** | **0.9667** | **0.9666** |

**Best K = 1 with Accuracy = 96.67%**

### Final Model Performance (K=1)

| Metric | Value |
|--------|-------|
| Accuracy | 96.67% |
| Precision | 0.9697 |
| Recall | 0.9667 |
| F1 Score | 0.9666 |

### Confusion Matrix

| | Predicted Setosa | Predicted Versicolor | Predicted Virginica |
|--|--|--|--|
| **Actual Setosa** | 10 | 0 | 0 |
| **Actual Versicolor** | 0 | 10 | 0 |
| **Actual Virginica** | 0 | 1 | 9 |

Only 1 flower was misclassified out of 30 test samples.

### Key Insights
- Feature scaling is critical for KNN since it uses Euclidean distance
- Small K can overfit, large K can underfit — always compare
- Iris dataset is nearly linearly separable which explains the high accuracy

### Output Files
```
outputs/
├── L1_T3_knn_performance.png
├── L1_T3_decision_boundary.png
├── L1_T3_k_comparison.png
└── L1_T3_k_comparison.csv
```

### How to Run
```bash
python scripts/L1_T3_KNN_Classifier.py
```

---

## Project Structure

```
Codveda-ML-Internship/
│
├── README.md
│
├── Level_1_Basic/
│   ├── scripts/
│   │   ├── L1_T1_Data_Preprocessing.py
│   │   ├── L1_T2_Linear_Regression.py
│   │   └── L1_T3_KNN_Classifier.py
│   └── outputs/
│       ├── iris_preprocessed.csv
│       ├── churn_preprocessed.csv
│       ├── L1_T2_actual_vs_predicted.png
│       ├── L1_T2_feature_coefficients.png
│       ├── L1_T2_error_distribution.png
│       ├── L1_T3_knn_performance.png
│       ├── L1_T3_decision_boundary.png
│       └── L1_T3_k_comparison.png
│
├── Level_2_Intermediate/        ← In progress
│   └── coming soon...
│
└── Level_3_Advanced/            ← Upcoming
    └── coming soon...
```

---

## Tools and Libraries

```python
Python        3.12
pandas        2.x
numpy         1.x
scikit-learn  1.x
matplotlib    3.x
```

---

## How to Set Up Locally

```bash
# Clone the repository
git clone https://github.com/YourUsername/Codveda-ML-Internship.git

# Navigate to project
cd Codveda-ML-Internship

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow

# Run any script
python Level_1_Basic/scripts/L1_T1_Data_Preprocessing.py
```

---

## Internship Details

| | |
|--|--|
| Company | Codveda Technologies |
| Position | Machine Learning Intern |
| Mode | Remote |
| Duration | 20 March 2026 – 20 April 2026 |
| LinkedIn | [Darshan Gowda T S](#) |

---

<div align="center">

Made with genuine effort during my ML Internship at Codveda Technologies

</div>
