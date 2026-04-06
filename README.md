# Codveda ML Internship — Complete Journey

<div align="center">

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Python-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-green?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Internship-Completed-success?style=for-the-badge)

<br/>

**Intern : Darshan Gowda T S**
**Company : [Codveda Technologies](https://www.codveda.com)**
**Duration : 20 March 2026 – 20 April 2026**
**Mode : Remote**

<br/>

> *"From raw data to neural networks — built everything from scratch."*

</div>

---

## What This Repository Contains

This repository is the complete record of my Machine Learning Internship at Codveda Technologies. Over the course of one month, I worked through 9 hands-on ML tasks across 3 levels — starting from data cleaning and ending with a fully trained neural network. Every script here was written, understood and executed by me.

---

## Internship Progress

| Level | Theme | Tasks | Status |
|-------|-------|-------|--------|
| Level 1 | Basic | Data Preprocessing · Linear Regression · KNN | ✅ Complete |
| Level 2 | Intermediate | Logistic Regression · Decision Trees · K-Means | ✅ Complete |
| Level 3 | Advanced | Random Forest · SVM · Neural Networks | ✅ Complete |

---

## Level 1 — Basic

> **Theme : Building the foundation**
> Before you can build any model, you need to understand your data and prepare it properly. Level 1 covers exactly that.

---

### Task 1 — Data Preprocessing

**Datasets :** `1__iris.csv` (150 rows) + `churn-bigml-80.csv` (2666 rows)

The most underrated step in ML. This task covered the complete preprocessing pipeline that every real project starts with.

| Technique | What it does |
|-----------|-------------|
| Mean Imputation | Fills missing values with the column average |
| Drop Strategy | Removes rows that contain NaN values |
| Label Encoding | Converts category names to integers |
| One-Hot Encoding | Creates a binary column per category |
| StandardScaler | Transforms features to mean=0, std=1 |
| MinMaxScaler | Scales all values to range [0, 1] |
| Train/Test Split | 80% training / 20% testing with stratification |

**Output files :** `iris_preprocessed.csv` · `churn_preprocessed.csv`

```bash
python Level_1_Basic/L1_T1_Data_Preprocessing.py
```

---

### Task 2 — Linear Regression

**Dataset :** `4__house_Prediction_Data_Set.csv` — 506 houses, 13 features
**Goal :** Predict median house prices in $1000s

| Metric | Value |
|--------|-------|
| R² Score (Train) | 0.7509 |
| R² Score (Test) | 0.6688 |
| RMSE | $4.93K |
| MAE | $3.19K |

**Key findings from model coefficients :**

| Feature | Effect |
|---------|--------|
| RM (rooms) | +$3.15K per extra room |
| LSTAT (poverty %) | -$3.61K per unit increase |
| PTRATIO (school ratio) | -$2.04K per unit increase |
| DIS (distance) | -$3.08K per unit increase |

**Plots generated :** Actual vs Predicted · Feature Coefficients · Error Distribution

```bash
python Level_1_Basic/L1_T2_Linear_Regression.py
```

---

### Task 3 — KNN Classifier

**Dataset :** `1__iris.csv` — 3 flower species
**Goal :** Classify iris flowers into Setosa, Versicolor or Virginica

Tested K values from 1 to 15 and compared all results.

| K | Accuracy | F1 Score |
|---|----------|----------|
| K=1 | **0.9667** | **0.9666** |
| K=3 | 0.9333 | 0.9327 |
| K=5 | 0.9333 | 0.9327 |
| K=7 | 0.9667 | 0.9666 |

**Best K = 1 · Accuracy = 96.67% · Only 1 flower misclassified out of 30**

**Plots generated :** K vs Accuracy · Decision Boundary · Confusion Matrix

```bash
python Level_1_Basic/L1_T3_KNN_Classifier.py
```

---

## Level 2 — Intermediate

> **Theme : Real classification and clustering**
> Moving from simple models to ones that handle imbalanced data, visualize decisions and discover hidden patterns.

---

### Task 1 — Logistic Regression for Binary Classification

**Dataset :** `churn-bigml-80.csv` — 2666 customers
**Goal :** Predict whether a customer will churn (binary outcome)

| Metric | Value |
|--------|-------|
| Accuracy | 83.9% |
| Precision | 0.4000 |
| Recall | 0.2051 |
| F1 Score | 0.2712 |
| AUC-ROC | 0.7561 |

**Key finding :** Customers with high customer service calls have an odds ratio of 2.10 — meaning they are twice as likely to churn. International plan users are 1.99x more likely to leave.

**Plots generated :** ROC Curve · Confusion Matrix · Feature Odds Ratios

```bash
python Level_2_Intermediate/L2_T1_Logistic_Regression.py
```

---

### Task 2 — Decision Trees for Classification

**Dataset :** `1__iris.csv`
**Goal :** Classify flowers and visualize the exact decision rules

The biggest advantage of decision trees — you can read what the model actually learned:

```
Is petal_length <= 2.45?
    Yes → Setosa (always correct)
    No  → Is petal_width <= 1.65?
              Yes → Versicolor
              No  → Virginica
```

Demonstrated overfitting clearly — unpruned tree hit 100% train accuracy but only 93.3% on test. Pruning at depth 3 improved test accuracy to **96.7%**.

| Depth | Train Acc | Test Acc |
|-------|-----------|----------|
| 1 | 0.6667 | 0.6667 |
| 2 | 0.9667 | 0.9333 |
| **3** | **0.9833** | **0.9667** |
| 5 (full) | 1.0000 | 0.9333 |

**Plots generated :** Tree Structure · Overfitting Graph · Confusion Matrix · Feature Importance

```bash
python Level_2_Intermediate/L2_T2_Decision_Tree.py
```

---

### Task 3 — K-Means Clustering

**Dataset :** `2__Stock_Prices_Data_Set.csv` — 497,472 rows of S&P 500 data (2014–2017)
**Goal :** Group 505 stocks into clusters based on price behaviour

Engineered 7 features per stock: average close price, average volume, price range, volatility and total return percentage. Used the elbow method and silhouette score to find the optimal number of clusters.

**Best K = 2 · Silhouette Score = 0.8496**

| Cluster | Stocks | Avg Price | Volatility | Return |
|---------|--------|-----------|------------|--------|
| 0 | 497 regular stocks | $76 | 13.7 | 52.3% |
| 1 | 8 premium stocks (AMZN, GOOG, PCLN...) | $677 | 157.6 | 77.6% |

The algorithm separated high-value tech giants from regular stocks entirely on its own — no labels, no guidance.

**Plots generated :** Elbow Curve · Silhouette Scores · 2D Scatter · Price vs Volatility

```bash
python Level_2_Intermediate/L2_T3_KMeans_Clustering.py
```

---

## Level 3 — Advanced

> **Theme : Ensemble methods, margins and deep learning**
> The most challenging level — hyperparameter tuning, kernel tricks and building a neural network from scratch.

---

### Task 1 — Random Forest Classifier

**Dataset :** `churn-bigml-80.csv`
**Goal :** Build a more powerful churn predictor using ensemble learning

Random Forest builds hundreds of decision trees and combines them — much more stable and accurate than a single tree.

| Metric | Value |
|--------|-------|
| CV Mean Accuracy | 0.9542 ± 0.0093 |
| Test Accuracy | 95.5% |
| Precision | 96.5% |
| Recall | 71.8% |
| F1 Score | 0.8235 |

**Best hyperparameters found via GridSearchCV :**
- `n_estimators` : 100
- `max_depth` : 15
- `min_samples_split` : 5

**Top features by importance :**

| Feature | Importance |
|---------|-----------|
| Total Day Charge | 0.1410 |
| Total Day Minutes | 0.1385 |
| Customer Service Calls | 0.1315 |
| International Plan | 0.0964 |

**Plots generated :** Feature Importance · Confusion Matrix · Trees vs Accuracy

```bash
python Level_3_Advanced/L3_T1_Random_Forest.py
```

---

### Task 2 — Support Vector Machine (SVM)

**Dataset :** `1__iris.csv`
**Goal :** Classify species using maximum margin boundaries, compare kernels

SVM finds the hyperplane that separates classes with the widest possible margin.

| Kernel | Accuracy | F1 Score | AUC |
|--------|----------|----------|-----|
| **Linear** | **1.0000** | **1.0000** | **1.0000** |
| RBF | 0.9667 | 0.9666 | 0.9967 |
| Polynomial | 0.9000 | 0.8977 | 0.9933 |
| Sigmoid | 0.9000 | 0.8977 | 0.9900 |

**Linear kernel achieved perfect 100% accuracy** — confirming that Iris classes are linearly separable. A straight boundary is all that is needed.

**Cross Validation Mean : 0.9667 ± 0.0298**

**Plots generated :** Linear Decision Boundary · RBF Decision Boundary · Kernel Comparison · Confusion Matrix

```bash
python Level_3_Advanced/L3_T2_SVM.py
```

---

### Task 3 — Neural Network with TensorFlow / Keras

**Dataset :** Sklearn Digits — 1797 handwritten digit images (8x8 pixels, 10 classes)
**Goal :** Build a feed-forward neural network to classify digits 0–9

**Network Architecture :**

```
Input Layer    →  64 neurons  (one per pixel)
Hidden Layer 1 →  128 neurons  (ReLU activation)
Dropout        →  30% (prevents overfitting)
Hidden Layer 2 →  64 neurons  (ReLU activation)
Dropout        →  20%
Hidden Layer 3 →  32 neurons  (ReLU activation)
Output Layer   →  10 neurons  (Softmax — one per digit)

Total Parameters : 18,986
Optimizer        : Adam (lr=0.001)
Loss Function    : Categorical Crossentropy
```

| Metric | Value |
|--------|-------|
| Training Accuracy | 98.69% |
| Validation Accuracy | 96.53% |
| Test Accuracy | **97.50%** |
| Test Loss | 0.1138 |
| Epochs Trained | 29 (early stopping) |

The model used early stopping — it automatically stopped training when validation loss stopped improving, preventing overfitting.

**Plots generated :** Training/Validation Accuracy · Training/Validation Loss · Confusion Matrix · Sample Predictions with digit images

```bash
python Level_3_Advanced/L3_T3_Neural_Network.py
```

---

## Results at a Glance

| Task | Model | Dataset | Key Result |
|------|-------|---------|-----------|
| L1-T1 | Data Preprocessing | Iris + Churn | Pipeline complete |
| L1-T2 | Linear Regression | Boston Housing | R² = 0.67 |
| L1-T3 | KNN Classifier | Iris | 96.7% accuracy |
| L2-T1 | Logistic Regression | Churn | AUC = 0.756 |
| L2-T2 | Decision Trees | Iris | 96.7% pruned |
| L2-T3 | K-Means Clustering | S&P 500 Stocks | Silhouette = 0.85 |
| L3-T1 | Random Forest | Churn | 95.5% accuracy |
| L3-T2 | SVM | Iris | 100% linear kernel |
| L3-T3 | Neural Network | Digits | 97.5% accuracy |

---

## Project Structure

```
Codveda-ML-Internship/
│
├── README.md
│
├── datasets/
│   ├── 1__iris.csv
│   ├── 2__Stock_Prices_Data_Set.csv
│   ├── 3__Sentiment_dataset.csv
│   ├── 4__house_Prediction_Data_Set.csv
│   ├── churn-bigml-80.csv
│   └── churn-bigml-20.csv
│
├── Level_1_Basic/
│   ├── L1_T1_Data_Preprocessing.py
│   ├── L1_T2_Linear_Regression.py
│   └── L1_T3_KNN_Classifier.py
│
├── Level_2_Intermediate/
│   ├── L2_T1_Logistic_Regression.py
│   ├── L2_T2_Decision_Tree.py
│   └── L2_T3_KMeans_Clustering.py
│
├── Level_3_Advanced/
│   ├── L3_T1_Random_Forest.py
│   ├── L3_T2_SVM.py
│   └── L3_T3_Neural_Network.py
│
└── outputs/
    ├── iris_preprocessed.csv
    ├── churn_preprocessed.csv
    ├── L1_T2_actual_vs_predicted.png
    ├── L1_T2_feature_coefficients.png
    ├── L1_T3_knn_performance.png
    ├── L1_T3_decision_boundary.png
    ├── L2_T1_logistic_regression_results.png
    ├── L2_T2_decision_tree_results.png
    ├── L2_T2_feature_importance.png
    ├── L2_T3_kmeans_results.png
    ├── L3_T1_random_forest_results.png
    ├── L3_T2_svm_results.png
    └── L3_T3_neural_network_results.png
```

---

## Tools and Libraries

```python
Python          3.12
pandas          2.x
numpy           1.x
scikit-learn    1.x
matplotlib      3.x
seaborn         0.x
tensorflow      2.21
keras           (via tensorflow)
```

---

## How to Set Up Locally

```bash
# Clone the repository
git clone https://github.com/Darshan-paapani06/Codveda_ML_Internship.git

# Navigate into project
cd Codveda_ML_Internship

# Install all dependencies
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow

# Run any script — example
python Level_1_Basic/L1_T1_Data_Preprocessing.py
```

---

## Internship Details

| | |
|--|--|
| Company | Codveda Technologies |
| Position | Machine Learning Intern |
| Mode | Remote |
| Duration | 20 March 2026 – 20 April 2026 |
| GitHub | [Darshan-paapani06](https://github.com/Darshan-paapani06) |

---

<div align="center">

Built with genuine effort across 9 tasks, 3 levels and 1 month

**#CodvedaInternship #MachineLearning #Python #maamuu**

</div>
