

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ── Path setup
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "datasets")
OUT  = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


print("=" * 60)
print("  LEVEL 1 — TASK 1 : DATA PREPROCESSING")
print("=" * 60)

print("\n📂 Loading Iris Dataset ...")
iris = pd.read_csv(os.path.join(DATA, "1__iris.csv"))

# ── Step 1 : Exploration
print("\n── Step 1 : Initial Exploration ──")
print(f"  Shape   : {iris.shape}")
print(f"  Columns : {list(iris.columns)}")
print(f"\n  Data types:\n{iris.dtypes}")
print(f"\n  First 5 rows:\n{iris.head()}")
print(f"\n  Statistics:\n{iris.describe().round(2)}")

# ── Step 2 : Check missing values
print("\n── Step 2 : Missing Values ──")
print(iris.isnull().sum())
print("  → No missing values. Creating some to demonstrate handling.")

# ── Step 3 : Missing value handling
print("\n── Step 3 : Missing Value Handling ──")
iris_demo = iris.copy()
np.random.seed(42)
miss_idx = np.random.choice(iris_demo.index, size=10, replace=False)
iris_demo.loc[miss_idx, "sepal_length"] = np.nan
print(f"  Introduced  : {iris_demo['sepal_length'].isnull().sum()} missing values")

# Strategy A — Mean Imputation
imputer = SimpleImputer(strategy="mean")
iris_demo[["sepal_length"]] = imputer.fit_transform(iris_demo[["sepal_length"]])
print(f"  After MEAN fill : {iris_demo['sepal_length'].isnull().sum()} missing")
print(f"  Value used      : {imputer.statistics_[0]:.4f}")

# Strategy B — Drop rows
iris_drop = iris.copy()
iris_drop.loc[miss_idx, "sepal_length"] = np.nan
before = len(iris_drop)
iris_drop.dropna(inplace=True)
print(f"\n  DROP : {before} → {len(iris_drop)} rows  ({before-len(iris_drop)} removed)")

# ── Step 4 : Categorical encoding
print("\n── Step 4 : Encoding Categorical Variable ──")
print(f"  Unique species : {iris['species'].unique()}")

# Label Encoding
le = LabelEncoder()
iris["species_label"] = le.fit_transform(iris["species"])
print(f"\n  Label Encoding : {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(iris[["species","species_label"]].drop_duplicates().to_string(index=False))

# One-Hot Encoding
ohe = pd.get_dummies(iris["species"], prefix="species")
iris_ohe = pd.concat([iris.drop("species", axis=1), ohe], axis=1)
print(f"\n  One-Hot columns : {list(ohe.columns)}")
print(iris_ohe.head(3).to_string())

# ── Step 5 : Scaling
print("\n── Step 5 : Feature Scaling ──")
num_cols = ["sepal_length","sepal_width","petal_length","petal_width"]

std_sc = StandardScaler()
iris_std = iris[num_cols].copy()
iris_std[num_cols] = std_sc.fit_transform(iris[num_cols])
print("\n  StandardScaler (mean≈0, std≈1):")
print(iris_std.describe().round(3).to_string())

mm_sc = MinMaxScaler()
iris_mm = iris[num_cols].copy()
iris_mm[num_cols] = mm_sc.fit_transform(iris[num_cols])
print("\n  MinMaxScaler (range [0,1]):")
print(iris_mm.describe().round(3).to_string())

# ── Step 6 : Train / Test split
print("\n── Step 6 : Train / Test Split ──")
X = iris_std[num_cols]
y = iris["species_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Total  : {len(iris)}")
print(f"  Train  : {len(X_train)}  (80%)")
print(f"  Test   : {len(X_test)}   (20%)")
print(f"  Train class counts : {y_train.value_counts().to_dict()}")
print(f"  Test  class counts : {y_test.value_counts().to_dict()}")

iris_ohe[num_cols] = std_sc.fit_transform(iris_ohe[num_cols])
iris_ohe.to_csv(os.path.join(OUT, "iris_preprocessed.csv"), index=False)
print(f"\n  ✅ Saved → outputs/iris_preprocessed.csv")

print("\n\n" + "="*60)
print("📂 Loading Churn Dataset ...")
churn = pd.read_csv(os.path.join(DATA, "churn-bigml-80.csv"))

print("\n── Step 1 : Exploration ──")
print(f"  Shape   : {churn.shape}")
print(f"\n  First 3 rows:\n{churn.head(3).to_string()}")

print("\n── Step 2 : Missing Values ──")
print(churn.isnull().sum().to_string())

print("\n── Step 3 : Encoding Categorical Variables ──")
for col in ["International plan","Voice mail plan"]:
    churn[col] = churn[col].map({"Yes":1,"No":0})
    print(f"  '{col}' → Yes=1, No=0")

le2 = LabelEncoder()
churn["State_encoded"] = le2.fit_transform(churn["State"])
print(f"  'State' encoded → {churn['State_encoded'].nunique()} codes")

churn["Churn"] = churn["Churn"].astype(int)
print(f"  'Churn' → int  0:{(churn['Churn']==0).sum()}  1:{(churn['Churn']==1).sum()}")

print("\n── Step 4 : Standardizing Features ──")
drop_cols = ["State","Area code","Churn"]
num_churn = churn.drop(columns=drop_cols).select_dtypes(include="number").columns.tolist()
sc2 = StandardScaler()
churn_scaled = churn.copy()
churn_scaled[num_churn] = sc2.fit_transform(churn[num_churn])
print(f"  {len(num_churn)} columns scaled. Max mean ≈ {churn_scaled[num_churn].mean().abs().max():.5f}")

print("\n── Step 5 : Train / Test Split ──")
X_c = churn_scaled.drop(columns=["State","Churn"])
y_c = churn["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.2,
                                            random_state=42, stratify=y_c)
print(f"  Train : {len(X_tr)}  Test : {len(X_te)}")
print(f"  Churn in train → 0:{(y_tr==0).sum()}  1:{y_tr.sum()}")
print(f"  Churn in test  → 0:{(y_te==0).sum()}  1:{y_te.sum()}")

churn_scaled.to_csv(os.path.join(OUT,"churn_preprocessed.csv"), index=False)
print(f"\n  ✅ Saved → outputs/churn_preprocessed.csv")


print("\n\n" + "="*60)
print("  ✅  TASK 1 COMPLETE — DATA PREPROCESSING")
print("="*60)
print("""
  TECHNIQUES COVERED
  ─────────────────────────────────────────────
  1. Missing Value Handling
       Mean Imputation  → fills NaN with mean
       Drop Strategy    → removes rows with NaN

  2. Categorical Encoding
       Label Encoding   → category → integer
       One-Hot Encoding → category → binary cols
       Manual Mapping   → Yes/No → 1/0

  3. Feature Scaling
       StandardScaler   → mean=0, std=1
       MinMaxScaler     → range [0, 1]

  4. Train / Test Split
       80% train / 20% test
       stratify=y → balanced class split
  ─────────────────────────────────────────────
  Outputs saved in outputs/ folder:
       iris_preprocessed.csv
       churn_preprocessed.csv
""")