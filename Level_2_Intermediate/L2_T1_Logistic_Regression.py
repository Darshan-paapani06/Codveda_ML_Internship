# Logistic Regression - Customer Churn Prediction
# Dataset : churn-bigml-80.csv
# Goal : predict if a customer will leave or not (binary classification)
# Tools : pandas, scikit-learn, matplotlib

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             classification_report, roc_curve, roc_auc_score)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'datasets')
OUT  = os.path.join(BASE, '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, 'churn-bigml-80.csv'))
print("Dataset loaded successfully")
print("Shape :", df.shape)
print("\nFirst few rows :")
print(df.head())

print("\nChurn distribution :")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).mul(100).round(1))

data = df.copy()
data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
data['Voice mail plan']    = data['Voice mail plan'].map({'Yes': 1, 'No': 0})
data['Churn'] = data['Churn'].astype(int)
le = LabelEncoder()
data['State'] = le.fit_transform(data['State'])
data.drop(columns=['Area code'], inplace=True)

X = data.drop(columns=['Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining samples :", len(X_train))
print("Testing  samples :", len(X_test))

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc  = sc.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_sc, y_train)
print("\nModel training done!")

coeff_data = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio' : np.exp(model.coef_[0])
})
coeff_data = coeff_data.sort_values('Odds Ratio', ascending=False)
print("\nFeature Coefficients and Odds Ratios :")
print(coeff_data.to_string(index=False))

y_pred       = model.predict(X_test_sc)
y_pred_proba = model.predict_proba(X_test_sc)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_proba)

print("\n--- Model Evaluation Results ---")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC-ROC   : {auc:.4f}")

print("\nIn simple terms :")
print(f"  - Out of every 100 customers, model correctly predicts {acc*100:.1f}")
print(f"  - When model says someone will churn, it is right {prec*100:.1f}% of the time")
print(f"  - Out of all actual churners, model catches {rec*100:.1f}% of them")
print(f"  - AUC of {auc:.2f} means model is much better than random guessing")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix :")
print(cm)
print("TP :", cm[1][1], "| FP :", cm[0][1], "| TN :", cm[0][0], "| FN :", cm[1][0])

print("\nDetailed Classification Report :")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Logistic Regression — Churn Prediction Results', fontsize=14)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.4f}')
axes[0].plot([0,1],[0,1],'k--',lw=1.5,label='Random guess')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(alpha=0.3)

im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['No Churn','Churn'])
axes[1].set_yticklabels(['No Churn','Churn'])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')
plt.colorbar(im, ax=axes[1])
for i in range(2):
    for j in range(2):
        col = 'white' if cm[i,j] > cm.max()/2 else 'black'
        axes[1].text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=16, fontweight='bold', color=col)

top10 = coeff_data.head(10)
bar_colors = ['coral' if x > 1 else 'steelblue' for x in top10['Odds Ratio']]
axes[2].barh(top10['Feature'], top10['Odds Ratio'], color=bar_colors, edgecolor='white')
axes[2].axvline(x=1, color='black', linestyle='--', lw=1.5)
axes[2].set_xlabel('Odds Ratio')
axes[2].set_title('Feature Odds Ratios\n(coral = increases churn risk)')
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'L2_T1_logistic_regression_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved!")

results = pd.DataFrame({
    'Actual': y_test.values, 'Predicted': y_pred,
    'Churn_Prob': y_pred_proba.round(4), 'Correct': (y_test.values == y_pred)
})
results.to_csv(os.path.join(OUT, 'L2_T1_churn_predictions.csv'), index=False)

print("\n--- Summary ---")
print(f"Accuracy : {acc*100:.1f}% | AUC-ROC : {auc:.4f}")
print("Key driver : Customer service calls and International plan")
print("\nTask 1 of Level 2 completed!")