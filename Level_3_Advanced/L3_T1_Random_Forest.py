import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'datasets')
OUT  = os.path.join(BASE, '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, 'churn-bigml-80.csv'))
print("Dataset loaded!")
print("Shape :", df.shape)
print("\nChurn distribution :")
print(df['Churn'].value_counts())

data = df.copy()
data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
data['Voice mail plan']    = data['Voice mail plan'].map({'Yes': 1, 'No': 0})
data['Churn']              = data['Churn'].astype(int)
le = LabelEncoder()
data['State'] = le.fit_transform(data['State'])
data.drop(columns=['Area code'], inplace=True)

X = data.drop(columns=['Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

print("\n--- Default Random Forest ---")
rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
acc_default    = accuracy_score(y_test, y_pred_default)
print(f"Accuracy : {acc_default:.4f}")

print("\n--- Cross Validation 5-fold ---")
cv_scores = cross_val_score(rf_default, X, y, cv=5, scoring='accuracy')
print(f"CV Scores  : {cv_scores.round(4)}")
print(f"CV Mean    : {cv_scores.mean():.4f}")
print(f"CV Std Dev : {cv_scores.std():.4f}")

print("\n--- Hyperparameter Tuning with GridSearchCV ---")
param_grid = {
    'n_estimators'     : [50, 100, 200],
    'max_depth'        : [5, 10, 15, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
print(f"Best parameters : {grid_search.best_params_}")
print(f"Best F1 score   : {grid_search.best_score_:.4f}")

print("\n--- Final Model with Best Parameters ---")
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"\nConfusion Matrix :\n{cm}")
print(f"\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("\n--- Feature Importance ---")
importance_df = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)
print("Top 10 features :")
print(importance_df.head(10).to_string(index=False))

print(f"\nDefault accuracy : {acc_default:.4f}")
print(f"Tuned   accuracy : {acc:.4f}")
print(f"Gain             : +{(acc - acc_default)*100:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Random Forest Classifier — Churn Prediction', fontsize=14)

top15 = importance_df.head(15)
c = ['steelblue' if i < 3 else 'lightsteelblue' for i in range(len(top15))]
axes[0].barh(top15['Feature'], top15['Importance'], color=c, edgecolor='white')
axes[0].set_xlabel('Importance Score')
axes[0].set_title('Feature Importance\n(top 3 highlighted)')
axes[0].grid(axis='x', alpha=0.3)
axes[0].invert_yaxis()

im = axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['No Churn','Churn'])
axes[1].set_yticklabels(['No Churn','Churn'])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title(f'Confusion Matrix\nAccuracy = {acc:.4f}')
plt.colorbar(im, ax=axes[1])
for i in range(2):
    for j in range(2):
        col = 'white' if cm[i,j] > cm.max()/2 else 'black'
        axes[1].text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=16, fontweight='bold', color=col)

n_trees_list = [10, 25, 50, 100, 150, 200]
tree_acc = []
for n in n_trees_list:
    rf_t = RandomForestClassifier(
        n_estimators=n,
        max_depth=grid_search.best_params_['max_depth'],
        random_state=42
    )
    rf_t.fit(X_train, y_train)
    tree_acc.append(accuracy_score(y_test, rf_t.predict(X_test)))

axes[2].plot(n_trees_list, tree_acc, 'go-', lw=2, markersize=8)
axes[2].set_xlabel('Number of Trees')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Accuracy vs Number of Trees')
axes[2].grid(alpha=0.3)
axes[2].set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'L3_T1_random_forest_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved!")

importance_df.to_csv(os.path.join(OUT, 'L3_T1_feature_importance.csv'), index=False)
print("Feature importance CSV saved!")

print("\n--- Summary ---")
print(f"Dataset     : Churn (2666 customers, 18 features)")
print(f"Model       : Random Forest Classifier")
print(f"Best params : {grid_search.best_params_}")
print(f"CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"Accuracy    : {acc:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"Top feature : {importance_df.iloc[0]['Feature']}")
print("\nTask 1 of Level 3 completed!")