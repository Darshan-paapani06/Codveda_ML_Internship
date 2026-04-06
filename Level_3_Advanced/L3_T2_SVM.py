import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'datasets')
OUT  = os.path.join(BASE, '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, '1__iris.csv'))
print("Dataset loaded!")
print("Shape :", df.shape)
print("\nSpecies distribution :")
print(df['species'].value_counts())
print("\nFirst 5 rows :")
print(df.head())

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("\nLabel encoding :", dict(zip(le.classes_, le.transform(le.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("Features scaled!")

print("\n--- Comparing Different Kernels ---")
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_results = []

print(f"\n{'Kernel':<10} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>9} {'AUC':>9}")
print(f"{'─'*10} {'─'*10} {'─'*11} {'─'*9} {'─'*9} {'─'*9}")

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42, probability=True)
    svm.fit(X_train_sc, y_train)
    y_pred  = svm.predict(X_test_sc)
    y_proba = svm.predict_proba(X_test_sc)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    auc  = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

    kernel_results.append({
        'kernel': kernel, 'accuracy': acc,
        'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc
    })
    print(f"{kernel:<10} {acc:>10.4f} {prec:>11.4f} {rec:>9.4f} {f1:>9.4f} {auc:>9.4f}")

best_kernel = max(kernel_results, key=lambda x: x['accuracy'])
print(f"\nBest kernel = {best_kernel['kernel']} with accuracy = {best_kernel['accuracy']:.4f}")

print(f"\n--- Training Final SVM with {best_kernel['kernel']} kernel ---")
final_svm = SVC(kernel=best_kernel['kernel'], random_state=42, probability=True, C=1.0)
final_svm.fit(X_train_sc, y_train)
y_pred_final  = final_svm.predict(X_test_sc)
y_proba_final = final_svm.predict_proba(X_test_sc)

acc  = accuracy_score(y_test, y_pred_final)
prec = precision_score(y_test, y_pred_final, average='weighted')
rec  = recall_score(y_test, y_pred_final, average='weighted')
f1   = f1_score(y_test, y_pred_final, average='weighted')
auc  = roc_auc_score(y_test, y_proba_final, multi_class='ovr', average='weighted')
cm   = confusion_matrix(y_test, y_pred_final)

print(f"\nAccuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")
print(f"\nConfusion Matrix :\n{cm}")
print(f"\nClassification Report :")
print(classification_report(y_test, y_pred_final, target_names=le.classes_))

print("\n--- Cross Validation 5-fold ---")
cv_scores = cross_val_score(final_svm, scaler.fit_transform(X), y_encoded, cv=5)
print(f"CV Scores  : {cv_scores.round(4)}")
print(f"CV Mean    : {cv_scores.mean():.4f}")
print(f"CV Std     : {cv_scores.std():.4f}")

print("\n--- Linear vs RBF Direct Comparison ---")
for k in ['linear', 'rbf']:
    r = next(x for x in kernel_results if x['kernel'] == k)
    print(f"  {k:<8} → Accuracy: {r['accuracy']:.4f}  F1: {r['f1']:.4f}  AUC: {r['auc']:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('SVM Classification — Iris Dataset', fontsize=15)

X_2d     = df[['petal_length', 'petal_width']].values
y_2d     = le.transform(df['species'])
sc2d     = StandardScaler()
X_2d_sc  = sc2d.fit_transform(X_2d)

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_2d_sc, y_2d, test_size=0.2, random_state=42, stratify=y_2d
)

colors  = ['#E74C3C', '#3498DB', '#2ECC71']
markers = ['o', 's', '^']

h = 0.02
x_min, x_max = X_2d_sc[:, 0].min() - 1, X_2d_sc[:, 0].max() + 1
y_min, y_max = X_2d_sc[:, 1].min() - 1, X_2d_sc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for idx, (kernel_name, ax) in enumerate(zip(['linear', 'rbf'], [axes[0, 0], axes[0, 1]])):
    svm_2d = SVC(kernel=kernel_name, random_state=42)
    svm_2d.fit(X_tr2, y_tr2)
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, colors=colors)
    ax.contour(xx, yy, Z, colors='gray', linewidths=0.8, alpha=0.5)
    for cls, color, marker in zip([0, 1, 2], colors, markers):
        mask = y_2d == cls
        ax.scatter(X_2d_sc[mask, 0], X_2d_sc[mask, 1],
                   c=color, marker=marker, s=70,
                   edgecolors='black', linewidths=0.5,
                   label=le.classes_[cls])
    svm_acc = accuracy_score(y_te2, svm_2d.predict(X_te2))
    ax.set_xlabel('Petal Length (scaled)')
    ax.set_ylabel('Petal Width (scaled)')
    ax.set_title(f'Decision Boundary — {kernel_name.upper()} kernel\nAccuracy = {svm_acc:.4f}')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.2)

k_names  = [r['kernel'] for r in kernel_results]
k_acc    = [r['accuracy'] for r in kernel_results]
k_f1     = [r['f1'] for r in kernel_results]
k_auc    = [r['auc'] for r in kernel_results]
x_pos    = np.arange(len(k_names))
width    = 0.28

axes[1, 0].bar(x_pos - width, k_acc, width, label='Accuracy', color='steelblue', edgecolor='white')
axes[1, 0].bar(x_pos,         k_f1,  width, label='F1 Score', color='coral',     edgecolor='white')
axes[1, 0].bar(x_pos + width, k_auc, width, label='AUC',      color='mediumseagreen', edgecolor='white')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(k_names)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Kernel Comparison — All Metrics')
axes[1, 0].legend()
axes[1, 0].set_ylim([0.7, 1.05])
axes[1, 0].grid(axis='y', alpha=0.3)

im = axes[1, 1].imshow(cm, cmap='Purples')
axes[1, 1].set_xticks([0, 1, 2])
axes[1, 1].set_yticks([0, 1, 2])
axes[1, 1].set_xticklabels(le.classes_, rotation=15)
axes[1, 1].set_yticklabels(le.classes_)
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title(f'Confusion Matrix ({best_kernel["kernel"].upper()} kernel)\nAccuracy = {acc:.4f}')
plt.colorbar(im, ax=axes[1, 1])
for i in range(3):
    for j in range(3):
        col = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=col)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'L3_T2_svm_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved!")

results_df = pd.DataFrame(kernel_results)
results_df.to_csv(os.path.join(OUT, 'L3_T2_kernel_comparison.csv'), index=False)
print("Kernel comparison CSV saved!")

print("\n--- Summary ---")
print(f"Dataset    : Iris (150 samples, 4 features, 3 classes)")
print(f"Model      : Support Vector Machine (SVM)")
print(f"Best kernel: {best_kernel['kernel']}")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"AUC        : {auc:.4f}")
print(f"CV Mean    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print("\nTask 2 of Level 3 completed!")