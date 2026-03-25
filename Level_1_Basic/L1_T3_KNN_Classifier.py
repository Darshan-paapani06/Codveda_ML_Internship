
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report,
                             precision_score, recall_score, f1_score)

# ── Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "datasets")
OUT  = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  LEVEL 1 — TASK 3 : KNN CLASSIFIER")
print("  Classifying Iris Flower Species")
print("=" * 60)

print("\n── Step 1 : Load Dataset ──")
df = pd.read_csv(os.path.join(DATA, "1__iris.csv"))

print(f"  Shape   : {df.shape}")
print(f"  Columns : {list(df.columns)}")
print(f"\n  First 5 rows:\n{df.head().to_string()}")
print(f"\n  Species distribution:")
for species, count in df['species'].value_counts().items():
    print(f"    {species:<12} : {count} samples")


print("\n── Step 2 : Preprocessing ──")

# Features and Target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"  Label encoding : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"  Train : {len(X_train)} samples | Test : {len(X_test)} samples")

# Scale features (VERY important for KNN — distance based!)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"  Features scaled ✅  (critical for KNN distance calculations)")

print("\n── Step 3 : Testing Different Values of K ──")
print(f"\n  {'K':<6} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>9}")
print(f"  {'─'*6} {'─'*10} {'─'*11} {'─'*9} {'─'*9}")

k_values   = [1, 3, 5, 7, 9, 11, 13, 15]
k_results  = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_sc, y_train)
    y_pred = knn.predict(X_test_sc)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    k_results.append({'K': k, 'Accuracy': acc, 'Precision': prec,
                      'Recall': rec, 'F1': f1})
    print(f"  K={k:<4} {acc:>10.4f} {prec:>11.4f} {rec:>9.4f} {f1:>9.4f}")

# Best K
best = max(k_results, key=lambda x: x['Accuracy'])
print(f"\n  ★ Best K = {best['K']} with Accuracy = {best['Accuracy']:.4f}")


print(f"\n── Step 4 : Final Model with Best K={best['K']} ──")

best_knn = KNeighborsClassifier(n_neighbors=best['K'])
best_knn.fit(X_train_sc, y_train)
y_pred_final = best_knn.predict(X_test_sc)


print("\n── Step 5 : Model Evaluation ──")

acc_final  = accuracy_score(y_test, y_pred_final)
prec_final = precision_score(y_test, y_pred_final, average='weighted')
rec_final  = recall_score(y_test, y_pred_final, average='weighted')
f1_final   = f1_score(y_test, y_pred_final, average='weighted')

print(f"""
  
         FINAL MODEL PERFORMANCE          
                K = {best['K']}                       

   Metric               Value             
  
   Accuracy              {acc_final:.4f}            
   Precision (weighted)  {prec_final:.4f}            
   Recall    (weighted)  {rec_final:.4f}            
   F1 Score  (weighted) {f1_final:.4f}            
  
""")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print("  Confusion Matrix:")
print(f"  {'':>12} {'Pred:setosa':>13} {'Pred:versi':>11} {'Pred:virgi':>11}")
print(f"  {'─'*50}")
for i, label in enumerate(le.classes_):
    row = cm[i]
    print(f"  {'True:'+label:<14} {row[0]:>11}   {row[1]:>9}   {row[2]:>9}")

print(f"""
  How to read the confusion matrix:
  • Diagonal values = correct predictions
  • Off-diagonal    = misclassifications
  • Perfect model would have all values on diagonal
""")

# Classification Report
print("  Detailed Classification Report:")
report = classification_report(
    y_test, y_pred_final,
    target_names=le.classes_
)
print(report)

print("── Step 6 : Generating Plots ──")

# ── Plot 1: K vs Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('KNN Classifier — Iris Dataset', fontsize=15, fontweight='bold')

k_list  = [r['K'] for r in k_results]
acc_list = [r['Accuracy'] for r in k_results]
f1_list  = [r['F1'] for r in k_results]

axes[0].plot(k_list, acc_list, 'bo-', linewidth=2, markersize=8, label='Accuracy')
axes[0].plot(k_list, f1_list,  'rs--', linewidth=2, markersize=8, label='F1 Score')
axes[0].axvline(x=best['K'], color='green', linestyle=':', lw=2,
                label=f'Best K={best["K"]}')
axes[0].set_xlabel('K Value', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('K Value vs Model Performance', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(k_list)
axes[0].set_ylim([0.85, 1.02])

# ── Plot 2: Confusion Matrix Heatmap
im = axes[1].imshow(cm, interpolation='nearest', cmap='Blues')
axes[1].set_title(f'Confusion Matrix (K={best["K"]})', fontsize=13)
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xticks([0, 1, 2])
axes[1].set_yticks([0, 1, 2])
axes[1].set_xticklabels(le.classes_, rotation=15)
axes[1].set_yticklabels(le.classes_)
plt.colorbar(im, ax=axes[1])

# Add numbers inside confusion matrix
for i in range(3):
    for j in range(3):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        axes[1].text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     fontsize=16, fontweight='bold', color=color)

plt.tight_layout()
plot1 = os.path.join(OUT, 'L1_T3_knn_performance.png')
plt.savefig(plot1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → outputs/L1_T3_knn_performance.png")

# ── Plot 2: Decision Boundary (using 2 best features: petal_length vs petal_width)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('KNN Decision Boundary — Petal Features', fontsize=14, fontweight='bold')

colors  = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', '^']
species_names = le.classes_

# Use only petal features for 2D visualization
X_petal = df[['petal_length', 'petal_width']].values
y_enc   = le.transform(df['species'])

sc2 = StandardScaler()
X_petal_sc = sc2.fit_transform(X_petal)

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_petal_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

knn2d = KNeighborsClassifier(n_neighbors=best['K'])
knn2d.fit(X_tr2, y_tr2)

# Mesh grid for decision boundary
h = 0.02
x_min, x_max = X_petal_sc[:, 0].min() - 1, X_petal_sc[:, 0].max() + 1
y_min, y_max = X_petal_sc[:, 1].min() - 1, X_petal_sc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

for ax_idx, (X_plot, y_plot, title) in enumerate([
    (X_tr2, y_tr2, f'Training Set (K={best["K"]})'),
    (X_te2, y_te2, f'Test Set (K={best["K"]})')
]):
    axes[ax_idx].contourf(xx, yy, Z, alpha=0.25,
                          colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[ax_idx].contour(xx, yy, Z, colors='gray', linewidths=0.5, alpha=0.5)

    for cls, color, marker in zip([0, 1, 2], colors, markers):
        mask = y_plot == cls
        axes[ax_idx].scatter(X_plot[mask, 0], X_plot[mask, 1],
                             c=color, marker=marker, s=60,
                             edgecolors='black', linewidths=0.5,
                             label=species_names[cls])

    axes[ax_idx].set_xlabel('Petal Length (scaled)', fontsize=11)
    axes[ax_idx].set_ylabel('Petal Width (scaled)', fontsize=11)
    axes[ax_idx].set_title(title, fontsize=12)
    axes[ax_idx].legend(loc='upper left', fontsize=9)
    axes[ax_idx].grid(True, alpha=0.2)

plt.tight_layout()
plot2 = os.path.join(OUT, 'L1_T3_decision_boundary.png')
plt.savefig(plot2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved → outputs/L1_T3_decision_boundary.png")

# ── Plot 3: All K results bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(k_list))
width  = 0.35
bars1  = ax.bar(x - width/2, acc_list, width, label='Accuracy',
                color='steelblue', edgecolor='white')
bars2  = ax.bar(x + width/2, f1_list, width, label='F1 Score',
                color='coral', edgecolor='white')

ax.set_xlabel('K Value', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Accuracy & F1 Score for Different K Values', fontsize=13,
             fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'K={k}' for k in k_list])
ax.set_ylim([0.85, 1.05])
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# Value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plot3 = os.path.join(OUT, 'L1_T3_k_comparison.png')
plt.savefig(plot3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → outputs/L1_T3_k_comparison.png")


results_df = pd.DataFrame(k_results)
results_df.to_csv(os.path.join(OUT, 'L1_T3_k_comparison.csv'), index=False)
print(f"  ✅ Saved → outputs/L1_T3_k_comparison.csv")

print("\n── Step 7 : Sample Predictions ──")
print(f"\n  {'#':<5} {'Actual':>12} {'Predicted':>12} {'Correct?':>10}")
print(f"  {'─'*5} {'─'*12} {'─'*12} {'─'*10}")
for i in range(15):
    actual    = le.classes_[y_test[i]]
    predicted = le.classes_[y_pred_final[i]]
    correct   = "✅" if y_test[i] == y_pred_final[i] else "❌"
    print(f"  {i+1:<5} {actual:>12} {predicted:>12} {correct:>10}")

print(f"""
{"="*60}
  ✅  TASK 3 COMPLETE — KNN CLASSIFIER
{"="*60}

  WHAT WAS DONE:
  ─────────────────────────────────────────────
  Dataset   : Iris (150 samples, 4 features, 3 classes)
  Model     : K-Nearest Neighbors Classifier
  K tested  : {k_values}
  Best K    : {best['K']}
  Split     : 80% train ({len(X_train)}) / 20% test ({len(X_test)})

  BEST MODEL RESULTS (K={best['K']}):
  ─────────────────────────────────────────────
  Accuracy  : {acc_final:.4f}  ({acc_final*100:.1f}% correct)
  Precision : {prec_final:.4f}
  Recall    : {rec_final:.4f}
  F1 Score  : {f1_final:.4f}

  KEY INSIGHTS:
  ─────────────────────────────────────────────
  • KNN classifies by finding the K closest neighbors
  • Scaling is CRITICAL — KNN uses distance (Euclidean)
  • Small K → overfitting | Large K → underfitting
  • Best K found by comparing accuracy across K values
  • Iris is nearly linearly separable → high accuracy!

  OUTPUT FILES (in outputs/ folder):
  ─────────────────────────────────────────────
  • L1_T3_knn_performance.png   (K vs accuracy + confusion matrix)
  • L1_T3_decision_boundary.png (visual boundary between species)
  • L1_T3_k_comparison.png      (bar chart all K values)
  • L1_T3_k_comparison.csv      (all K results as table)
""")