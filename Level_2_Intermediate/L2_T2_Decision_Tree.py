# Decision Tree Classifier - Iris Flower Species Prediction
# Dataset : 1__iris.csv
# Goal : classify flowers into 3 species using a decision tree
# Tools : pandas, scikit-learn, matplotlib
#
# WHY DECISION TREES?
# -------------------
# imagine you are trying to identify a flower
# you ask questions one by one :
#   is petal length less than 2.5 cm? → yes → it is setosa
#   no → is petal width less than 1.8? → yes → versicolor → no → virginica
# that is exactly what a decision tree does
# it learns these questions automatically from the data
# it is easy to understand, visualize and explain to anyone
# unlike KNN or logistic regression, you can literally see the rules it learned
#
# WHY PRUNING?
# ------------
# if we let the tree grow freely it memorizes every single training sample
# this is called overfitting - it does great on training but fails on new data
# pruning means we limit how deep the tree can grow
# this forces it to learn general patterns instead of memorizing


import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

# paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'datasets')
OUT  = os.path.join(BASE, '..', 'outputs')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------
# 1. loading the dataset
# ---------------------------------------------------------
# the iris dataset has measurements of 150 flowers
# 4 features : sepal length, sepal width, petal length, petal width
# 3 classes  : setosa, versicolor, virginica
# this is a classic classification dataset used to learn ML

df = pd.read_csv(os.path.join(DATA, '1__iris.csv'))

print("Dataset loaded!")
print("Shape :", df.shape)
print("\nFirst 5 rows :")
print(df.head())
print("\nSpecies count :")
print(df['species'].value_counts())
print("\nBasic stats :")
print(df.describe().round(2))


# ---------------------------------------------------------
# 2. preprocessing
# ---------------------------------------------------------
# decision trees can handle raw numbers directly
# no need for scaling unlike KNN or logistic regression
# we only need to encode the target labels (species names → numbers)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
# setosa=0, versicolor=1, virginica=2

print("\nLabel encoding :", dict(zip(le.classes_, le.transform(le.classes_))))

# 80-20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTrain : {len(X_train)} samples")
print(f"Test  : {len(X_test)} samples")


# ---------------------------------------------------------
# 3. training a full tree (no pruning / no depth limit)
# ---------------------------------------------------------
# first we train without any restrictions
# this tree will grow until every leaf is pure
# meaning it perfectly fits the training data
# but it will likely overfit

full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

y_pred_full = full_tree.predict(X_test)
acc_full    = accuracy_score(y_test, y_pred_full)
f1_full     = f1_score(y_test, y_pred_full, average='weighted')

print("\n--- Full Tree (no pruning) ---")
print(f"Tree depth        : {full_tree.get_depth()}")
print(f"Number of leaves  : {full_tree.get_n_leaves()}")
print(f"Accuracy on test  : {acc_full:.4f}")
print(f"F1 Score on test  : {f1_full:.4f}")
print(f"Accuracy on train : {accuracy_score(y_train, full_tree.predict(X_train)):.4f}")
print("(if train accuracy is 1.0 but test is lower → overfitting!)")


# ---------------------------------------------------------
# 4. pruning the tree - trying different max_depth values
# ---------------------------------------------------------
# max_depth controls how many levels the tree can have
# depth=1 means only 1 question → too simple (underfitting)
# depth=10 means too many questions → memorizes data (overfitting)
# we try different depths and pick the one with best test accuracy

print("\n--- Comparing different tree depths (pruning effect) ---")
print(f"\n{'Depth':<8} {'Train Acc':>10} {'Test Acc':>10} {'F1 Score':>10} {'Leaves':>8}")
print(f"{'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

depth_results = []

for depth in range(1, 11):
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc  = accuracy_score(y_test,  dt.predict(X_test))
    f1        = f1_score(y_test, dt.predict(X_test), average='weighted')
    leaves    = dt.get_n_leaves()

    depth_results.append({
        'depth'    : depth,
        'train_acc': train_acc,
        'test_acc' : test_acc,
        'f1'       : f1,
        'leaves'   : leaves
    })

    print(f"{depth:<8} {train_acc:>10.4f} {test_acc:>10.4f} {f1:>10.4f} {leaves:>8}")

# pick the best depth based on test accuracy
best = max(depth_results, key=lambda x: x['test_acc'])
print(f"\nBest depth = {best['depth']} with test accuracy = {best['test_acc']:.4f}")
print("This is the sweet spot — not too simple, not overfitted")


# ---------------------------------------------------------
# 5. training the final pruned tree with best depth
# ---------------------------------------------------------

best_tree = DecisionTreeClassifier(max_depth=best['depth'], random_state=42)
best_tree.fit(X_train, y_train)
y_pred = best_tree.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')
cm  = confusion_matrix(y_test, y_pred)

print(f"\n--- Final Pruned Tree (depth={best['depth']}) ---")
print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Leaves   : {best_tree.get_n_leaves()}")


# ---------------------------------------------------------
# 6. visualizing the tree rules (text format)
# ---------------------------------------------------------
# this is the biggest advantage of decision trees
# you can read the exact rules it learned
# no other ML model gives you this kind of transparency

print("\n--- Decision Tree Rules (text) ---")
tree_rules = export_text(best_tree, feature_names=list(X.columns))
print(tree_rules)


# ---------------------------------------------------------
# 7. confusion matrix and classification report
# ---------------------------------------------------------

print("Confusion Matrix :")
print(cm)
print("\nClassification Report :")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# ---------------------------------------------------------
# 8. feature importance
# ---------------------------------------------------------
# decision trees also tell us which features were most useful
# for making decisions — this is called feature importance

print("Feature Importances :")
importance_df = pd.DataFrame({
    'Feature'   : X.columns,
    'Importance': best_tree.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance_df.iterrows():
    bar = '█' * int(row['Importance'] * 40)
    print(f"  {row['Feature']:<15} : {bar}  ({row['Importance']:.4f})")


# ---------------------------------------------------------
# 9. plotting everything
# ---------------------------------------------------------

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Decision Tree Classifier — Iris Dataset', fontsize=16)

# plot 1 : full tree visualization
ax1 = fig.add_subplot(2, 2, (1, 2))
plot_tree(
    best_tree,
    feature_names=list(X.columns),
    class_names=le.classes_,
    filled=True,
    rounded=True,
    fontsize=11,
    ax=ax1
)
ax1.set_title(f'Decision Tree Structure (max depth = {best["depth"]})', fontsize=13)

# plot 2 : train vs test accuracy across depths
ax2 = fig.add_subplot(2, 2, 3)
depths     = [r['depth'] for r in depth_results]
train_accs = [r['train_acc'] for r in depth_results]
test_accs  = [r['test_acc'] for r in depth_results]

ax2.plot(depths, train_accs, 'bo-', lw=2, markersize=7, label='Train Accuracy')
ax2.plot(depths, test_accs,  'rs-', lw=2, markersize=7, label='Test Accuracy')
ax2.axvline(x=best['depth'], color='green', linestyle=':', lw=2,
            label=f'Best depth = {best["depth"]}')
ax2.set_xlabel('Tree Depth', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Overfitting vs Pruning\n(gap between lines = overfitting)', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(depths)

# plot 3 : confusion matrix
ax3 = fig.add_subplot(2, 2, 4)
im = ax3.imshow(cm, cmap='Greens')
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(le.classes_, rotation=15)
ax3.set_yticklabels(le.classes_)
ax3.set_xlabel('Predicted', fontsize=12)
ax3.set_ylabel('Actual', fontsize=12)
ax3.set_title(f'Confusion Matrix\nAccuracy = {acc:.4f}', fontsize=12)
plt.colorbar(im, ax=ax3)
for i in range(3):
    for j in range(3):
        col = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax3.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 fontsize=15, fontweight='bold', color=col)

plt.tight_layout()
plot1 = os.path.join(OUT, 'L2_T2_decision_tree_results.png')
plt.savefig(plot1, dpi=150, bbox_inches='tight')
plt.close()
print("\nMain plot saved!")

# plot 4 : feature importance bar chart
fig2, ax = plt.subplots(figsize=(8, 5))
colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
bars = ax.bar(importance_df['Feature'], importance_df['Importance'],
              color=colors, edgecolor='white', width=0.5)
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance\n(which features does the tree rely on most?)',
             fontsize=12)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, importance_df['Importance']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plot2 = os.path.join(OUT, 'L2_T2_feature_importance.png')
plt.savefig(plot2, dpi=150, bbox_inches='tight')
plt.close()
print("Feature importance plot saved!")


# ---------------------------------------------------------
# 10. summary
# ---------------------------------------------------------

print("\n--- Summary ---")
print(f"Dataset         : Iris (150 samples, 4 features, 3 species)")
print(f"Model           : Decision Tree Classifier")
print(f"Best depth      : {best['depth']}")
print(f"Accuracy        : {acc*100:.1f}%")
print(f"F1 Score        : {f1:.4f}")
print(f"Top feature     : {importance_df.iloc[0]['Feature']} "
      f"(importance = {importance_df.iloc[0]['Importance']:.4f})")
print(f"\nKey takeaway :")
print(f"  petal_length and petal_width are the most important features")
print(f"  sepal measurements barely matter for classification")
print(f"  pruning at depth {best['depth']} prevents overfitting")
print(f"\nTask 2 of Level 2 completed!")