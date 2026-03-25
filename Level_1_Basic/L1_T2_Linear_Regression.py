

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                   
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "datasets")
OUT  = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  LEVEL 1 — TASK 2 : LINEAR REGRESSION")
print("  Predicting House Prices (Boston Housing Dataset)")
print("=" * 60)


print("\n── Step 1 : Load Dataset ──")

# Column names for the Boston Housing dataset
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM',
        'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

df = pd.read_csv(
    os.path.join(DATA, "4__house_Prediction_Data_Set.csv"),
    header=None, sep=r'\s+', names=cols
)

print(f"  Shape        : {df.shape}")
print(f"  Columns      : {list(df.columns)}")
print(f"\n  Column meanings:")
print("    CRIM    — per capita crime rate by town")
print("    ZN      — proportion of residential land zoned")
print("    INDUS   — proportion of non-retail business acres")
print("    CHAS    — Charles River dummy variable (1=bounds river)")
print("    NOX     — nitric oxides concentration")
print("    RM      — average number of rooms per dwelling")
print("    AGE     — proportion of owner-occupied units built pre-1940")
print("    DIS     — distances to Boston employment centres")
print("    RAD     — index of accessibility to radial highways")
print("    TAX     — full-value property-tax rate per $10,000")
print("    PTRATIO — pupil-teacher ratio by town")
print("    B       — 1000(Bk-0.63)^2 where Bk = proportion of blacks")
print("    LSTAT   — % lower status of the population")
print("    MEDV    — Median value of homes in $1000s  ← TARGET")
print(f"\n  First 5 rows:\n{df.head().to_string()}")

print("\n── Step 2 : Exploratory Data Analysis ──")
print(f"\n  Basic Statistics:\n{df.describe().round(2).to_string()}")
print(f"\n  Missing values : {df.isnull().sum().sum()} (none!)")
print(f"\n  Target (MEDV) range : ${df['MEDV'].min()}K — ${df['MEDV'].max()}K")
print(f"  Target (MEDV) mean  : ${df['MEDV'].mean():.2f}K")

# Plot 1 — Correlation heatmap (text-based)
print(f"\n  Top correlations with MEDV (house price):")
corr = df.corr()['MEDV'].sort_values()
for feat, val in corr.items():
    bar = "█" * int(abs(val) * 20)
    sign = "+" if val > 0 else "-"
    print(f"    {feat:>8} : {sign}{bar}  ({val:.3f})")


print("\n── Step 3 : Preprocessing ──")

X = df.drop('MEDV', axis=1)   # 13 features
y = df['MEDV']                 # target: house price

# Train/Test Split — 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train set : {X_train.shape[0]} rows")
print(f"  Test  set : {X_test.shape[0]} rows")

# Standardize features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"  Features standardized (mean≈0, std≈1)")

print("\n── Step 4 : Training Linear Regression Model ──")

model = LinearRegression()
model.fit(X_train_sc, y_train)
print("  Model trained successfully!")


print("\n── Step 5 : Model Coefficients (Feature Importance) ──")
print(f"  Intercept (base price) : ${model.intercept_:.2f}K")
print(f"\n  Feature Coefficients:")
print(f"  {'Feature':<10} {'Coefficient':>12}  Interpretation")
print(f"  {'─'*10} {'─'*12}  {'─'*35}")

coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

for _, row in coef_df.iterrows():
    effect = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {row['Feature']:<10} {row['Coefficient']:>12.4f}  "
          f"1 unit increase → price {effect} by ${abs(row['Coefficient']):.2f}K")

print("\n── Step 6 : Model Evaluation ──")

y_pred_train = model.predict(X_train_sc)
y_pred_test  = model.predict(X_test_sc)

# Metrics
mse_train  = mean_squared_error(y_train, y_pred_train)
mse_test   = mean_squared_error(y_test,  y_pred_test)
rmse_test  = np.sqrt(mse_test)
mae_test   = mean_absolute_error(y_test, y_pred_test)
r2_train   = r2_score(y_train, y_pred_train)
r2_test    = r2_score(y_test,  y_pred_test)

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │           MODEL PERFORMANCE             │")
print(f"  ├─────────────────────┬───────────────────┤")
print(f"  │ Metric              │ Value             │")
print(f"  ├─────────────────────┼───────────────────┤")
print(f"  │ R² Score (train)    │ {r2_train:.4f}    │")
print(f"  │ R² Score (test)     │ {r2_test:.4f}     │")
print(f"  │ MSE  (test)         │ {mse_test:.4f}    │")
print(f"  │ RMSE (test)         │ {rmse_test:.4f}   │")
print(f"  │ MAE  (test)         │ {mae_test:.4f}    │")
print(f"  └─────────────────────┴───────────────────┘")

print(f"""
  What these metrics mean:
  • R² = {r2_test:.4f} → model explains {r2_test*100:.1f}% of price variation
  • RMSE = {rmse_test:.2f} → predictions are off by ~${rmse_test:.2f}K on average
  • MAE  = {mae_test:.2f}  → average absolute error is ${mae_test:.2f}K
""")

print("── Step 7 : Generating Plots ──")

# ── Plot 1: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Linear Regression — House Price Prediction', fontsize=15, fontweight='bold')

axes[0].scatter(y_test, y_pred_test, alpha=0.6, color='steelblue', edgecolors='white', linewidths=0.5)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price ($1000s)', fontsize=12)
axes[0].set_ylabel('Predicted Price ($1000s)', fontsize=12)
axes[0].set_title(f'Actual vs Predicted\nR² = {r2_test:.4f}', fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Plot 2: Residuals
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.6, color='coral', edgecolors='white', linewidths=0.5)
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price ($1000s)', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Residual Plot\n(should be random around 0)', fontsize=13)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot1_path = os.path.join(OUT, 'L1_T2_actual_vs_predicted.png')
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → outputs/L1_T2_actual_vs_predicted.png")

# ── Plot 2: Feature Coefficients Bar Chart
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['steelblue' if c > 0 else 'coral' for c in coef_df['Coefficient']]
bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Feature Coefficients\n(Blue = increases price, Red = decreases price)',
             fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, coef_df['Coefficient']):
    ax.text(val + (0.05 if val >= 0 else -0.05),
            bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=10)

plt.tight_layout()
plot2_path = os.path.join(OUT, 'L1_T2_feature_coefficients.png')
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → outputs/L1_T2_feature_coefficients.png")

# ── Plot 3: Distribution of errors
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
ax.axvline(x=residuals.mean(), color='orange', linestyle='--', lw=2,
           label=f'Mean Error: {residuals.mean():.2f}')
ax.set_xlabel('Prediction Error ($1000s)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot3_path = os.path.join(OUT, 'L1_T2_error_distribution.png')
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved → outputs/L1_T2_error_distribution.png")

# ── Save predictions to CSV
results_df = pd.DataFrame({
    'Actual_Price'   : y_test.values,
    'Predicted_Price': y_pred_test.round(2),
    'Error'          : residuals.values.round(2)
})
results_df.to_csv(os.path.join(OUT, 'L1_T2_predictions.csv'), index=False)
print(f"  ✅ Saved → outputs/L1_T2_predictions.csv")


print("\n── Step 8 : Sample Predictions ──")
print(f"\n  {'#':<5} {'Actual ($K)':>12} {'Predicted ($K)':>15} {'Error ($K)':>12}")
print(f"  {'─'*5} {'─'*12} {'─'*15} {'─'*12}")
for i in range(10):
    print(f"  {i+1:<5} {y_test.values[i]:>12.1f} "
          f"{y_pred_test[i]:>15.1f} "
          f"{residuals.values[i]:>12.1f}")


print(f"""
{"="*60}
  ✅  TASK 2 COMPLETE — LINEAR REGRESSION
{"="*60}

  WHAT WAS DONE:
  ─────────────────────────────────────────────
  Dataset  : Boston Housing (506 houses, 13 features)
  Target   : MEDV — Median House Price in $1000s
  Model    : Linear Regression (scikit-learn)
  Split    : 80% train ({X_train.shape[0]}) / 20% test ({X_test.shape[0]})

  RESULTS:
  ─────────────────────────────────────────────
  R² Score : {r2_test:.4f}  ({r2_test*100:.1f}% variance explained)
  RMSE     : ${rmse_test:.2f}K  (avg prediction error)
  MAE      : ${mae_test:.2f}K

  KEY INSIGHT:
  ─────────────────────────────────────────────
  Top features affecting house price:
  • RM      (rooms)    → more rooms = higher price
  • LSTAT   (% poor)  → more poverty = lower price
  • PTRATIO (school)  → worse school = lower price

  OUTPUT FILES (in outputs/ folder):
  ─────────────────────────────────────────────
  • L1_T2_actual_vs_predicted.png
  • L1_T2_feature_coefficients.png
  • L1_T2_error_distribution.png
  • L1_T2_predictions.csv
""")