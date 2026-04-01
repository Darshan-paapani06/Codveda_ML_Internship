import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'datasets')
OUT  = os.path.join(BASE, '..', 'outputs')
os.makedirs(OUT, exist_ok=True)


df = pd.read_csv(os.path.join(DATA, '2__Stock_Prices_Data_Set.csv'))

print("Dataset loaded!")
print("Shape :", df.shape)
print("\nFirst few rows :")
print(df.head())
print("\nColumn types :")
print(df.dtypes)
print("\nMissing values :", df.isnull().sum().sum())


stock_features = df.groupby('symbol').agg(
    avg_close    = ('close',  'mean'),
    avg_volume   = ('volume', 'mean'),
    avg_high     = ('high',   'mean'),
    avg_low      = ('low',    'mean'),
    price_range  = ('close',  lambda x: x.max() - x.min()),
    volatility   = ('close',  'std'),
    return_pct   = ('close',  lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
).reset_index()

print("\nFeatures engineered per stock :")
print(stock_features.head(10).to_string())
print("\nShape after aggregation :", stock_features.shape)


features_for_clustering = ['avg_close', 'avg_volume', 'avg_high',
                            'avg_low', 'price_range', 'volatility', 'return_pct']

X = stock_features[features_for_clustering].copy()
X = X.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures scaled successfully")
print("Mean after scaling (should be ~0) :", X_scaled.mean().round(4))


print("\nFinding optimal K using Elbow Method :")
print(f"\n{'K':<6} {'Inertia':>12} {'Silhouette':>12}")
print(f"{'─'*6} {'─'*12} {'─'*12}")

inertia_values    = []
silhouette_values = []
k_range           = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia    = km.inertia_
    sil_score  = silhouette_score(X_scaled, km.labels_)
    inertia_values.append(inertia)
    silhouette_values.append(sil_score)
    print(f"{k:<6} {inertia:>12.2f} {sil_score:>12.4f}")

best_k = k_range[silhouette_values.index(max(silhouette_values))]
print(f"\nBest K based on silhouette score = {best_k}")


final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_model.fit(X_scaled)

stock_features['Cluster'] = final_model.labels_

print(f"\nCluster distribution :")
print(stock_features['Cluster'].value_counts().sort_index())


print("\nCluster profiles (average values per cluster) :")
cluster_profile = stock_features.groupby('Cluster')[features_for_clustering].mean().round(2)
print(cluster_profile.to_string())

print("\nSample stocks per cluster :")
for cluster_id in sorted(stock_features['Cluster'].unique()):
    stocks = stock_features[stock_features['Cluster'] == cluster_id]['symbol'].values[:8]
    print(f"  Cluster {cluster_id} : {', '.join(stocks)}")


from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

stock_features['PCA1'] = X_pca[:, 0]
stock_features['PCA2'] = X_pca[:, 1]

print(f"\nPCA variance explained : {pca.explained_variance_ratio_.sum()*100:.1f}%")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('K-Means Clustering — Stock Market Segmentation', fontsize=15)

colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple',
          'goldenrod', 'tomato', 'teal', 'peru', 'slategray']

axes[0, 0].plot(list(k_range), inertia_values, 'bo-', lw=2, markersize=8)
axes[0, 0].axvline(x=best_k, color='red', linestyle='--', lw=2,
                   label=f'Best K = {best_k}')
axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 0].set_ylabel('Inertia (WCSS)', fontsize=12)
axes[0, 0].set_title('Elbow Method\n(look for the bend in the curve)', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xticks(list(k_range))

axes[0, 1].plot(list(k_range), silhouette_values, 'rs-', lw=2, markersize=8)
axes[0, 1].axvline(x=best_k, color='blue', linestyle='--', lw=2,
                   label=f'Best K = {best_k}')
axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
axes[0, 1].set_title('Silhouette Score\n(higher is better)', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xticks(list(k_range))

for cluster_id in sorted(stock_features['Cluster'].unique()):
    mask = stock_features['Cluster'] == cluster_id
    axes[1, 0].scatter(
        stock_features.loc[mask, 'PCA1'],
        stock_features.loc[mask, 'PCA2'],
        c=colors[cluster_id], label=f'Cluster {cluster_id}',
        alpha=0.7, s=60, edgecolors='white', linewidths=0.4
    )
axes[1, 0].set_xlabel('PCA Component 1', fontsize=12)
axes[1, 0].set_ylabel('PCA Component 2', fontsize=12)
axes[1, 0].set_title(f'2D Scatter Plot of Clusters (K={best_k})\nvia PCA', fontsize=12)
axes[1, 0].legend(loc='best', fontsize=9)
axes[1, 0].grid(alpha=0.3)

for cluster_id in sorted(stock_features['Cluster'].unique()):
    mask = stock_features['Cluster'] == cluster_id
    axes[1, 1].scatter(
        stock_features.loc[mask, 'avg_close'],
        stock_features.loc[mask, 'volatility'],
        c=colors[cluster_id], label=f'Cluster {cluster_id}',
        alpha=0.7, s=60, edgecolors='white', linewidths=0.4
    )
axes[1, 1].set_xlabel('Average Close Price ($)', fontsize=12)
axes[1, 1].set_ylabel('Volatility (Std Dev)', fontsize=12)
axes[1, 1].set_title('Price vs Volatility by Cluster\n(key business insight)', fontsize=12)
axes[1, 1].legend(loc='best', fontsize=9)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'L2_T3_kmeans_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nMain plot saved!")


fig2, ax = plt.subplots(figsize=(12, 6))
cluster_counts = stock_features['Cluster'].value_counts().sort_index()
bar_colors = [colors[i] for i in cluster_counts.index]
bars = ax.bar([f'Cluster {i}' for i in cluster_counts.index],
              cluster_counts.values, color=bar_colors, edgecolor='white', width=0.5)
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Number of Stocks', fontsize=12)
ax.set_title('Number of Stocks in Each Cluster', fontsize=13)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, cluster_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            str(val), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'L2_T3_cluster_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Cluster distribution plot saved!")


stock_features.to_csv(os.path.join(OUT, 'L2_T3_clustered_stocks.csv'), index=False)
print("Results saved to CSV!")

print("\n--- Summary ---")
print(f"Dataset       : S&P 500 Stocks (505 stocks, 2014-2017)")
print(f"Features used : avg close, avg volume, price range, volatility, return %")
print(f"Best K        : {best_k}")
print(f"Silhouette    : {max(silhouette_values):.4f}")
print(f"\nCluster interpretation :")
for cluster_id in sorted(stock_features['Cluster'].unique()):
    profile = cluster_profile.loc[cluster_id]
    count   = (stock_features['Cluster'] == cluster_id).sum()
    print(f"  Cluster {cluster_id} ({count} stocks) → "
          f"avg price ${profile['avg_close']:.0f}, "
          f"volatility {profile['volatility']:.1f}, "
          f"return {profile['return_pct']:.1f}%")

print("\nTask 3 of Level 2 completed!")