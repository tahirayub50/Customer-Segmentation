# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from features import load_customer_features, FEATURES
from training import (dbscan_min_samples, k_distances, find_eps, fit_dbscan, outlier_flags,
                      evaluate_k, choose_k, fit_kmeans, name_segments)

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# Customers with transformed + scaled features (DBSCAN uses distances, so scaling is essential)
customers, scaled, scaler = load_customer_features()
print(f"{len(scaled):,} customers x {len(FEATURES)} features\n")


# 1. Choose min_samples and eps
# min_samples: how many neighbours make a dense area. Rule of thumb: 2 x number of features.
min_samples = dbscan_min_samples()
# eps: how close is "a neighbour". The k-distance plot sorts every customer's distance to their
# min_samples-th neighbour; the knee is where distances start shooting up.
distances = k_distances(scaled, min_samples)
eps = find_eps(distances)
print(f"min_samples = {min_samples} (2 x {len(FEATURES)} features)")
print(f"eps = {eps:.3f} (knee of the k-distance curve; {(distances > eps).mean():.1%} of customers lie beyond it)")

plt.figure(figsize=(9, 5))
plt.plot(distances, color='steelblue')
plt.axhline(eps, color='red', linestyle='--', label=f'Knee: eps = {eps:.2f}')
plt.xlabel('Customers, sorted by distance')
plt.ylabel(f'Distance to {min_samples}th nearest neighbour')
plt.title('k-Distance Plot: Choosing eps for DBSCAN')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '15_k_distance.png'))


# 2. How sensitive is DBSCAN to eps?
sweep = []
for e in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, round(eps, 2), 1.5, 2.0]:
    labels = fit_dbscan(scaled, e, min_samples).labels_
    sizes = pd.Series(labels[labels >= 0]).value_counts()
    sweep.append({'eps': e, 'clusters': len(sizes), 'noise_share': (labels == -1).mean(),
                  'largest_cluster': int(sizes.max()) if len(sizes) else 0})
sweep = pd.DataFrame(sweep).drop_duplicates('eps').sort_values('eps').set_index('eps')
print("\nHow the result changes with eps:")
print(sweep.assign(noise_share=(sweep['noise_share'] * 100).round(1).astype(str) + '%'))

fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.bar(sweep.index.astype(str), sweep['noise_share'] * 100, color='lightcoral', label='Noise (%)')
ax1.set_ylabel('Customers marked as noise (%)')
ax1.set_xlabel('eps')
ax2 = ax1.twinx()
ax2.plot(sweep.index.astype(str), sweep['clusters'], color='steelblue', marker='o', linewidth=2, label='Clusters')
ax2.set_ylabel('Number of clusters')
fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
plt.title('DBSCAN Is Very Sensitive to eps')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '16_eps_sweep.png'))

# At a small eps DBSCAN does find several clusters. What are they?
small_eps = 0.5
small = fit_dbscan(scaled, small_eps, min_samples).labels_
big_ones = pd.Series(small[small >= 0]).value_counts().head(2).index
print(f"\nAt eps = {small_eps}, the two largest clusters (medians):")
print(customers.assign(cluster=small)[lambda d: d['cluster'].isin(big_ones)]
      .groupby('cluster')[FEATURES].median().round(2))
print("-> customers with exactly the same small values (1 order, no returns) stack up into very "
      "dense piles, and DBSCAN calls each pile a cluster")


# 3. DBSCAN at the chosen eps: one dense group + outliers
model = fit_dbscan(scaled, eps, min_samples)
labels = model.labels_
outliers = labels == -1
print(f"\nAt eps = {eps:.2f}: {len(set(labels) - {-1})} cluster(s), "
      f"{outliers.sum()} outliers ({outliers.mean():.1%} of customers)")

# Our rule for new customers must give the same answer as DBSCAN on the training customers
assert (outlier_flags(model, scaled) == outliers).all()
print("outlier_flags() reproduces DBSCAN's noise labels exactly, so the app can use it for new customers")

comparison = pd.DataFrame({
    'outliers (median)': customers[outliers][FEATURES + ['AvgOrderValue']].median(),
    'everyone else (median)': customers[~outliers][FEATURES + ['AvgOrderValue']].median()
})
print("\nOutliers vs everyone else:")
print(comparison.round(2))
print(f"\nOutliers are {outliers.mean():.1%} of customers but bring "
      f"{customers.loc[outliers, 'Monetary'].sum() / customers['Monetary'].sum():.1%} of revenue")
print(f"Outliers outside the UK: {1 - customers.loc[outliers, 'IsUK'].mean():.0%} "
      f"(vs {1 - customers['IsUK'].mean():.0%} of all customers)")

# Why is each outlier an outlier? Its most extreme feature (largest distance from average)
reason = scaled[outliers].abs().idxmax(axis=1)
direction = np.sign(scaled[outliers].values[np.arange(outliers.sum()), [FEATURES.index(r) for r in reason]])
reason = [f"{'very high' if d > 0 else 'very low'} {r}" for r, d in zip(reason, direction)]
reasons = pd.Series(reason).value_counts()
print("\nMost extreme feature of each outlier:")
print(reasons)

plt.figure(figsize=(8, 4.5))
sns.barplot(x=reasons.values, y=reasons.index, color='lightcoral')
plt.xlabel('Outliers')
plt.ylabel('')
plt.title("Why Customers Are Outliers (each outlier's most extreme feature)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '17_outlier_reasons.png'))

print("\nThe 8 biggest outliers by net spend:")
print(customers[outliers].nlargest(8, 'Monetary')[FEATURES + ['AvgOrderValue', 'Country']].round(2))


# 4. Compare with K-Means: which segments do the outliers fall into?
k = choose_k(evaluate_k(scaled))
kmeans = fit_kmeans(scaled, k)
names = name_segments(pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES))
segments = pd.Series(kmeans.labels_).map(names)
print("\nWhere K-Means put the DBSCAN outliers:")
table = pd.crosstab(segments, np.where(outliers, 'outlier', 'normal'))
table['outlier_share'] = table['outlier'] / table.sum(axis=1)
print(table.round(3))

# Does removing the outliers change the K-Means segments?
clean_model = fit_kmeans(scaled[~outliers], k)
agreement = adjusted_rand_score(kmeans.labels_[~outliers], clean_model.labels_)
print(f"\nK-Means without the outliers vs with them (on the same {(~outliers).sum():,} customers): "
      f"Adjusted Rand Index = {agreement:.3f}")
clean_names = name_segments(pd.DataFrame(clean_model.cluster_centers_, columns=FEATURES))
print("Segments without the outliers:",
      dict(pd.Series(clean_model.labels_).map(clean_names).value_counts()))
print("How the original segments were redistributed:")
print(pd.crosstab(segments[~outliers].values, pd.Series(clean_model.labels_).map(clean_names).values,
                  rownames=['with outliers'], colnames=['without outliers']))
print(f"Silhouette with outliers: {silhouette_score(scaled, kmeans.labels_):.3f}, "
      f"without: {silhouette_score(scaled[~outliers], clean_model.labels_):.3f}")


# 5. The outliers on the PCA map
pca = PCA(n_components=2, random_state=42).fit(scaled)
points = pca.transform(scaled)
plt.figure(figsize=(9, 7))
plt.scatter(points[~outliers, 0], points[~outliers, 1], s=4, alpha=0.35, color='steelblue',
            label=f"Dense cluster ({(~outliers).sum():,})")
plt.scatter(points[outliers, 0], points[outliers, 1], s=28, color='red', edgecolor='black', linewidth=0.5,
            label=f"Outliers ({outliers.sum()})")
plt.xlabel(f"PC1: customer size ({pca.explained_variance_ratio_[0]:.0%} of variance)")
plt.ylabel(f"PC2: customer age ({pca.explained_variance_ratio_[1]:.0%} of variance)")
plt.title(f'DBSCAN Outliers on the PCA Map (eps = {eps:.2f}, min_samples = {min_samples})')
plt.legend(markerscale=2)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '18_dbscan_outliers_on_pca.png'))

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
