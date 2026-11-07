# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from features import load_customer_features, FEATURES
from training import (evaluate_k, choose_k, fit_kmeans, name_segments, profile_segments,
                      stability, segment_palette, K_RANGE)

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

# Customers with transformed + scaled features (K-Means uses distances, so scaling is essential)
customers, scaled, scaler = load_customer_features()
print(f"{len(scaled):,} customers x {len(FEATURES)} features\n")


# 1. How many segments? Try k = 2 to 10 and score each
scores = evaluate_k(scaled)
k = choose_k(scores)
print("Scores for each k (silhouette: higher = better; Davies-Bouldin: lower = better):")
print(scores.round(3))
print(f"\nChosen k = {k}: the best silhouette within k = {K_RANGE.start} to {K_RANGE.stop - 1}")
print(f"(k = {int(scores['silhouette'].idxmax())} scores highest overall, but k = 2 only splits "
      "'good customers' from 'the rest', too coarse to act on)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, column, title in [(axes[0], 'inertia', 'Inertia (look for the elbow)'),
                          (axes[1], 'silhouette', 'Silhouette (higher = better)'),
                          (axes[2], 'davies_bouldin', 'Davies-Bouldin (lower = better)')]:
    ax.plot(scores.index, scores[column], marker='o', color='steelblue')
    ax.plot(k, scores.loc[k, column], marker='o', markersize=14, color='red', fillstyle='none', markeredgewidth=2)
    ax.axvspan(K_RANGE.start - 0.5, K_RANGE.stop - 0.5, color='green', alpha=0.06)
    ax.set_xlabel('k (number of segments)')
    ax.set_title(title)
    ax.set_xticks(scores.index)
plt.suptitle(f'Choosing k: chosen k = {k} (red circle), allowed range shaded green')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_choose_k.png'))


# 2. Fit the final K-Means and name the segments from their centres
model = fit_kmeans(scaled, k)
labels = model.labels_
centers = pd.DataFrame(model.cluster_centers_, columns=FEATURES)
names = name_segments(centers)

print("\nSegment centres (scaled: 0 = average customer, +1 = one standard deviation above):")
print(centers.rename(index=names).round(2))


# 3. Profile: what each segment looks like in real units
profile = profile_segments(customers, labels, names)
print("\nSegment profiles (medians in real units):")
print(profile.round({'share_of_customers': 3, 'share_of_revenue': 3, 'share_uk': 3,
                     'Recency': 0, 'Frequency': 0, 'Monetary': 0, 'Tenure': 0,
                     'ProductVariety': 0, 'ReturnRate': 3, 'AvgOrderValue': 0}).to_string())

for segment, row in profile.iterrows():
    print(f"  {segment:20s} {row['share_of_customers']:5.1%} of customers bring {row['share_of_revenue']:5.1%} of revenue")


# 4. Snake plot: each segment's centre across the features
order = profile.index.tolist()
palette = segment_palette(order)
plt.figure(figsize=(10, 5))
for segment in order:
    number = [s for s, n in names.items() if n == segment][0]
    plt.plot(FEATURES, centers.loc[number], marker='o', linewidth=2.5, color=palette[segment],
             label=f"{segment} ({profile.loc[segment, 'customers']:,})")
plt.axhline(0, color='gray', linestyle=':')
plt.ylabel('Segment centre (scaled; 0 = average customer)')
plt.title('Snake Plot: How Each Segment Differs From the Average Customer')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '12_snake_plot.png'))


# 5. Silhouette per customer: how well does each customer fit its own segment?
# (+1 = clearly in the right segment, 0 = on the border, negative = closer to another segment)
sample_scores = silhouette_samples(scaled, labels)
print("\nAverage silhouette per segment:")
for number, name in names.items():
    in_segment = sample_scores[labels == number]
    print(f"  {name:20s} {in_segment.mean():.3f}  ({(in_segment < 0).mean():.0%} of its customers are negative)")

plt.figure(figsize=(9, 6))
y = 0
for segment in order:
    number = [s for s, n in names.items() if n == segment][0]
    values = np.sort(sample_scores[labels == number])
    plt.fill_betweenx(np.arange(y, y + len(values)), 0, values, color=palette[segment], alpha=0.8)
    plt.text(-0.05, y + len(values) / 2, segment, ha='right', va='center')
    y += len(values) + 60
plt.axvline(sample_scores.mean(), color='red', linestyle='--', label=f'Average: {sample_scores.mean():.3f}')
plt.xlabel('Silhouette score of each customer')
plt.yticks([])
plt.xlim(-0.5, 1)
plt.title('Silhouette Plot: How Clearly Each Customer Belongs to Its Segment')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '13_silhouette_plot.png'))


# 6. The segments on the PCA map from step 4
pca = PCA(n_components=2, random_state=42).fit(scaled)
points = pca.transform(scaled)
center_points = pca.transform(centers)
plt.figure(figsize=(9, 7))
for segment in order:
    number = [s for s, n in names.items() if n == segment][0]
    mask = labels == number
    plt.scatter(points[mask, 0], points[mask, 1], s=5, alpha=0.5, color=palette[segment], label=segment)
    plt.scatter(*center_points[number], s=250, color=palette[segment], edgecolor='black', linewidth=2, marker='X')
plt.xlabel(f"PC1: customer size ({pca.explained_variance_ratio_[0]:.0%} of variance)")
plt.ylabel(f"PC2: customer age ({pca.explained_variance_ratio_[1]:.0%} of variance)")
plt.title('K-Means Segments on the PCA Map (X = segment centre)')
plt.legend(markerscale=4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '14_segments_on_pca.png'))


# 7. Stability: do we get the same segments from different samples of customers?
ari = stability(scaled, k)
print(f"\nStability over {len(ari)} random 80% samples (Adjusted Rand Index, 1 = identical segments):")
print(f"  mean {ari.mean():.3f}, worst {ari.min():.3f}, best {ari.max():.3f}")

# Compare: a k that scored worse should usually be less stable
other_k = 6 if k != 6 else 5
other = stability(scaled, other_k)
print(f"  for comparison, k = {other_k}: mean {other.mean():.3f}, worst {other.min():.3f}")

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
