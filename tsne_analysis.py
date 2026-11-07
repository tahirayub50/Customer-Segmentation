# Import necessary libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from features import load_customer_features, FEATURES
from training import (evaluate_k, choose_k, fit_kmeans, name_segments, fit_tsne,
                      dbscan_min_samples, k_distances, find_eps, fit_dbscan, segment_palette,
                      TSNE_PERPLEXITY)

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Customers with transformed + scaled features
customers, scaled, scaler = load_customer_features()
print(f"{len(scaled):,} customers x {len(FEATURES)} features\n")

# Segments (step 5) and outliers (step 6), to colour the maps
k = choose_k(evaluate_k(scaled))
kmeans = fit_kmeans(scaled, k)
names = name_segments(pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES))
segments = pd.Series(kmeans.labels_).map(names).values
min_samples = dbscan_min_samples()
outliers = fit_dbscan(scaled, find_eps(k_distances(scaled, min_samples)), min_samples).labels_ == -1
order = sorted(set(segments))
palette = segment_palette(order)


def draw(ax, points, title):
    """Scatter the customers coloured by segment, with DBSCAN outliers ringed in black."""
    for segment in order:
        mask = segments == segment
        ax.scatter(points[mask, 0], points[mask, 1], s=3, alpha=0.5, color=palette[segment], label=segment)
    ax.scatter(points[outliers, 0], points[outliers, 1], s=22, facecolor='none', edgecolor='black',
               linewidth=0.8, label='DBSCAN outlier')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


# 1. PCA vs t-SNE on the same customers
pca_points = PCA(n_components=2, random_state=42).fit_transform(scaled)
start = time.time()
tsne_points = fit_tsne(scaled)
print(f"t-SNE (perplexity {TSNE_PERPLEXITY}) took {time.time() - start:.0f} s")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
draw(axes[0], pca_points, 'PCA: a linear projection (keeps global distances)')
draw(axes[1], tsne_points, f't-SNE, perplexity {TSNE_PERPLEXITY}: keeps local neighbourhoods')
axes[1].legend(markerscale=4, loc='best')
plt.suptitle('The Same 5,832 Customers, Two Ways to Draw Them in 2D')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '19_pca_vs_tsne.png'))


# 2. How well does each map keep each customer's real neighbours?
# Trustworthiness: 1 = the neighbours you see on the map are the real neighbours.
data = np.asarray(scaled)
print("\nTrustworthiness (are the neighbours on the map real neighbours? 1 = perfect):")
for name, points in [('PCA', pca_points), ('t-SNE', tsne_points)]:
    print(f"  {name:6s} {trustworthiness(data, points, n_neighbors=10):.3f}")


# 3. But t-SNE distorts DISTANCES. Compare the distances between segment centres
# in the real feature space with the distances on each map.
def segment_centres(points):
    return np.array([points[segments == s].mean(axis=0) for s in order])

real = pdist(segment_centres(data))
print("\nDo distances BETWEEN segments survive? (rank correlation with real distances, 1 = same order)")
for name, points in [('PCA', pca_points), ('t-SNE', tsne_points)]:
    rho = spearmanr(real, pdist(segment_centres(points))).statistic
    print(f"  {name:6s} {rho:.2f}")

# And segment SIZES: how spread out is each segment, relative to the whole map?
def spread(points, mask):
    return np.linalg.norm(points[mask] - points[mask].mean(axis=0), axis=1).mean()

print("\nHow spread out each segment looks (relative to the whole map; real = in the 6 features):")
rows = []
for segment in order:
    mask = segments == segment
    rows.append({'segment': segment,
                 'real': spread(data, mask) / spread(data, np.ones(len(data), bool)),
                 'PCA': spread(pca_points, mask) / spread(pca_points, np.ones(len(data), bool)),
                 't-SNE': spread(tsne_points, mask) / spread(tsne_points, np.ones(len(data), bool))})
print(pd.DataFrame(rows).set_index('segment').round(2))


# 4. Perplexity changes the picture a lot
perplexities = [5, 30, 50]
fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 6.5))
for ax, perplexity in zip(axes, perplexities):
    points = tsne_points if perplexity == TSNE_PERPLEXITY else fit_tsne(scaled, perplexity=perplexity)
    draw(ax, points, f'perplexity = {perplexity}')
axes[-1].legend(markerscale=4, loc='best')
plt.suptitle('t-SNE With Different Perplexity: Each Map Keeps a Different Size of Neighbourhood')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '20_tsne_perplexity.png'))


# 5. Different random seeds: same neighbourhoods, different layout
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
seed_maps = {}
for ax, seed in zip(axes, [42, 7, 2024]):
    points = tsne_points if seed == 42 else fit_tsne(scaled, random_state=seed)
    seed_maps[seed] = points
    draw(ax, points, f'random_state = {seed}')
axes[-1].legend(markerscale=4, loc='best')
plt.suptitle('t-SNE With Different Random Seeds: the Layout Moves, the Neighbourhoods Stay')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '21_tsne_seeds.png'))

print("\nTrustworthiness for each seed:")
for seed, points in seed_maps.items():
    print(f"  random_state {seed:4d}: {trustworthiness(data, points, n_neighbors=10):.3f}")


# 6. No transform(): t-SNE can't place a new customer on an existing map
print(f"\nDoes TSNE have transform()? {hasattr(TSNE(), 'transform')}. "
      "Only fit_transform(), which builds a new map from scratch.")
print("So the app shows the t-SNE map for existing customers only, and uses PCA (which has transform())")
print("to place a new customer on a map.")

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
