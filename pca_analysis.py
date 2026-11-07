# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from features import load_customer_features, FEATURES

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Customers with transformed + scaled features (PCA needs scaled data: it looks for directions of
# largest variance, so an unscaled feature in big units would win just because of its units)
customers, scaled, scaler = load_customer_features()
print(f"{len(scaled):,} customers x {len(FEATURES)} features\n")


# 1. How much information does each principal component keep?
pca = PCA(random_state=42)
components = pca.fit_transform(scaled)
pc_names = [f"PC{i + 1}" for i in range(len(FEATURES))]

variance = pd.DataFrame({
    'explained': pca.explained_variance_ratio_,
    'cumulative': np.cumsum(pca.explained_variance_ratio_)
}, index=pc_names)
print("Explained variance per component:")
print((variance * 100).round(1).astype(str) + '%')

for target in (0.80, 0.90, 0.95):
    n = int(np.argmax(variance['cumulative'] >= target)) + 1
    print(f"Components needed for {target:.0%} of the variance: {n} of {len(FEATURES)}")

plt.figure(figsize=(8, 4.5))
plt.bar(pc_names, variance['explained'] * 100, color='steelblue', label='Each component')
plt.plot(pc_names, variance['cumulative'] * 100, color='darkorange', marker='o', label='Cumulative')
for i, value in enumerate(variance['cumulative']):
    plt.annotate(f"{value:.0%}", (i, value * 100), textcoords='offset points', xytext=(0, 8), ha='center')
plt.axhline(90, color='gray', linestyle=':', label='90%')
plt.ylim(0, 110)
plt.ylabel('Variance explained (%)')
plt.title('PCA Scree Plot: How Much Each Component Keeps')
plt.legend(loc='center right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '8_pca_scree.png'))


# 2. Loadings: what does each component mean?
# A loading is the weight of an original feature in a component. Big positive or negative
# weights tell you what the component measures.
loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=pc_names)
print("\nLoadings of the first 3 components:")
print(loadings[pc_names[:3]].round(2))

for pc in pc_names[:3]:
    top = loadings[pc].abs().sort_values(ascending=False).index[:3]
    described = ', '.join(f"{f} ({loadings.loc[f, pc]:+.2f})" for f in top)
    print(f"{pc} is driven mostly by: {described}")

plt.figure(figsize=(8, 5.5))
sns.heatmap(loadings[pc_names[:4]], annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('PCA Loadings: How Much Each Feature Counts in Each Component')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '9_pca_loadings.png'))


# 3. The 2D map: every customer placed by PC1 and PC2, with the feature directions drawn on top
# (a "biplot"). Colour shows real values, to check what the axes mean.
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
for ax, feature, cmap in [(axes[0], 'Monetary', 'viridis'), (axes[1], 'Recency', 'plasma')]:
    points = ax.scatter(components[:, 0], components[:, 1], c=np.log10(customers[feature] + 1),
                        cmap=cmap, s=5, alpha=0.6)
    plt.colorbar(points, ax=ax, label=f"log10({feature} + 1)")

    # Arrows: where each original feature points on this map
    scale = np.abs(components[:, :2]).max() * 0.8
    for f in FEATURES:
        dx, dy = loadings.loc[f, 'PC1'] * scale, loadings.loc[f, 'PC2'] * scale
        ax.arrow(0, 0, dx, dy, color='black', width=0.02, head_width=0.15, alpha=0.8)
        ax.text(dx * 1.12, dy * 1.12, f, fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_xlabel(f"PC1 ({variance.loc['PC1', 'explained']:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({variance.loc['PC2', 'explained']:.0%} of variance)")
    ax.set_title(f"Customers on the PCA map, coloured by {feature}")
plt.suptitle(f"PCA Biplot: {variance.loc['PC2', 'cumulative']:.0%} of the information in 2 dimensions")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_pca_biplot.png'))


# 4. What does dropping components cost? Rebuild the data from the first n components
# and measure how far each customer moves (reconstruction error)
print("\nReconstruction error when keeping only n components (0 = perfect):")
for n in (2, 3, 4, len(FEATURES)):
    reduced = PCA(n_components=n, random_state=42).fit(scaled)
    rebuilt = reduced.inverse_transform(reduced.transform(scaled))
    error = ((scaled.values - rebuilt) ** 2).mean()
    print(f"  {n} components: mean squared error = {error:.3f}")

# PCA is just a rotation: with ALL components kept, distances between customers don't change
original = np.linalg.norm(scaled.values[0] - scaled.values[1])
rotated = np.linalg.norm(components[0] - components[1])
print(f"\nDistance between the first two customers: {original:.4f} in feature space, "
      f"{rotated:.4f} in PCA space (all components)")


# 5. Why AvgOrderValue is not a clustering feature.
# With it included, the last component kept only 0.04% of the variance: a component with
# (almost) no variance means one feature is a combination of the others.
# AvgOrderValue = Monetary / Frequency, so after the log: log(AOV) ~ log(Monetary) - log(Frequency)
from sklearn.linear_model import LinearRegression
logged = np.log1p(customers[['Frequency', 'Monetary', 'AvgOrderValue']])
check = LinearRegression().fit(logged[['Frequency', 'Monetary']], logged['AvgOrderValue'])
r2 = check.score(logged[['Frequency', 'Monetary']], logged['AvgOrderValue'])
print(f"\nlog AvgOrderValue predicted from log Frequency + log Monetary: R^2 = {r2:.3f}")
print("-> it adds almost no new information, so it is used to describe segments, not to cluster")
print(f"Smallest component now: {pc_names[-1]} keeps {variance.loc[pc_names[-1], 'explained']:.1%} of the variance")

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
