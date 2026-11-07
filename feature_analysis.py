# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data, clean_data
from features import build_customer_features, make_scaler, FEATURES, LOG_FEATURES, SQRT_FEATURES

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Transactions -> one row per customer
df = clean_data(load_data())
customers = build_customer_features(df)
print(f"Customer table: {customers.shape[0]:,} customers x {len(FEATURES)} clustering features")
print(f"(dropped {df['Customer ID'].nunique() - len(customers)} customers whose net spend is under GBP 1)\n")
print(customers.head())

print("\nFeature summary (raw values):")
print(customers[FEATURES].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).round(1).T)

# AvgOrderValue and Country are kept for describing segments later, but NOT used for clustering
# (see PROFILE_COLUMNS in features.py for why)
print(f"\nUK customers: {customers['IsUK'].mean():.1%} (kept for profiling, not used for clustering)")
print(f"Median order value: GBP {customers['AvgOrderValue'].median():.0f} (kept for profiling, not used for clustering)")


# 1. Skew before and after the transforms
scaler = make_scaler()
scaled = scaler.fit_transform(customers[FEATURES])[FEATURES]  # same column order as FEATURES

transform_used = {f: 'log1p' if f in LOG_FEATURES else 'sqrt' if f in SQRT_FEATURES else 'none' for f in FEATURES}
skew = pd.DataFrame({
    'transform': pd.Series(transform_used),
    'skew before': customers[FEATURES].skew(),
    'skew after': scaled.skew()
})
print("\nSkewness (0 = symmetric; above 1 = a long tail of big values):")
print(skew.round(2))

print("\nAfter scaling every feature has mean 0 and standard deviation 1:")
print(scaled.agg(['mean', 'std']).round(2))


# 2. Distributions before (top row) and after (bottom row) the transforms
fig, axes = plt.subplots(2, len(FEATURES), figsize=(22, 6.5))
for i, feature in enumerate(FEATURES):
    axes[0, i].hist(customers[feature], bins=40, color='lightcoral')
    axes[0, i].set_title(f"{feature}\nskew {skew.loc[feature, 'skew before']:.1f}")
    axes[1, i].hist(scaled[feature], bins=40, color='steelblue')
    axes[1, i].set_title(f"after {transform_used[feature]} + scaling\nskew {skew.loc[feature, 'skew after']:.1f}")
axes[0, 0].set_ylabel('Raw values')
axes[1, 0].set_ylabel('Transformed + scaled')
plt.suptitle('Customer Features Before and After Transforming')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '5_feature_transforms.png'))


# 3. Correlation: features that move together count twice in every distance
corr = scaled.corr()
print("\nMost correlated feature pairs:")
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().sort_values(key=abs, ascending=False)
print(pairs.head(5).round(2))

plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation of Transformed Customer Features')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '6_feature_correlation.png'))


# 4. Relationships between the RFM features (a random 2,000 customers, to keep the plot readable)
sample = scaled.sample(2000, random_state=42)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (x, y) in zip(axes, [('Recency', 'Frequency'), ('Frequency', 'Monetary'), ('Frequency', 'ProductVariety')]):
    ax.scatter(sample[x], sample[y], s=6, alpha=0.4, color='steelblue')
    ax.set_xlabel(f"{x} (transformed)")
    ax.set_ylabel(f"{y} (transformed)")
    ax.set_title(f"{x} vs {y}: r = {corr.loc[x, y]:.2f}")
plt.suptitle('How the RFM Features Relate (2,000 random customers)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '7_rfm_relationships.png'))

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
