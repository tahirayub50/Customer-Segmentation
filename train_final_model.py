# Rebuild the whole segmentation model from the current data and save it for the app.
# Nothing about the model is hardcoded: k, the segment names, DBSCAN's eps and both maps are
# recalculated on every run, so adding transactions to the data and rerunning updates everything.
import os
import sys
import time
import argparse
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from load_data import load_data, clean_data
from features import build_customer_features, make_scaler, FEATURES, PROFILE_COLUMNS
from training import (evaluate_k, choose_k, fit_kmeans, name_segments, profile_segments, stability,
                      dbscan_min_samples, k_distances, find_eps, fit_dbscan, outlier_flags,
                      fit_tsne, predict_customers, K_RANGE, RANDOM_STATE)

# Where the trained model is saved (the Flask app loads it from here)
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

# Guard: if fewer customers than this keep the same segments as in the saved model,
# the segments changed a lot (e.g. marketing lists built on them would change), so ask for --force
MIN_AGREEMENT = 0.5

parser = argparse.ArgumentParser(description="Rebuild the customer segmentation model from the current data.")
parser.add_argument('--force', action='store_true',
                    help="save the new model even if its segments differ a lot from the saved model's")
args = parser.parse_args()
started = time.time()


# Step 1: Data -> one row per customer -> scaled features
transactions = clean_data(load_data())
customers = build_customer_features(transactions)
scaler = make_scaler()
scaled = scaler.fit_transform(customers[FEATURES])[FEATURES]
print(f"Step 1: {len(transactions):,} clean transaction lines -> {len(customers):,} customers "
      f"({transactions['InvoiceDate'].min():%d %b %Y} to {transactions['InvoiceDate'].max():%d %b %Y})")


# Step 2: K-Means. Choose k, fit, name and profile the segments
scores = evaluate_k(scaled)
k = choose_k(scores)
kmeans = fit_kmeans(scaled, k)
names = name_segments(pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES))
profile = profile_segments(customers, kmeans.labels_, names)
ari = stability(scaled, k)
print(f"\nStep 2: K-Means with k = {k} (best silhouette within {K_RANGE.start}-{K_RANGE.stop - 1}: "
      f"{scores.loc[k, 'silhouette']:.3f}); stability {ari.mean():.3f} (worst {ari.min():.3f})")
for segment, row in profile.iterrows():
    print(f"  {segment:20s} {int(row['customers']):>5,} customers ({row['share_of_customers']:5.1%}), "
          f"{row['share_of_revenue']:5.1%} of revenue")


# Step 3: DBSCAN as an outlier detector
min_samples = dbscan_min_samples()
eps = find_eps(k_distances(scaled, min_samples))
dbscan = fit_dbscan(scaled, eps, min_samples)
outliers = dbscan.labels_ == -1
assert (outlier_flags(dbscan, scaled) == outliers).all(), "outlier rule must match DBSCAN on its own data"
print(f"\nStep 3: DBSCAN with min_samples = {min_samples}, eps = {eps:.3f} (knee): "
      f"{outliers.sum()} outliers ({outliers.mean():.1%} of customers, "
      f"{customers.loc[outliers, 'Monetary'].sum() / customers['Monetary'].sum():.1%} of revenue)")


# Step 4: The two maps. PCA can place new customers; t-SNE is only for the existing ones
pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaled)
print(f"\nStep 4: PCA map keeps {pca.explained_variance_ratio_.sum():.0%} of the variance; building t-SNE map...")
tsne_points = fit_tsne(scaled)


# Step 5: Everything the app needs about the existing customers, in one table
bundle = {
    'scaler': scaler,
    'kmeans': kmeans,
    'dbscan': dbscan,
    'pca': pca,
    'k': k,
    'segment_names': names,
    'k_scores': scores,
    'profile': profile,
    'stability': {'mean': float(ari.mean()), 'worst': float(ari.min())},
    'silhouette': float(silhouette_score(scaled, kmeans.labels_)),
    'dbscan_params': {'eps': eps, 'min_samples': min_samples},
    'pca_variance': pca.explained_variance_ratio_.tolist(),
    'pca_loadings': pd.DataFrame(pca.components_.T, index=FEATURES, columns=['PC1', 'PC2']),
    'features': FEATURES,
    'n_customers': len(customers),
    'data_range': (transactions['InvoiceDate'].min().isoformat(), transactions['InvoiceDate'].max().isoformat()),
    'trained_at': datetime.now().isoformat(timespec='seconds')
}

customer_map = customers[FEATURES + PROFILE_COLUMNS].copy()
customer_map = customer_map.join(predict_customers(bundle, customers)[['Segment', 'IsOutlier', 'PCA_x', 'PCA_y']])
customer_map['TSNE_x'], customer_map['TSNE_y'] = tsne_points[:, 0], tsne_points[:, 1]
# predict_customers must give the training customers their training segment and outlier flag
assert (customer_map['Segment'] == pd.Series(kmeans.labels_, index=customers.index).map(names)).all()
assert (customer_map['IsOutlier'] == outliers).all()
bundle['customers'] = customer_map


# Step 6: Compare with the model currently in use
print("\nStep 6: comparing with the current model...")
old = None
if os.path.exists(MODEL_FILE):
    try:
        old = joblib.load(MODEL_FILE)
    except Exception as error:
        print(f"  the current model.joblib can't be loaded ({error}), so it will be replaced")

if old is None:
    print("  no usable saved model, nothing to compare")
else:
    old_sizes = old['customers']['Segment'].value_counts()
    new_sizes = customer_map['Segment'].value_counts()
    print(f"  customers: {old['n_customers']:,} -> {len(customers):,}")
    print(f"  k: {old['k']} -> {k}; silhouette: {old['silhouette']:.3f} -> {bundle['silhouette']:.3f}")
    print(f"  DBSCAN eps: {old['dbscan_params']['eps']:.3f} -> {eps:.3f}; "
          f"outliers: {int(old['customers']['IsOutlier'].sum())} -> {int(outliers.sum())}")
    for segment in sorted(set(old_sizes.index) | set(new_sizes.index)):
        print(f"  {segment:20s} {int(old_sizes.get(segment, 0)):>5,} -> {int(new_sizes.get(segment, 0)):>5,}")

    # Do customers that are in both models keep their segment?
    common = customer_map.index.intersection(old['customers'].index)
    agreement = adjusted_rand_score(old['customers'].loc[common, 'Segment'], customer_map.loc[common, 'Segment'])
    moved = (old['customers'].loc[common, 'Segment'] != customer_map.loc[common, 'Segment']).mean()
    print(f"  {len(common):,} customers in both: agreement (ARI) {agreement:.3f}, {moved:.1%} changed segment")

    if agreement < MIN_AGREEMENT and not args.force:
        print(f"\nNOT SAVED: the segments changed a lot (agreement {agreement:.3f} < {MIN_AGREEMENT}).")
        print("Check the report above, then rerun with --force to replace the model anyway.")
        sys.exit(1)


# Step 7: Save
joblib.dump(bundle, MODEL_FILE)
size = os.path.getsize(MODEL_FILE) / 1e6
print(f"\nModel saved to: {MODEL_FILE} ({size:.1f} MB, {time.time() - started:.0f} s in total)")
