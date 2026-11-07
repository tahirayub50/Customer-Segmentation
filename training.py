# The clustering procedure, as reusable functions.
# The analysis scripts use these to explain each step; train_final_model.py (step 8) will use
# them to rebuild everything automatically, so nothing here is hardcoded from one run.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from features import FEATURES

# Decisions made by people, not learned from data
K_RANGE = range(3, 9)   # k = 2 only splits "good" from "the rest", too coarse to act on; above 8 is hard to use
RANDOM_STATE = 42
N_INIT = 10             # K-Means restarts from 10 random starting points and keeps the best
MIN_SAMPLES_PER_FEATURE = 2  # DBSCAN rule of thumb: min_samples = 2 x number of features

# Features that together measure "customer size" (they are strongly correlated, see feature_analysis.py)
VALUE_FEATURES = ['Frequency', 'Monetary', 'ProductVariety']


def fit_kmeans(scaled, k, random_state=RANDOM_STATE):
    """K-Means with k clusters on the scaled features."""
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=random_state).fit(scaled)


def evaluate_k(scaled, ks=range(2, 11)):
    """Score each k three ways. Inertia always falls as k grows (more centres = closer points),
    so it needs the elbow; silhouette (higher = better) and Davies-Bouldin (lower = better) don't."""
    rows = []
    for k in ks:
        model = fit_kmeans(scaled, k)
        rows.append({
            'k': k,
            'inertia': model.inertia_,
            'silhouette': silhouette_score(scaled, model.labels_),
            'davies_bouldin': davies_bouldin_score(scaled, model.labels_),
            'smallest_segment': int(np.bincount(model.labels_).min())
        })
    return pd.DataFrame(rows).set_index('k')


def choose_k(scores, k_range=K_RANGE):
    """The k with the best silhouette score within the allowed range."""
    allowed = scores.loc[[k for k in k_range if k in scores.index]]
    return int(allowed['silhouette'].idxmax())


def name_segments(centers):
    """Give each segment a readable name from its centre (in scaled units: 0 = average customer,
    +1 = one standard deviation above). Returns {segment number: name}.

    The rules use only the centres, so they still work when the data or k changes.
    """
    names = {}
    for segment, center in centers.iterrows():
        value = center[VALUE_FEATURES].mean()   # how big a customer is
        if center['ReturnRate'] > 1.0:
            name = 'Frequent returners'
        elif value > 0.5:
            name = 'Champions' if center['Recency'] < 0 else 'At risk'      # valuable, but slipping away
        elif center['Tenure'] < -0.75:
            name = 'New customers'
        elif center['Recency'] > 0.5:
            name = 'Lapsed'
        elif center['Recency'] <= 0:
            name = 'Promising'
        else:
            name = 'Needs attention'
        names[segment] = name

    # Two segments with the same name get "(higher value)" / "(lower value)" style suffixes
    for name in set(names.values()):
        same = [s for s in names if names[s] == name]
        if len(same) > 1:
            ranked = sorted(same, key=lambda s: -centers.loc[s, VALUE_FEATURES].mean())
            for rank, segment in enumerate(ranked, start=1):
                names[segment] = f"{name} {rank}"
    return names


def profile_segments(customers, labels, names):
    """One row per segment: size, share of revenue and the median of each feature (in real units)."""
    data = customers.assign(Segment=pd.Series(labels, index=customers.index).map(names))
    profile = data.groupby('Segment')[FEATURES + ['AvgOrderValue']].median()
    profile.insert(0, 'customers', data['Segment'].value_counts())
    profile.insert(1, 'share_of_customers', profile['customers'] / len(data))
    profile.insert(2, 'share_of_revenue', data.groupby('Segment')['Monetary'].sum() / data['Monetary'].sum())
    profile['share_uk'] = data.groupby('Segment')['IsUK'].mean()
    return profile.sort_values('share_of_revenue', ascending=False)


def stability(scaled, k, n_runs=20, sample_share=0.8):
    """Are the segments real or an accident of this sample? Fit K-Means on random 80% samples,
    label EVERY customer with each model, and compare with the full model using the
    Adjusted Rand Index (1 = identical segments, 0 = no better than random)."""
    reference = fit_kmeans(scaled, k).labels_
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for run in range(n_runs):
        sample = scaled.sample(frac=sample_share, random_state=int(rng.integers(1_000_000)))
        model = fit_kmeans(sample, k, random_state=run)
        scores.append(adjusted_rand_score(reference, model.predict(scaled)))
    return np.array(scores)


def dbscan_min_samples(n_features=len(FEATURES)):
    """How many neighbours (including the point itself) make a point a 'core' point."""
    return MIN_SAMPLES_PER_FEATURE * n_features


def k_distances(scaled, min_samples):
    """Distance from every customer to its min_samples-th nearest neighbour, sorted.
    NearestNeighbors counts the point itself as its first neighbour, just like DBSCAN's
    min_samples does, so the two use the same definition."""
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(scaled).kneighbors(scaled)
    return np.sort(distances[:, -1])


def find_eps(sorted_distances):
    """The 'knee' of the k-distance curve: the point furthest below the straight line from the
    first to the last distance. Before the knee distances grow slowly (dense areas); after it
    they shoot up (isolated customers), so it is a natural cut-off for 'close enough'."""
    x = np.arange(len(sorted_distances))
    first, last = sorted_distances[0], sorted_distances[-1]
    line = first + (last - first) * x / x[-1]
    return float(sorted_distances[np.argmax(line - sorted_distances)])


def fit_dbscan(scaled, eps, min_samples):
    """DBSCAN on the scaled features. Label -1 means noise (an outlier)."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit(scaled)


def outlier_flags(model, scaled):
    """DBSCAN has no predict(), so decide for ANY customers (including new ones) the way DBSCAN does:
    a customer is NOT an outlier when they are within eps of a core point. On the training customers
    this reproduces DBSCAN's own noise labels exactly."""
    core_points = model.components_   # a plain array, so compare with a plain array too
    distance, _ = NearestNeighbors(n_neighbors=1).fit(core_points).kneighbors(np.asarray(scaled))
    return distance[:, 0] > model.eps


TSNE_PERPLEXITY = 30   # roughly "how many neighbours each point pays attention to"; 30 is the usual default


def fit_tsne(scaled, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_STATE):
    """2D t-SNE map of the customers. Starting from the PCA layout (init='pca') makes the result
    more stable between runs than a random start. Note: t-SNE has no transform() for new points;
    the map exists only for the customers it was fitted on."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto',
                random_state=random_state).fit_transform(np.asarray(scaled))


def segment_palette(segment_names):
    """The same colour for the same segment name in every plot (and later in the app)."""
    import seaborn as sns
    ordered = sorted(segment_names)
    return dict(zip(ordered, sns.color_palette('tab10', len(ordered))))


def predict_customers(bundle, customers):
    """Segment, outlier flag and PCA map position for any customers (rows of the customer table),
    using a saved model bundle from train_final_model.py. Works for customers the model has
    never seen, which is exactly what t-SNE and DBSCAN can't do on their own:
      - K-Means: the nearest segment centre
      - DBSCAN:  outlier when no core point is within eps (outlier_flags)
      - PCA:     transform() places the customer on the existing map
    """
    scaled = bundle['scaler'].transform(customers[FEATURES])[FEATURES]
    distances = bundle['kmeans'].transform(scaled)   # distance to every segment centre
    names = bundle['segment_names']
    nearest = distances.argmin(axis=1)
    position = bundle['pca'].transform(scaled)

    result = pd.DataFrame({
        'Segment': [names[n] for n in nearest],
        'IsOutlier': outlier_flags(bundle['dbscan'], scaled),
        'PCA_x': position[:, 0],
        'PCA_y': position[:, 1]
    }, index=customers.index)
    for number, name in names.items():
        result[f'Distance to {name}'] = distances[:, number]
    return result
