# Turn transactions into one row per customer, and scale the features for clustering.
# Used by the analysis scripts, the training pipeline and the app (for uploaded CSVs).
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# The features the clustering uses (a human decision: they define what "similar customers" means)
FEATURES = [
    'Recency',         # days since the last purchase (lower = more recent)
    'Frequency',       # number of separate orders
    'Monetary',        # net spend in GBP: purchases minus cancellations
    'Tenure',          # days since the first purchase (how long they have been a customer)
    'ProductVariety',  # number of different products bought
    'ReturnRate'       # share of purchase value that was cancelled (0 to 1)
]

# Columns kept to DESCRIBE segments, but not used for clustering:
# - AvgOrderValue = Monetary / Frequency, so it is almost fully redundant (pca_analysis.py
#   finds R^2 = 0.98 from those two); as a feature it would give money extra weight in every distance
# - Country / IsUK: a 0/1 feature makes K-Means split customers into "UK" and "not UK"
PROFILE_COLUMNS = ['AvgOrderValue', 'Country', 'IsUK']

# Heavily right-skewed features get log1p (log of 1 + x) before scaling, so a few huge
# customers don't dominate every distance. Chosen from the skew check in feature_analysis.py.
LOG_FEATURES = ['Recency', 'Frequency', 'Monetary', 'ProductVariety']
# ReturnRate is 0 for most customers and log1p barely changes small fractions, so it gets a square root
SQRT_FEATURES = ['ReturnRate']
# Tenure is left-skewed (most customers have been around a while), so it is only scaled

# Customers below this net spend cancelled (almost) everything; < 1 also catches float leftovers like 1e-13
MIN_NET_SPEND = 1.0


def build_customer_features(df, snapshot_date=None):
    """One row per customer from cleaned transactions (see clean_data in load_data.py).

    snapshot_date is "today" for Recency and Tenure. By default it is the day after the
    last invoice, as if we were analysing the data the morning after it ends.
    """
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max().normalize() + pd.Timedelta(days=1)

    purchases = df[~df['IsCancellation']]
    cancellations = df[df['IsCancellation']]
    by_customer = purchases.groupby('Customer ID')

    customers = pd.DataFrame({
        'Recency': (snapshot_date - by_customer['InvoiceDate'].max()).dt.days,
        'Frequency': by_customer['Invoice'].nunique(),
        'PurchaseValue': by_customer['Revenue'].sum(),
        'Tenure': (snapshot_date - by_customer['InvoiceDate'].min()).dt.days,
        'ProductVariety': by_customer['StockCode'].nunique(),
        # A few customers appear under two countries: use the one they buy from most
        'Country': by_customer['Country'].agg(lambda countries: countries.mode()[0])
    })

    # Cancellations are negative revenue; customers who never cancelled get 0
    cancelled = (-cancellations.groupby('Customer ID')['Revenue'].sum()).reindex(customers.index, fill_value=0)
    customers['Monetary'] = customers['PurchaseValue'] - cancelled
    # Can exceed 1 when someone cancels an order placed before the data starts, so cap it
    customers['ReturnRate'] = (cancelled / customers['PurchaseValue']).clip(0, 1)
    customers['AvgOrderValue'] = customers['Monetary'] / customers['Frequency']

    # Customers who cancelled everything aren't really customers.
    # Customers with only cancellations never appear in `purchases`, so they are already gone.
    customers = customers[customers['Monetary'] >= MIN_NET_SPEND]

    customers['IsUK'] = (customers['Country'] == 'United Kingdom').astype(int)
    return customers[FEATURES + PROFILE_COLUMNS]


def make_scaler():
    """Transform the skewed features (log1p / square root), then standardize everything (mean 0, std 1).

    Fit it on the customers you cluster; the same fitted scaler must transform any new customer.
    """
    def then_scale(function):
        # np.log1p and np.sqrt are named numpy functions, so the fitted scaler can be saved with joblib
        return Pipeline([
            ('transform', FunctionTransformer(function, feature_names_out='one-to-one')),
            ('scale', StandardScaler())
        ])

    plain_features = [f for f in FEATURES if f not in LOG_FEATURES + SQRT_FEATURES]
    return ColumnTransformer([
        ('log', then_scale(np.log1p), LOG_FEATURES),
        ('sqrt', then_scale(np.sqrt), SQRT_FEATURES),
        ('scale', StandardScaler(), plain_features)
    ], verbose_feature_names_out=False).set_output(transform='pandas')


def load_customer_features():
    """Shortcut for the analysis scripts: load -> clean -> customer table -> fitted scaler.
    Returns (customers, scaled features in FEATURES order, fitted scaler)."""
    from load_data import load_data, clean_data  # imported here so the app can use this file without the raw data
    customers = build_customer_features(clean_data(load_data()))
    scaler = make_scaler()
    scaled = scaler.fit_transform(customers[FEATURES])[FEATURES]
    return customers, scaled, scaler
