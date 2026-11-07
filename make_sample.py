# Create samples/sample_transactions.csv: real raw transactions for a few customers from every
# segment, so the app's "upload transactions" feature can be tried without the full dataset.
import os
import pandas as pd
import joblib
from load_data import load_data

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
SAMPLE_FILE = os.path.join(SAMPLE_DIR, "sample_transactions.csv")
CUSTOMERS_PER_SEGMENT = 8

bundle = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib"))
customers = bundle['customers']

# A few customers per segment (skip the very largest ones to keep the file small), plus a few outliers
small = customers[customers['Frequency'] <= 30]
chosen = small.groupby('Segment').sample(CUSTOMERS_PER_SEGMENT, random_state=42).index.tolist()
chosen += small[small['IsOutlier']].sample(4, random_state=42).index.tolist()

# RAW rows (cancellations, postage lines and duplicates included), so the upload shows the cleaning working
raw = load_data()
sample = raw[raw['Customer ID'].isin(chosen)].sort_values('InvoiceDate')

os.makedirs(SAMPLE_DIR, exist_ok=True)
sample.to_csv(SAMPLE_FILE, index=False)
print(f"Saved {len(sample):,} transaction lines for {sample['Customer ID'].nunique()} customers "
      f"to {SAMPLE_FILE} ({os.path.getsize(SAMPLE_FILE) / 1e3:.0f} KB)")
print("Their segments in the model:")
print(customers.loc[sorted(set(chosen)), 'Segment'].value_counts())
