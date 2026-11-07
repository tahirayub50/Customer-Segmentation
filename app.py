# Import necessary libraries
import os
import re
import math
import joblib
import pandas as pd
from matplotlib.colors import to_hex
from flask import Flask, jsonify, render_template, request, send_file
from load_data import clean_data
from features import build_customer_features, FEATURES
from training import predict_customers, segment_palette

# Load the trained model once, when the server starts (run train_final_model.py first)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.joblib")
SAMPLE_FILE = os.path.join(BASE_DIR, "samples", "sample_transactions.csv")
if not os.path.exists(MODEL_FILE):
    raise SystemExit("model.joblib not found. Run 'python train_final_model.py' first.")

bundle = joblib.load(MODEL_FILE)
SEGMENTS = sorted(bundle['segment_names'].values())
COLORS = {name: to_hex(color) for name, color in segment_palette(SEGMENTS).items()}  # same colours as the plots

# What each segment means and what a shop could do about it. Names come from the naming rules in
# training.py, so a retrained model can produce any of these (numbered when two segments share a name).
SEGMENT_GUIDE = {
    'Champions': ("Buy often, spend the most and bought recently.",
                  "Reward and keep them: loyalty perks, early access to new products, ask for reviews."),
    'At risk': ("Used to be valuable, but haven't bought for a while.",
                "Reach out personally before they lapse: a check-in or a tailored offer."),
    'New customers': ("Made their first purchases recently.",
                      "Onboard them: a welcome series and an incentive for the second order."),
    'Promising': ("Recent buyers with average spend.",
                  "Recommend related products to grow their basket."),
    'Lapsed': ("Haven't bought for a long time, and spent little when they did.",
               "Try a time-limited win-back offer; stop costly mailings if they don't respond."),
    'Needs attention': ("Average customers who are starting to drift away.",
                        "Remind them with personalised recommendations."),
    'Frequent returners': ("Cancel or return a large share of what they buy.",
                           "Find out why (quality, sizing, expectations) and check what the returns cost."),
}

# Allowed ranges for the "score a customer" form (a little wider than the training data)
FEATURE_RANGES = {
    'Recency': (0, 3650),
    'Frequency': (1, 10_000),
    'Monetary': (1, 10_000_000),
    'Tenure': (0, 3650),
    'ProductVariety': (1, 50_000),
    'ReturnRate': (0, 1)
}

# Columns an uploaded transactions CSV must have (the same as the Online Retail II data)
REQUIRED_COLUMNS = ['Invoice', 'StockCode', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # uploads up to 50 MB


def guide(segment):
    """Description and suggested action; 'Champions 2' uses the 'Champions' text."""
    return SEGMENT_GUIDE.get(re.sub(r'\s\d+$', '', segment), ("", ""))


def validate(data):
    """Check one customer's features. Returns (customer, errors)."""
    if not isinstance(data, dict):
        return None, ["Send a JSON object with the customer's features."]
    customer, errors = {}, []
    for name, (low, high) in FEATURE_RANGES.items():
        value = data.get(name)
        # bool is a subclass of int in Python, so reject true/false explicitly
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"'{name}' is missing or not a number.")
        elif not low <= value <= high:
            errors.append(f"'{name}' must be between {low:,} and {high:,}.")
        else:
            customer[name] = float(value)
    if not errors and customer['Recency'] > customer['Tenure']:
        errors.append("'Recency' can't be larger than 'Tenure': the last purchase can't be before the first.")
    return customer, errors


def describe(predictions):
    """Turn predict_customers() rows into JSON-friendly dicts."""
    rows = []
    for customer_id, row in predictions.iterrows():
        distances = {s: float(row[f'Distance to {s}']) for s in bundle['segment_names'].values()}
        ranked = sorted(distances, key=distances.get)
        rows.append({
            'customer': customer_id,
            'segment': row['Segment'],
            'outlier': bool(row['IsOutlier']),
            'pca': [round(float(row['PCA_x']), 3), round(float(row['PCA_y']), 3)],
            'distances': {s: round(d, 3) for s, d in distances.items()},
            'runner_up': ranked[1] if len(ranked) > 1 else None
        })
    return rows


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/overview")
def overview():
    """The model and its segments, for the dashboard."""
    customers = bundle['customers']
    segments = []
    for name, row in bundle['profile'].iterrows():
        description, action = guide(name)
        segments.append({
            'name': name,
            'color': COLORS[name],
            'customers': int(row['customers']),
            'share_of_customers': float(row['share_of_customers']),
            'share_of_revenue': float(row['share_of_revenue']),
            'outliers': int(customers.loc[customers['Segment'] == name, 'IsOutlier'].sum()),
            'medians': {f: float(row[f]) for f in FEATURES + ['AvgOrderValue']},
            'share_uk': float(row['share_uk']),
            'description': description,
            'action': action
        })
    return jsonify({
        'segments': segments,
        'k': bundle['k'],
        'silhouette': bundle['silhouette'],
        'stability': bundle['stability'],
        'n_customers': bundle['n_customers'],
        'n_outliers': int(customers['IsOutlier'].sum()),
        'outlier_revenue_share': float(customers.loc[customers['IsOutlier'], 'Monetary'].sum() / customers['Monetary'].sum()),
        'dbscan': bundle['dbscan_params'],
        'pca_variance': bundle['pca_variance'],
        'pca_loadings': bundle['pca_loadings'].round(3).to_dict(),
        'features': FEATURES,
        'feature_ranges': FEATURE_RANGES,
        'data_range': bundle['data_range'],
        'trained_at': bundle['trained_at']
    })


@app.get("/api/map")
def customer_map():
    """Every existing customer with both map positions (compact arrays, about 5,800 points)."""
    c = bundle['customers']
    return jsonify({
        'segments': SEGMENTS,
        'colors': [COLORS[s] for s in SEGMENTS],
        'id': c.index.astype(int).tolist(),
        'segment': c['Segment'].map(SEGMENTS.index).tolist(),
        'outlier': c['IsOutlier'].astype(int).tolist(),
        'pca': c[['PCA_x', 'PCA_y']].round(3).values.tolist(),
        'tsne': c[['TSNE_x', 'TSNE_y']].round(2).values.tolist(),
        'recency': c['Recency'].astype(int).tolist(),
        'frequency': c['Frequency'].astype(int).tolist(),
        'monetary': c['Monetary'].round(0).astype(int).tolist(),
        'return_rate': c['ReturnRate'].round(3).tolist()
    })


@app.post("/api/predict")
def predict():
    """Segment, outlier flag and PCA map position for one customer described by their 6 features."""
    customer, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify({'errors': errors}), 400
    result = describe(predict_customers(bundle, pd.DataFrame([customer])))[0]
    result['description'], result['action'] = guide(result['segment'])
    return jsonify(result)


@app.post("/api/predict-batch")
def predict_batch():
    """Upload raw transactions (CSV, same columns as Online Retail II) and segment every customer in it."""
    upload = request.files.get('file')
    if upload is None or not upload.filename:
        return jsonify({'errors': ["Attach a CSV file in the 'file' field."]}), 400

    try:
        df = pd.read_csv(upload, dtype={'Invoice': str, 'StockCode': str, 'Description': str, 'Country': str})
    except Exception as error:
        return jsonify({'errors': [f"Couldn't read the file as CSV: {error}"]}), 400

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return jsonify({'errors': [f"Missing columns: {', '.join(missing)}. "
                                   f"Expected: {', '.join(REQUIRED_COLUMNS)} (Description is optional)."]}), 400

    rows_read = len(df)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    for col in ['Quantity', 'Price', 'Customer ID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    unreadable = int(df[['InvoiceDate', 'Quantity', 'Price']].isna().any(axis=1).sum())
    df = df.dropna(subset=['InvoiceDate', 'Quantity', 'Price'])
    df['Quantity'] = df['Quantity'].astype(int)
    df['Invoice'] = df['Invoice'].fillna('').astype(str)
    df['StockCode'] = df['StockCode'].fillna('').astype(str)

    clean = clean_data(df)
    if clean.empty:
        return jsonify({'errors': ["No usable transactions left after cleaning (rows need a Customer ID, "
                                   "a positive price and a real product)."]}), 400

    # "Today" for Recency: the day after the last invoice in the file, unless the user picked a date
    last_invoice = clean['InvoiceDate'].max()
    snapshot_text = request.form.get('snapshot_date', '').strip()
    if snapshot_text:
        snapshot = pd.to_datetime(snapshot_text, errors='coerce')
        if pd.isna(snapshot):
            return jsonify({'errors': [f"'{snapshot_text}' is not a date (use YYYY-MM-DD)."]}), 400
        if snapshot < last_invoice.normalize():
            return jsonify({'errors': [f"The analysis date can't be before the last invoice ({last_invoice:%Y-%m-%d})."]}), 400
    else:
        snapshot = last_invoice.normalize() + pd.Timedelta(days=1)

    customers = build_customer_features(clean, snapshot_date=snapshot)
    if customers.empty:
        return jsonify({'errors': ["No customers with at least GBP 1 of net spend in this file."]}), 400

    predictions = predict_customers(bundle, customers)
    rows = []
    for row, (customer_id, features) in zip(describe(predictions), customers.iterrows()):
        row['customer'] = int(customer_id)
        row['features'] = {f: round(float(features[f]), 3) for f in FEATURES}
        row['avg_order_value'] = round(float(features['AvgOrderValue']), 2)
        row['country'] = features['Country']
        rows.append(row)

    return jsonify({
        'customers': rows,
        'summary': predictions['Segment'].value_counts().to_dict(),
        'outliers': int(predictions['IsOutlier'].sum()),
        'rows_read': rows_read,
        'rows_unreadable': unreadable,
        'rows_used': len(clean),
        'snapshot_date': snapshot.strftime('%Y-%m-%d'),
        'last_invoice': last_invoice.strftime('%Y-%m-%d %H:%M'),
        'colors': COLORS
    })


@app.get("/api/sample")
def sample():
    """The sample transactions file (made by make_sample.py), to try the upload."""
    if not os.path.exists(SAMPLE_FILE):
        return jsonify({'errors': ["No sample file. Run 'python make_sample.py' to create it."]}), 404
    return send_file(SAMPLE_FILE, mimetype='text/csv', as_attachment=True, download_name='sample_transactions.csv')


@app.errorhandler(413)
def too_large(error):
    return jsonify({'errors': ["The file is larger than 50 MB."]}), 413


if __name__ == "__main__":
    app.run(debug=True)
