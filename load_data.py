# Import necessary libraries
import os
import zipfile
import urllib.request
import pandas as pd

# UCI Online Retail II: every transaction of a UK online gift shop, Dec 2009 - Dec 2011
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# Reading the Excel file takes about a minute, so we convert it once to Parquet
# (a fast, compressed column format) and reuse that copy afterwards
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EXCEL_FILE = os.path.join(DATA_DIR, "online_retail_II.xlsx")
DATA_FILE = os.path.join(DATA_DIR, "online_retail.parquet")
DOWNLOAD_ATTEMPTS = 3

# Column meanings (names as they appear in the file)
COLUMNS = {
    'Invoice': 'Invoice number. Starts with "C" when the invoice is a cancellation',
    'StockCode': 'Product code (a few codes are not products, e.g. POST = postage)',
    'Description': 'Product name',
    'Quantity': 'Number of items on this invoice line (negative for cancellations)',
    'InvoiceDate': 'Date and time of the invoice',
    'Price': 'Price per item in pounds sterling',
    'Customer ID': 'Customer number (missing for some rows)',
    'Country': 'Country the customer lives in'
}


def download_data():
    """Download the zip from UCI and extract the Excel file into data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "online_retail_II.zip")

    # The UCI server can be slow (sometimes ~100 KB/s) and sometimes drops the connection
    # part-way, so save to disk in chunks, show progress, and retry a few times
    print("Downloading Online Retail II from UCI (about 45 MB, can take several minutes)...")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(DATA_URL, timeout=60) as response, open(zip_path + ".part", 'wb') as target:
                done = 0
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    done += len(chunk)
                    print(f"\r  {done / 1e6:.1f} MB", end="", flush=True)
            print()

            # A cut-off download still produces a file, so check the zip is complete before using it
            if not zipfile.is_zipfile(zip_path + ".part"):
                raise IOError("the downloaded file is not a complete zip")
            os.replace(zip_path + ".part", zip_path)
            break
        except Exception as error:
            print(f"\n  attempt {attempt} of {DOWNLOAD_ATTEMPTS} failed: {error}")
            if attempt == DOWNLOAD_ATTEMPTS:
                raise SystemExit("Download failed. Try again later, or download the zip yourself from "
                                 f"{DATA_URL} and put online_retail_II.xlsx in {DATA_DIR}")

    # Extract only the Excel file, by name, into data/
    with zipfile.ZipFile(zip_path) as archive:
        excel_name = next(name for name in archive.namelist() if name.endswith('.xlsx'))
        with archive.open(excel_name) as source, open(EXCEL_FILE, 'wb') as target:
            target.write(source.read())
    os.remove(zip_path)


def load_data():
    """Load all transactions (both years) as one DataFrame."""

    # Build the Parquet copy the first time, then reuse it
    if not os.path.exists(DATA_FILE):
        if not os.path.exists(EXCEL_FILE):
            download_data()

        # The Excel file has one sheet per year; sheet_name=None reads them all.
        # Text columns are read as text explicitly: a few product codes and names are stored
        # as numbers in Excel, and a column mixing text and numbers can't be saved to Parquet.
        print("Reading the Excel file (this takes about a minute, only the first time)...")
        text_columns = {'Invoice': str, 'StockCode': str, 'Description': str, 'Country': str}
        sheets = pd.read_excel(EXCEL_FILE, sheet_name=None, dtype=text_columns)
        raw = pd.concat(sheets.values(), ignore_index=True)
        raw.to_parquet(DATA_FILE, index=False)
        print(f"Saved a fast copy to {DATA_FILE}")

    return pd.read_parquet(DATA_FILE)


# Stock codes that are not products: postage, fees, manual adjustments, discounts, samples,
# gift vouchers and test items. (Codes starting with DCGS are real web-shop products, so they stay.)
NON_PRODUCT_CODES = ['POST', 'DOT', 'M', 'C2', 'D', 'S', 'B', 'BANK CHARGES', 'AMAZONFEE', 'CRUK']
NON_PRODUCT_PREFIXES = ('ADJUST', 'GIFT_', 'TEST')


def clean_data(df, verbose=False):
    """Clean the transactions for customer segmentation.

    Cancellations are KEPT (flagged with IsCancellation) instead of dropped: some huge orders
    were cancelled minutes later, so spending has to be computed as purchases minus cancellations.
    Adds a Revenue column (negative for cancellations).
    """
    steps = [('Raw data', len(df))]

    def log(name, data):
        steps.append((name, len(data)))
        return data

    # 1. Rows without a Customer ID can't be assigned to anyone. This also removes all the
    #    stock adjustments ("damages", "check", "?"), which never have a customer.
    df = log('Has a Customer ID', df.dropna(subset=['Customer ID']))

    # 2. Exact duplicate rows are almost certainly the same line scanned twice
    df = log('Duplicates removed', df.drop_duplicates())

    # 3. Keep real products only
    codes = df['StockCode'].str.upper()
    is_product = ~codes.isin(NON_PRODUCT_CODES) & ~codes.str.startswith(NON_PRODUCT_PREFIXES)
    df = log('Products only', df[is_product])

    # 4. Prices must be positive (a few free lines have price 0)
    df = log('Price > 0', df[df['Price'] > 0])

    # 5. Purchases have positive quantities and cancellations negative ones; drop anything else
    is_cancellation = df['Invoice'].str.startswith('C')
    consistent = (is_cancellation & (df['Quantity'] < 0)) | (~is_cancellation & (df['Quantity'] > 0))
    df = log('Consistent quantities', df[consistent]).copy()

    df['Customer ID'] = df['Customer ID'].astype(int)
    df['IsCancellation'] = df['Invoice'].str.startswith('C')
    df['Revenue'] = df['Quantity'] * df['Price']

    if verbose:
        print("Cleaning steps (rows left after each):")
        for name, rows in steps:
            print(f"  {name:22s} {rows:>10,}")

    df.attrs['cleaning_steps'] = steps
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()

    # Display the first few rows of the data
    print(df.head())

    # Shape of the dataset
    print(f"\nDataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Data types of each column
    print("\nColumn data types:")
    print(df.dtypes)

    # Date range covered by the data
    print(f"\nFirst invoice: {df['InvoiceDate'].min()}")
    print(f"Last invoice:  {df['InvoiceDate'].max()}")

    # Missing values per column
    missing = df.isnull().sum()
    print("\nMissing values:")
    print(pd.DataFrame({'missing': missing, 'percent': (missing / len(df) * 100).round(1)})[missing > 0])

    # How many customers, invoices, products and countries
    print(f"\nCustomers: {df['Customer ID'].nunique():,}")
    print(f"Invoices:  {df['Invoice'].nunique():,}")
    print(f"Products:  {df['StockCode'].nunique():,}")
    print(f"Countries: {df['Country'].nunique()}")

    # Things that will need cleaning in step 2
    cancellations = df['Invoice'].str.startswith('C')
    print("\nRows that will need cleaning:")
    print(f"  Cancellation lines (invoice starts with 'C'): {cancellations.sum():,}")
    print(f"  Lines with quantity <= 0:                     {(df['Quantity'] <= 0).sum():,}")
    print(f"  Lines with price <= 0:                        {(df['Price'] <= 0).sum():,}")
    print(f"  Exact duplicate rows:                         {df.duplicated().sum():,}")

    # Top countries by number of lines
    print("\nTop 5 countries by number of lines:")
    print((df['Country'].value_counts(normalize=True).head(5) * 100).round(1).astype(str) + '%')
