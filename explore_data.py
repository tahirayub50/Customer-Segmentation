# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data, clean_data

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load and clean the data
raw = load_data()
df = clean_data(raw, verbose=True)

purchases = df[~df['IsCancellation']]
cancellations = df[df['IsCancellation']]
print(f"\nCustomers: {df['Customer ID'].nunique():,}")
print(f"Purchase lines: {len(purchases):,}, cancellation lines: {len(cancellations):,}")
print(f"Net revenue: GBP {df['Revenue'].sum():,.0f} "
      f"(purchases GBP {purchases['Revenue'].sum():,.0f}, cancellations GBP {cancellations['Revenue'].sum():,.0f})")


# 1. How much data each cleaning step removed
steps = pd.DataFrame(df.attrs['cleaning_steps'], columns=['step', 'rows'])
plt.figure(figsize=(9, 4.5))
sns.barplot(data=steps, x='rows', y='step', color='steelblue')
for i, rows in enumerate(steps['rows']):
    plt.text(rows, i, f"  {rows:,}", va='center')
plt.xlim(0, steps['rows'].max() * 1.2)
plt.xlabel('Rows left')
plt.ylabel('')
plt.title('Rows Left After Each Cleaning Step')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '1_cleaning_steps.png'))


# 2. Sales over time: monthly net revenue and active customers
monthly = df.set_index('InvoiceDate').resample('MS').agg({'Revenue': 'sum', 'Customer ID': 'nunique'})
# The data ends on 9 Dec 2011, so the last month is incomplete
monthly = monthly.iloc[:-1]
print("\nBusiest months by net revenue:")
print(monthly['Revenue'].nlargest(3).map('GBP {:,.0f}'.format))

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.bar(monthly.index, monthly['Revenue'] / 1000, width=20, color='steelblue', label='Net revenue')
ax1.set_ylabel('Net revenue (£ thousands)')
ax2 = ax1.twinx()
ax2.plot(monthly.index, monthly['Customer ID'], color='darkorange', marker='o', label='Active customers')
ax2.set_ylabel('Active customers')
fig.legend(loc='upper left', bbox_to_anchor=(0.08, 0.9))
plt.title('Monthly Net Revenue and Active Customers (Dec 2009 - Nov 2011)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '2_monthly_sales.png'))


# 3. Where customers live
countries = df.groupby('Customer ID')['Country'].agg(lambda c: c.mode()[0]).value_counts()
print(f"\nCountries: {len(countries)}. Customers in the UK: {countries['United Kingdom'] / countries.sum():.1%}")
print(countries.head(8))

plt.figure(figsize=(8, 4.5))
top = countries.head(10)
sns.barplot(x=top.values, y=top.index, color='steelblue')
plt.xscale('log')
plt.xlabel('Customers (log scale)')
plt.ylabel('')
plt.title('Top 10 Countries by Number of Customers')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '3_countries.png'))


# 4. A first look at customers: how skewed are orders and spending?
# (Proper customer features come in step 3; this is just to see the shape of the data.)
per_customer = df.groupby('Customer ID').agg(
    orders=('Invoice', lambda inv: inv[~inv.str.startswith('C')].nunique()),
    net_spend=('Revenue', 'sum')
)
print("\nOrders and net spend per customer:")
print(per_customer.describe(percentiles=[0.25, 0.5, 0.75, 0.99]).round(1))

# Pareto: what share of revenue comes from the top 20% of customers?
spend = per_customer['net_spend'].clip(lower=0).sort_values(ascending=False)
top_20_share = spend.head(int(len(spend) * 0.2)).sum() / spend.sum()
print(f"\nTop 20% of customers bring in {top_20_share:.0%} of net revenue")
print(f"Customers with net spend <= 0 (cancelled everything): {(per_customer['net_spend'] <= 0).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, label in [(axes[0], 'orders', 'Orders per customer'), (axes[1], 'net_spend', 'Net spend per customer (£)')]:
    # Log scale needs positive values; >= 1 also skips tiny float leftovers like 1e-13 after a full refund
    values = per_customer[col][per_customer[col] >= 1]
    ax.hist(values, bins=np.logspace(np.log10(values.min()), np.log10(values.max()), 40), color='steelblue')
    ax.set_xscale('log')
    ax.set_xlabel(label + ' (log scale)')
    ax.set_ylabel('Customers')
    ax.axvline(values.median(), color='darkorange', linestyle='--', label=f'Median: {values.median():,.0f}')
    ax.legend()
plt.suptitle('Customers Are Very Unequal: Most Buy Little, a Few Buy a Lot')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '4_customer_distributions.png'))


# 5. Cancellations: how common are they?
cancel_share = (per_customer.index.isin(cancellations['Customer ID'])).mean()
print(f"\nCustomers who cancelled at least once: {cancel_share:.0%}")
print("Biggest cancellations:")
print(cancellations.nsmallest(3, 'Revenue')[['Invoice', 'Customer ID', 'Description', 'Quantity', 'Revenue', 'InvoiceDate']])


# 6. Best-selling products by net revenue
print("\nTop 10 products by net revenue:")
print(df.groupby('Description')['Revenue'].sum().nlargest(10).map('GBP {:,.0f}'.format))

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
