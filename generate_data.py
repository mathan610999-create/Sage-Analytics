"""
generate_data.py - Creates a realistic sample sales dataset for Sage
Run once to create sample_sales_data.csv
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

PRODUCTS = [
    ("Running Shoes", "Footwear", 89.99),
    ("Casual Sneakers", "Footwear", 64.99),
    ("Sports T-Shirt", "Apparel", 29.99),
    ("Track Pants", "Apparel", 44.99),
    ("Water Bottle", "Accessories", 19.99),
    ("Gym Bag", "Accessories", 54.99),
    ("Yoga Mat", "Equipment", 34.99),
    ("Resistance Bands", "Equipment", 24.99),
    ("Winter Jacket", "Apparel", 119.99),
    ("Sports Cap", "Accessories", 14.99),
]

REGIONS = ["North", "South", "East", "West", "Central"]

RETAILERS = [
    ("SportZone", "In-Store"),
    ("ActiveGear Online", "Online"),
    ("FitLife Stores", "In-Store"),
    ("QuickShip Sports", "Online"),
    ("MegaSport", "In-Store"),
]

# Seasonal multipliers per month
SEASONALITY = {
    1: 0.7, 2: 0.75, 3: 0.9, 4: 1.0,
    5: 1.1, 6: 1.2, 7: 1.15, 8: 1.1,
    9: 1.0, 10: 1.05, 11: 1.3, 12: 1.4
}

# Regional performance multipliers
REGION_MULT = {
    "North": 1.1, "South": 0.78,
    "East": 1.05, "West": 1.15, "Central": 0.95
}

rows = []
start_date = datetime(2023, 1, 1)

for _ in range(2000):
    date = start_date + timedelta(days=random.randint(0, 364))
    product, category, base_price = random.choice(PRODUCTS)
    region = random.choice(REGIONS)
    retailer, channel = random.choice(RETAILERS)

    seasonal = SEASONALITY[date.month]
    regional = REGION_MULT[region]

    units = max(1, int(np.random.normal(15, 6) * seasonal * regional))
    discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)
    unit_price = round(base_price * (1 - discount), 2)
    revenue = round(units * unit_price, 2)
    cost = round(units * base_price * 0.55, 2)
    profit = round(revenue - cost, 2)
    margin = round((profit / revenue) * 100, 1) if revenue > 0 else 0

    rows.append({
        "order_id": f"ORD{10000 + len(rows)}",
        "date": date.strftime("%Y-%m-%d"),
        "month": date.strftime("%B"),
        "quarter": f"Q{(date.month - 1) // 3 + 1}",
        "product": product,
        "category": category,
        "region": region,
        "retailer": retailer,
        "channel": channel,
        "units_sold": units,
        "unit_price": unit_price,
        "discount_pct": discount * 100,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "margin_pct": margin,
    })

df = pd.DataFrame(rows)
df = df.sort_values("date").reset_index(drop=True)
df.to_csv("sample_sales_data.csv", index=False)

print(f"✅ Created sample_sales_data.csv")
print(f"   Rows: {len(df)}")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
print(f"   Total revenue: ${df['revenue'].sum():,.2f}")
print(f"   Regions: {sorted(df['region'].unique().tolist())}")
print(f"   Products: {len(df['product'].unique())}")
