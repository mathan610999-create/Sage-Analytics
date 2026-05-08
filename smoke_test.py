"""Smoke test — verify Sage works on three different dataset shapes."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tools as _tools
_tools._db_path = os.path.join(tempfile.gettempdir(), "sage_smoke.db")

from tools import (
    correlations,
    get_profile,
    get_schema,
    load_dataframe,
    quick_prompts_for_dataset,
    run_sql,
    time_series,
    top_n,
    value_counts,
)


def banner(s):
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


# Dataset 1 — Sales / retail
banner("DATASET 1 - Sales / retail")
sales = pd.DataFrame({
    "Region": np.random.choice(["North", "South", "East", "West"], 200),
    "Product Category": np.random.choice(["Footwear", "Apparel", "Accessories"], 200),
    "Total Sales": np.random.randint(100, 5000, 200),
    "Units Sold": np.random.randint(1, 50, 200),
    "Order Date": pd.date_range("2024-01-01", periods=200, freq="D"),
})
print("Cleaning:", load_dataframe(sales, "sales.csv"))
prof = get_profile()
print("Classes:", {k: v for k, v in prof["classes"].items() if v})
print("Quick prompts:", quick_prompts_for_dataset())
cat = prof["classes"]["categorical"][0]
num = prof["classes"]["numeric"][0]
print("\nvalue_counts:")
print(value_counts.invoke({"column": cat, "top_n": 4}))
print("\ntop_n:")
print(top_n.invoke({"group_by": cat, "metric": num, "n": 3}))
print("\nrun_sql:")
print(run_sql.invoke({"query": f"SELECT {cat}, SUM({num}) AS s FROM data GROUP BY {cat} ORDER BY s DESC LIMIT 3"}))


# Dataset 2 — HR
banner("DATASET 2 - HR / employees")
hr = pd.DataFrame({
    "Employee ID": range(1, 151),
    "Department": np.random.choice(["Engineering", "Sales", "HR", "Marketing"], 150),
    "Salary": np.random.randint(40000, 150000, 150),
    "Years at Company": np.random.randint(0, 25, 150),
    "Performance Score": np.round(np.random.uniform(2.5, 5.0, 150), 1),
    "Hire Date": pd.date_range("2010-01-01", periods=150, freq="20D"),
})
print("Cleaning:", load_dataframe(hr, "hr.csv"))
prof = get_profile()
print("Classes:", {k: v for k, v in prof["classes"].items() if v})
print("Quick prompts:", quick_prompts_for_dataset())
print("\ntop_n Department by Salary:")
print(top_n.invoke({"group_by": "department", "metric": "salary", "n": 4}))
print("\ncorrelations:")
print(correlations.invoke({"threshold": 0.0}))


# Dataset 3 — Sensor
banner("DATASET 3 - Sensor readings")
sensor = pd.DataFrame({
    "Timestamp": pd.date_range("2026-01-01", periods=400, freq="h"),
    "Temperature C": 18 + np.random.randn(400) * 3,
    "Humidity %": 50 + np.random.randn(400) * 12,
    "Pressure hPa": 1013 + np.random.randn(400) * 4,
    "Sensor": np.random.choice(["A", "B", "C"], 400),
})
print("Cleaning:", load_dataframe(sensor, "sensor.csv"))
prof = get_profile()
print("Classes:", {k: v for k, v in prof["classes"].items() if v})
print("Quick prompts:", quick_prompts_for_dataset())
print("\ntime_series of temperature_c (first lines):")
ts = time_series.invoke({"date_column": "timestamp", "metric": "temperature_c", "freq": "D"})
print(ts.split(os.linesep if False else "\n")[:3])
print("\nschema:")
print(get_schema.invoke({"input": ""}))

banner("ALL THREE DATASETS PASSED")
