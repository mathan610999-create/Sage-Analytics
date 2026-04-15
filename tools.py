"""
tools.py - All agent tools for Sage
AI-powered column detection + smart data cleaning for any uploaded file
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
import requests
from langchain_core.tools import tool

_df: pd.DataFrame = None
_db_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sage_data.db")
_cleaning_report: list = []


# ─────────────────────────────────────────────
# AI COLUMN DETECTOR
# Uses Claude to intelligently map any column
# names to standard names — works on ANY dataset
# ─────────────────────────────────────────────
STANDARD_COLUMNS = {
    "revenue":      "Total monetary value of sales (e.g. Total Sales, Gross Revenue, Amount)",
    "profit":       "Net earnings after costs (e.g. Operating Profit, Net Income, Earnings)",
    "units_sold":   "Number of items sold (e.g. Quantity, Units, Volume, Pieces)",
    "margin_pct":   "Profit as percentage of revenue (e.g. Operating Margin, Gross Margin %)",
    "region":       "Geographic area (e.g. Territory, Zone, Area, Market)",
    "category":     "Product group (e.g. Product Category, Type, Segment, Department)",
    "product":      "Product name or description (e.g. Item, SKU, Product Name)",
    "channel":      "Sales method or distribution (e.g. Sales Method, Retailer Type, Medium)",
    "date":         "Transaction date (e.g. Invoice Date, Order Date, Sale Date)",
    "retailer":     "Store or seller name (e.g. Retailer, Store, Vendor, Account)",
    "price":        "Unit selling price (e.g. Price per Unit, Unit Price, Rate)",
    "discount_pct": "Discount percentage applied (e.g. Discount %, Markdown, Promotion)",
    "state":        "State or province location",
    "city":         "City location",
    "customer":     "Customer name or ID",
}


def ai_detect_columns(df: pd.DataFrame) -> dict:
    """
    Uses Claude API to intelligently map any column names
    to standard names. Works on ANY dataset regardless of
    how columns are named.
    Returns: {original_col: standard_col}
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {}

    # Build a sample of the data to send to Claude
    sample_data = {}
    for col in df.columns:
        sample_vals = df[col].dropna().head(3).tolist()
        sample_data[col] = sample_vals

    standard_desc = "\n".join([f"- {k}: {v}" for k, v in STANDARD_COLUMNS.items()])

    prompt = f"""You are a data analyst. I have a dataset with these columns and sample values:

{json.dumps(sample_data, indent=2, default=str)}

Map each column to ONE of these standard names if it matches:
{standard_desc}

Rules:
- Only map if you are confident the column represents that concept
- Do not map columns that don't match any standard name
- Each standard name can only be used ONCE (pick the best match)
- Return ONLY a JSON object like: {{"original_col": "standard_col", ...}}
- Do not include any explanation, only the JSON

Example output:
{{"Total Sales": "revenue", "Operating Profit": "profit", "Units Sold": "units_sold"}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15
        )

        if response.status_code == 200:
            text = response.json()["content"][0]["text"].strip()
            # Clean up response — remove markdown if present
            text = text.replace("```json", "").replace("```", "").strip()
            mapping = json.loads(text)
            # Validate — only keep mappings where original col exists
            valid = {k: v for k, v in mapping.items()
                     if k in df.columns and v in STANDARD_COLUMNS}
            return valid
    except Exception as e:
        print(f"AI column detection failed: {e}")

    return {}


# ─────────────────────────────────────────────
# DATA CLEANER
# ─────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame, col_mapping: dict) -> tuple:
    """
    Cleans the dataframe using AI-detected column mapping.
    Returns (cleaned_df, list_of_changes)
    """
    changes = []
    df = df.copy()

    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Apply AI column mapping
    if col_mapping:
        df = df.rename(columns=col_mapping)
        for orig, std in col_mapping.items():
            changes.append(f"Renamed '{orig}' → '{std}'")

    # 3. Clean numeric columns — remove $, commas, % signs
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(10).astype(str)
            cleaned = sample.str.replace(r'[\$,%\s]', '', regex=True)
            try:
                pd.to_numeric(cleaned)
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[\$,%\s]', '', regex=True),
                    errors='coerce'
                )
                changes.append(f"Converted '{col}' to numeric")
            except:
                pass

    # 4. Fix margin_pct — convert decimal to percentage
    if "margin_pct" in df.columns:
        try:
            numeric_margin = pd.to_numeric(df["margin_pct"], errors="coerce")
            if numeric_margin.dropna().mean() < 1:
                df["margin_pct"] = numeric_margin * 100
                changes.append("Converted margin from decimal to percentage")
        except:
            pass

    # 5. Parse date column
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            valid_dates = df["date"].notna().sum()
            if valid_dates > 0:
                df["month"] = df["date"].dt.strftime("%B")
                df["quarter"] = "Q" + df["date"].dt.quarter.astype(str)
                df["year"] = df["date"].dt.year
                changes.append("Parsed date → extracted month, quarter, year")
        except:
            pass

    # 6. Fill missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    null_counts = df[numeric_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0)
        changes.append(f"Filled missing values in {len(cols_with_nulls)} column(s)")

    # 7. Strip whitespace from text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # 8. Remove fully empty rows
    before = len(df)
    df = df.dropna(how="all")
    removed = before - len(df)
    if removed > 0:
        changes.append(f"Removed {removed} empty rows")

    # 9. Derive profit if missing
    if "profit" not in df.columns and "revenue" in df.columns and "margin_pct" in df.columns:
        df["profit"] = (df["revenue"] * df["margin_pct"] / 100).round(2)
        changes.append("Derived 'profit' from revenue × margin_pct")

    # 10. Derive revenue if missing
    if "revenue" not in df.columns and "price" in df.columns and "units_sold" in df.columns:
        df["revenue"] = (df["price"] * df["units_sold"]).round(2)
        changes.append("Derived 'revenue' from price × units_sold")

    return df, changes


# ─────────────────────────────────────────────
# SMART EXCEL READER
# Detects and skips title rows automatically
# ─────────────────────────────────────────────
def smart_read_excel(file_buffer) -> pd.DataFrame:
    """Reads Excel file — auto-detects correct header row"""
    raw = pd.read_excel(file_buffer, header=None)

    header_row = 0
    for i in range(min(10, len(raw))):
        row = raw.iloc[i].astype(str).str.strip()
        non_null = (row.str.lower() != 'nan').sum()
        if non_null >= max(3, len(raw.columns) * 0.5):
            header_row = i
            break

    file_buffer.seek(0)
    df = pd.read_excel(file_buffer, header=header_row)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    return df


# ─────────────────────────────────────────────
# MAIN LOAD FUNCTION
# Called when user uploads any file
# ─────────────────────────────────────────────
def load_dataframe(df: pd.DataFrame, db_path: str = None):
    """
    1. Drops empty rows/cols
    2. AI detects column mapping
    3. Cleans and standardizes
    4. Loads to SQLite
    Returns list of changes made
    """
    global _df, _db_path, _cleaning_report

    # Always use absolute path so agent.py and tools.py share the same DB
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sage_data.db")

    # Drop fully empty rows and columns
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # AI-powered column detection
    col_mapping = ai_detect_columns(df)

    # Clean with detected mapping
    df_clean, changes = clean_dataframe(df, col_mapping)

    _df = df_clean
    _db_path = db_path
    _cleaning_report = changes

    # Save to SQLite
    conn = sqlite3.connect(db_path)
    df_clean.to_sql("sales", conn, if_exists="replace", index=False)
    conn.close()

    return changes


def get_df():
    return _df


def get_cleaning_report():
    return _cleaning_report


# ─────────────────────────────────────────────
# LANGCHAIN TOOLS
# ─────────────────────────────────────────────
@tool
def profile_data(input: str = "") -> str:
    """
    Profiles the uploaded dataset. Returns shape, columns, types,
    missing values, and statistics. Always call this first.
    """
    df = get_df()
    if df is None:
        return "No data loaded yet."

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": {},
        "categorical_summary": {},
        "cleaning_applied": get_cleaning_report(),
    }

    for col in df.select_dtypes(include=[np.number]).columns:
        profile["numeric_summary"][col] = {
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
        }

    for col in df.select_dtypes(include=["object"]).columns:
        profile["categorical_summary"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_5": df[col].value_counts().head(5).to_dict(),
        }

    return json.dumps(profile, indent=2)


@tool
def run_eda(focus_column: str = "") -> str:
    """
    Runs exploratory data analysis — correlations, outliers,
    top/bottom performers. Pass column name to focus on it.
    """
    df = get_df()
    if df is None:
        return "No data loaded yet."

    results = {}
    numeric_df = df.select_dtypes(include=[np.number])

    # Strong correlations
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr().round(2)
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.5:
                    strong.append({
                        "col1": corr.columns[i],
                        "col2": corr.columns[j],
                        "correlation": float(val)
                    })
        results["strong_correlations"] = strong

    # Outliers
    outliers = {}
    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        count = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
        if count > 0:
            outliers[col] = count
    results["outliers"] = outliers

    # Category analysis
    rev_col = next((c for c in ["revenue", "profit", "units_sold"]
                    if c in df.columns), None)
    if rev_col:
        for cat_col in df.select_dtypes(include=["object"]).columns[:3]:
            grouped = df.groupby(cat_col)[rev_col].sum().sort_values(ascending=False)
            results[f"{cat_col}_by_{rev_col}"] = grouped.round(2).to_dict()

    return json.dumps(results, indent=2)


@tool
def run_sql(query: str) -> str:
    """
    Executes a SQL SELECT query against the uploaded dataset.
    Table is always called 'sales'. Only SELECT is allowed.
    """
    if _db_path is None:
        return "No data loaded yet."

    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return "Only SELECT queries are allowed."

    try:
        conn = sqlite3.connect(_db_path)
        result = pd.read_sql_query(query, conn)
        conn.close()
        if result.empty:
            return "Query returned 0 rows."
        return result.head(20).to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"


@tool
def calculate_kpis(input: str = "") -> str:
    """
    Calculates key business KPIs. Call for business overviews.
    """
    df = get_df()
    if df is None:
        return "No data loaded yet."

    kpis = {}

    if "revenue" in df.columns:
        kpis["total_revenue"] = round(float(df["revenue"].sum()), 2)
        kpis["avg_order_revenue"] = round(float(df["revenue"].mean()), 2)

    if "profit" in df.columns:
        kpis["total_profit"] = round(float(df["profit"].sum()), 2)
        if "revenue" in df.columns and df["revenue"].sum() > 0:
            kpis["avg_margin_pct"] = round(
                float(df["profit"].sum() / df["revenue"].sum() * 100), 1)

    if "units_sold" in df.columns:
        kpis["total_units_sold"] = int(df["units_sold"].sum())

    for group_col in ["region", "category", "channel", "product", "retailer"]:
        if group_col in df.columns and "revenue" in df.columns:
            grouped = df.groupby(group_col)["revenue"].sum().sort_values(ascending=False)
            kpis[f"revenue_by_{group_col}"] = grouped.round(2).to_dict()
            if group_col in ["region", "category"]:
                kpis[f"top_{group_col}"] = grouped.index[0]

    if "month" in df.columns and "revenue" in df.columns:
        monthly = df.groupby("month")["revenue"].sum()
        kpis["monthly_revenue"] = monthly.round(2).to_dict()

    kpis["available_columns"] = df.columns.tolist()
    kpis["cleaning_applied"] = get_cleaning_report()

    return json.dumps(kpis, indent=2)


@tool
def get_schema(input: str = "") -> str:
    """
    Returns the schema of the uploaded dataset.
    Call before writing SQL to know exact column names.
    """
    df = get_df()
    if df is None:
        return "No data loaded yet."

    schema = "Table: sales\nColumns:\n"
    for col, dtype in df.dtypes.items():
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
        schema += f"  - {col} ({dtype}) — example: {sample}\n"
    return schema
