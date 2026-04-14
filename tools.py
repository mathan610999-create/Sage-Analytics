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
    Maps column names to standard names.
    First tries rule-based matching, then AI for anything remaining.
    """
    mapping = {}

    # Rule-based mapping first — no API needed
    RULES = {
        "revenue":      ["total sales", "totalsales", "gross sales", "gross_sales",
                         "net sales", "net_sales", "revenue", "sales", "amount",
                         "total revenue", "total_revenue", "turnover"],
        "profit":       ["operating profit", "operating_profit", "net profit",
                         "net_profit", "gross profit", "gross_profit", "profit",
                         "earnings", "operating income", "ebit"],
        "units_sold":   ["units sold", "units_sold", "quantity", "qty", "volume",
                         "units", "items sold", "items_sold", "pieces"],
        "margin_pct":   ["operating margin", "operating_margin", "margin",
                         "profit margin", "profit_margin", "margin %",
                         "gross margin", "net margin", "margin_pct"],
        "region":       ["region", "area", "territory", "zone", "market",
                         "geography", "geographic region"],
        "category":     ["category", "product category", "product_category",
                         "type", "segment", "division", "department"],
        "product":      ["product", "product name", "product_name", "item",
                         "sku", "description", "goods"],
        "channel":      ["channel", "sales method", "sales_method",
                         "sales channel", "method", "medium", "retailer type"],
        "date":         ["date", "invoice date", "invoice_date", "order date",
                         "order_date", "transaction date", "sale date", "sale_date"],
        "retailer":     ["retailer", "retailer name", "store", "vendor",
                         "seller", "merchant", "account"],
        "price":        ["price per unit", "price_per_unit", "unit price",
                         "unit_price", "price", "selling price", "rate"],
        "state":        ["state", "province", "state_province"],
        "city":         ["city", "town"],
        "discount_pct": ["discount", "discount %", "discount_pct",
                         "markdown", "promo"],
    }

    used_standards = set()
    col_lower = {col.lower().strip(): col for col in df.columns}

    for standard, aliases in RULES.items():
        if standard in used_standards:
            continue
        for alias in aliases:
            if alias in col_lower and standard not in used_standards:
                original = col_lower[alias]
                if original not in mapping:
                    mapping[original] = standard
                    used_standards.add(standard)
                    break

    # AI detection for any columns not yet mapped
    unmapped_cols = [c for c in df.columns if c not in mapping]
    if unmapped_cols and len(used_standards) < len(RULES):
        # Try to get API key from multiple sources
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("ANTHROPIC_API_KEY")
            except:
                pass

        if api_key:
            sample_data = {}
            for col in unmapped_cols[:10]:
                sample_data[col] = df[col].dropna().head(3).tolist()

            remaining_standards = {k: v for k, v in STANDARD_COLUMNS.items()
                                   if k not in used_standards}
            if remaining_standards and sample_data:
                standard_desc = "\n".join([f"- {k}: {v}"
                                          for k, v in remaining_standards.items()])
                prompt = f"""Map these columns to standard names if they match:
Columns: {json.dumps(sample_data, indent=2, default=str)}
Standards: {standard_desc}
Return ONLY JSON like: {{"Original Col": "standard_name"}}"""

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
                            "max_tokens": 300,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=10
                    )
                    if response.status_code == 200:
                        text = response.json()["content"][0]["text"].strip()
                        text = text.replace("```json", "").replace("```", "").strip()
                        ai_map = json.loads(text)
                        for k, v in ai_map.items():
                            if k in df.columns and v in STANDARD_COLUMNS and k not in mapping:
                                mapping[k] = v
                except:
                    pass

    return mapping






# ─────────────────────────────────────────────
# DATA CLEANER
# ─────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame, col_mapping: dict) -> tuple:
    """
    Cleans the dataframe using detected column mapping.
    Returns (cleaned_df, list_of_changes)
    """
    changes = []
    df = df.copy()

    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Apply column mapping
    if col_mapping:
        df = df.rename(columns=col_mapping)
        for orig, std in col_mapping.items():
            changes.append(f"Renamed '{orig}' → '{std}'")

    # 3. Remove header rows mixed into data
    # (rows where text columns contain column names like "Region", "Product")
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    col_names_lower = set(df.columns.str.lower().tolist())
    mask = pd.Series([True] * len(df), index=df.index)
    for col in text_cols:
        bad = df[col].astype(str).str.strip().str.lower().isin(col_names_lower)
        mask = mask & ~bad
    removed_headers = (~mask).sum()
    if removed_headers > 0:
        df = df[mask].reset_index(drop=True)
        changes.append(f"Removed {removed_headers} header rows mixed into data")

    # 4. Clean numeric columns — remove $, commas, % signs
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

    # 5. Fix margin_pct — convert decimal to percentage
    if "margin_pct" in df.columns:
        try:
            numeric_margin = pd.to_numeric(df["margin_pct"], errors="coerce")
            if numeric_margin.dropna().mean() < 1:
                df["margin_pct"] = numeric_margin * 100
                changes.append("Converted margin from decimal to percentage")
        except:
            pass

    # 6. Parse date column — try 'date' first, then any date-like column
    date_parsed = False
    date_candidates = ["date"] + [c for c in df.columns if c != "date" and
                                   any(x in c.lower() for x in ["date", "time", "invoice", "order"])]
    for date_col in date_candidates:
        if date_col in df.columns and not date_parsed:
            try:
                parsed = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:
                    df["date"] = parsed
                    df["month"] = parsed.dt.strftime("%B")
                    df["quarter"] = "Q" + parsed.dt.quarter.astype(str)
                    df["year"] = parsed.dt.year
                    date_parsed = True
                    changes.append(f"Parsed '{date_col}' → extracted month, quarter, year")
            except:
                pass

    # 7. Derive revenue if missing
    if "revenue" not in df.columns and "price" in df.columns and "units_sold" in df.columns:
        df["revenue"] = (pd.to_numeric(df["price"], errors="coerce") *
                         pd.to_numeric(df["units_sold"], errors="coerce")).round(2)
        changes.append("Derived 'revenue' from price × units_sold")

    # 8. Derive profit if missing
    if "profit" not in df.columns and "revenue" in df.columns and "margin_pct" in df.columns:
        df["profit"] = (pd.to_numeric(df["revenue"], errors="coerce") *
                        pd.to_numeric(df["margin_pct"], errors="coerce") / 100).round(2)
        changes.append("Derived 'profit' from revenue × margin_pct")

    # 9. Fill missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    null_counts = df[numeric_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0)
        changes.append(f"Filled missing values in {len(cols_with_nulls)} column(s)")

    # 10. Strip whitespace from text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # 11. Remove fully empty rows
    before = len(df)
    df = df.dropna(how="all")
    removed = before - len(df)
    if removed > 0:
        changes.append(f"Removed {removed} empty rows")

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
