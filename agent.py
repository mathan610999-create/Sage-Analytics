"""
agent.py - Sage AI Analyst Agent
Follows exact same pattern as the working telecom agent.
Tools, LLM, and agent all in one place.
"""

import sqlite3
import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Load .env from same folder as this script — same as telecom agent
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sage_data.db")

# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
@tool
def get_schema(input: str = "") -> str:
    """
    Returns the schema of the uploaded dataset — column names and types.
    Call this before writing SQL to know exact column names.
    """
    if not os.path.exists(DB_PATH):
        return "No data loaded yet. Please upload a file first."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(sales)")
        cols = cur.fetchall()
        conn.close()
        if not cols:
            return "No data loaded yet."
        schema = "Table: sales\nColumns:\n"
        for col in cols:
            schema += f"  - {col[1]} ({col[2]})\n"
        return schema
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def run_sql(query: str) -> str:
    """
    Executes a SQL SELECT query against the uploaded dataset.
    Table is always called 'sales'. Only SELECT is allowed.
    Use get_schema first to know column names.
    """
    if not os.path.exists(DB_PATH):
        return "No data loaded yet."
    query = query.strip()
    if not query.upper().startswith("SELECT"):
        return "Only SELECT queries are allowed."
    try:
        conn = sqlite3.connect(DB_PATH)
        result = pd.read_sql_query(query, conn)
        conn.close()
        if result.empty:
            return "Query returned 0 rows."
        return result.head(15).to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"


@tool
def calculate_kpis(input: str = "") -> str:
    """
    Calculates key business KPIs — revenue, profit, margin, top performers.
    Call this when the user wants a business overview or briefing.
    """
    if not os.path.exists(DB_PATH):
        return "No data loaded yet."
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM sales", conn)
        conn.close()

        kpis = {"total_rows": len(df), "columns": df.columns.tolist()}

        for col in ["revenue", "profit", "units_sold"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                kpis[f"total_{col}"] = round(float(vals.sum()), 2)
                kpis[f"avg_{col}"] = round(float(vals.mean()), 2)

        if "revenue" in df.columns and "profit" in df.columns:
            rev = pd.to_numeric(df["revenue"], errors="coerce").sum()
            prof = pd.to_numeric(df["profit"], errors="coerce").sum()
            if rev > 0:
                kpis["margin_pct"] = round(prof / rev * 100, 1)

        for grp in ["region", "category", "product", "channel"]:
            if grp in df.columns and "revenue" in df.columns:
                grouped = pd.to_numeric(df["revenue"], errors="coerce").groupby(df[grp]).sum().sort_values(ascending=False)
                kpis[f"top_{grp}"] = grouped.index[0]
                kpis[f"revenue_by_{grp}"] = grouped.round(2).head(5).to_dict()

        return json.dumps(kpis, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Sage — a friendly AI business analyst that helps people understand their data.

Your personality:
- Speak in plain English — no jargon
- You are a decision HELPER not a decision MAKER
- Never say "you should" — say "one option is" or "you could consider"
- Keep answers concise — 3-5 key points maximum
- Always end with one follow-up question

Your workflow:
1. For overviews — call calculate_kpis first, then interpret results
2. For specific questions — call get_schema first, then run_sql
3. Always explain WHY a number matters, not just what it is"""


# ─────────────────────────────────────────────
# BUILD AGENT — same pattern as telecom agent
# ─────────────────────────────────────────────
def build_agent():
    # Read key directly from .env file — guaranteed to work
    api_key = None
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    except:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(f".env file not found or key missing. Looked in: {env_path}")

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        api_key=api_key
    )
    memory = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=[get_schema, run_sql, calculate_kpis],
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent
