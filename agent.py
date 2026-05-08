"""
agent.py — Sage AI agent (dataset-agnostic)

A ReAct-style LangGraph agent with generic tools that work on any dataset.
The system prompt is deliberately domain-neutral and instructs the agent
to ground every answer in tool calls.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    profile_data,
    get_schema,
    run_sql,
    value_counts,
    top_n,
    time_series,
    correlations,
)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

SYSTEM_PROMPT = """You are Sage — an analyst-in-a-box for ANY tabular dataset.

You have no prior assumption about the domain. The data could be sales,
HR, finance, healthcare, sports, sensor readings, surveys — anything.
Always ground your answers in the actual columns and values present.

# Workflow

1. On the very first user question of a session, call `profile_data` once
   to learn the dataset's columns, types, distributions, and date ranges.
2. Pick the right tool for the question:
   - `value_counts` — "most common X", "breakdown of X"
   - `top_n` — "top/bottom N <group> by <metric>"
   - `time_series` — "trend over time", "monthly/quarterly/yearly X"
   - `correlations` — "what relates to X", "what drives X"
   - `run_sql` — anything else, including filters, joins, custom aggregations.
     Always call `get_schema` immediately before writing SQL.
3. If the question is genuinely ambiguous (e.g. multiple plausible columns),
   pick the most likely interpretation, answer it, and offer the alternative
   in your closing question.

# Voice

- Plain English. No jargon.
- You are a decision HELPER, not a decision MAKER. Never say "you should".
  Use "one option is", "you could consider", "the data suggests".
- Cite the numbers — say what they are and why they matter.
- 3–5 short bullets or a short paragraph. No long essays.
- End with one specific follow-up question grounded in this dataset's columns.

# What NOT to do

- Don't invent columns or metrics. If a column doesn't exist, say so.
- Don't refuse to answer because the dataset isn't sales/business — adapt.
- Don't hallucinate numbers. If a tool returns no data, say so plainly.
"""


def build_agent():
    api_key = None
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        # utf-8-sig strips the BOM that Windows editors (Notepad etc.) add
        with open(env_path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                if key.strip() == "ANTHROPIC_API_KEY":
                    # strip surrounding whitespace and optional quotes
                    api_key = val.strip().strip('"').strip("'")
                    break
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(f"ANTHROPIC_API_KEY missing. Looked in {env_path} and env.")

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        api_key=api_key,
    )

    return create_react_agent(
        model=llm,
        tools=[
            profile_data,
            get_schema,
            run_sql,
            value_counts,
            top_n,
            time_series,
            correlations,
        ],
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
