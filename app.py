"""
app.py — Sage (modern minimal, dataset-agnostic)

A clean, white, professional Streamlit UI. Builds itself dynamically from
whatever dataset the user uploads — no hardcoded business columns. Adds:
  • Dashboard: KPI strip + auto-charts based on column types
  • Chat: tool-call trace, inline charts, dataset-aware quick prompts,
    suggested follow-ups after every reply
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
# audiorecorder replaced with st.audio_input
from voice import transcribe_audio, speak, autoplay_audio, speak_thinking

from agent import build_agent
from tools import (
    get_cleaning_report,
    get_dataset_name,
    get_df,
    get_profile,
    load_dataframe,
    quick_prompts_for_dataset,
    smart_read_excel,
)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ─────────────────────────────────────────────────────────────────────────────
# Page config + theme
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sage — wise advice from any dataset",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#1a5c35"
ACCENT_VIVID = "#22a05a"
ACCENT_SOFT = "#e3f2ea"
ACCENT_GLOW = "rgba(34,160,90,0.15)"
GOLD = "#b8873a"
GOLD_SOFT = "#fdf3e3"
INK = "#0d1a10"
MUTED = "#5a7060"
LINE = "#e2e8e3"
SIDEBAR_BG = "#0b1810"
CARD_BG = "#ffffff"
PAGE_BG = "#f4f7f4"
SHADOW_SM = "0 1px 4px rgba(13,26,16,0.07), 0 2px 12px rgba(13,26,16,0.05)"
SHADOW_MD = "0 4px 24px rgba(13,26,16,0.10), 0 1px 4px rgba(13,26,16,0.06)"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* ── Background ── */
.stApp {{
    background: #faf8ff;
    background-image:
        radial-gradient(ellipse at 10% 10%, rgba(200,168,233,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(249,196,210,0.2) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 40%, rgba(175,169,236,0.1) 0%, transparent 50%);
}}

/* ── Topbar ── */
header[data-testid="stHeader"] {{
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(12px);
    border-bottom: 0.5px solid rgba(200,168,233,0.3);
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: rgba(250,248,255,0.95);
    border-right: 0.5px solid rgba(200,168,233,0.3);
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(200,168,233,0.12);
    border-radius: 10px;
    padding: 3px;
    gap: 2px;
    border: none;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 7px;
    color: #7F77DD;
    font-size: 13px;
    font-weight: 400;
    padding: 6px 16px;
    border: none;
}}
.stTabs [aria-selected="true"] {{
    background: white;
    color: #26215C;
    font-weight: 500;
    box-shadow: 0 1px 4px rgba(83,74,183,0.12);
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none;
}}

/* ── KPI Cards ── */
.kpi-card {{
    background: rgba(255,255,255,0.8);
    border: 0.5px solid rgba(200,168,233,0.3);
    border-radius: 14px;
    padding: 16px 18px;
    backdrop-filter: blur(8px);
}}
.kpi-label {{
    font-size: 10px;
    color: #AFA9EC;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: 24px;
    font-weight: 500;
    color: #26215C;
    margin-bottom: 3px;
}}
.kpi-sub {{
    font-size: 11px;
    color: #7F77DD;
}}

/* ── Chart Cards ── */
.chart-card {{
    background: rgba(255,255,255,0.8);
    border: 0.5px solid rgba(200,168,233,0.3);
    border-radius: 14px;
    padding: 16px 18px;
    backdrop-filter: blur(8px);
    margin-bottom: 12px;
}}

/* ── Chat banner ── */
.chat-banner {{
    background: linear-gradient(135deg, rgba(200,168,233,0.2), rgba(249,196,210,0.2));
    border: 0.5px solid rgba(200,168,233,0.4);
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 13px;
    color: #534AB7;
    margin-bottom: 12px;
}}

/* ── User bubble ── */
.bubble-user {{
    background: linear-gradient(135deg, #7F77DD, #D4537E);
    border-radius: 14px 14px 2px 14px;
    padding: 10px 16px;
    color: white;
    font-size: 14px;
    margin: 6px 0 6px auto;
    max-width: 68%;
    width: fit-content;
}}

/* ── Agent bubble ── */
.bubble-meta {{
    font-size: 11px;
    color: #AFA9EC;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 5px;
}}
.bubble-agent {{
    background: rgba(255,255,255,0.85);
    border: 0.5px solid rgba(200,168,233,0.3);
    border-radius: 2px 14px 14px 14px;
    padding: 12px 16px;
    font-size: 14px;
    color: #26215C;
    line-height: 1.7;
    backdrop-filter: blur(8px);
}}

/* ── Key finding ── */
.key-finding {{
    background: linear-gradient(135deg, rgba(200,168,233,0.15), rgba(249,196,210,0.15));
    border-left: 2px solid #c8a8e9;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin-top: 8px;
    font-size: 12px;
    color: #534AB7;
    font-style: italic;
}}

/* ── Tool trace ── */
.tool-trace {{
    background: rgba(200,168,233,0.1);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
    color: #AFA9EC;
    margin-top: 4px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}

/* ── Follow-up buttons ── */
.follow-label {{
    font-size: 10px;
    color: #AFA9EC;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 8px 0 4px;
}}

/* ── Chat input ── */
.stChatInput {{
    background: rgba(255,255,255,0.85) !important;
    border: 0.5px solid rgba(200,168,233,0.5) !important;
    border-radius: 14px !important;
}}
.stChatInput textarea {{
    color: #26215C !important;
    font-size: 13px !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: rgba(255,255,255,0.7);
    border: 0.5px solid rgba(200,168,233,0.5);
    border-radius: 20px;
    color: #534AB7;
    font-size: 12px;
    padding: 5px 14px;
    transition: all 0.15s;
}}
.stButton > button:hover {{
    background: rgba(200,168,233,0.15);
    border-color: rgba(127,119,221,0.5);
}}

/* ── File uploader ── */
.stFileUploader {{
    background: rgba(255,255,255,0.7);
    border: 1.5px dashed rgba(200,168,233,0.6);
    border-radius: 14px;
    padding: 8px;
}}

/* ── Expander ── */
.streamlit-expanderHeader {{
    background: rgba(200,168,233,0.08);
    border-radius: 8px;
    color: #7F77DD;
    font-size: 12px;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(200,168,233,0.4); border-radius: 2px; }}

/* ── Sidebar brand ── */
.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 12px;
}}
.sidebar-brand-name {{
    font-size: 16px;
    font-weight: 500;
    color: #26215C;
}}
.sidebar-brand-tag {{
    font-size: 11px;
    color: #AFA9EC;
}}
.sidebar-h {{
    font-size: 10px;
    color: #AFA9EC;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 12px 0 6px;
}}

/* ── Top bar dataset pill ── */
.dataset-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.7);
    border: 0.5px solid rgba(200,168,233,0.5);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #534AB7;
}}
.dataset-dot {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #5DCAA5;
    display: inline-block;
}}

/* ── Empty state ── */
.empty {{
    text-align: center;
    padding: 40px 20px;
}}
.empty-icon {{
    font-size: 32px;
    display: block;
    margin-bottom: 12px;
    background: linear-gradient(135deg, #7F77DD, #D4537E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.empty-title {{
    font-size: 22px;
    font-weight: 500;
    color: #26215C;
    margin-bottom: 6px;
}}
.empty-desc {{
    font-size: 13px;
    color: #7F77DD;
    max-width: 400px;
    margin: 0 auto;
}}

/* ── Upload zone ── */
.upload-zone {{
    background: rgba(255,255,255,0.7);
    border: 1.5px dashed rgba(200,168,233,0.6);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    cursor: pointer;
}}
.upload-label {{
    font-size: 13px;
    color: #534AB7;
    margin-top: 8px;
}}

/* ── Section headers ── */
.h-section {{
    font-size: 11px;
    font-weight: 500;
    color: #AFA9EC;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 16px 0 8px;
}}

/* ── Keyframe Animations (kept for chart cards) ── */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Dataframe table ── */
[data-testid="stDataFrame"] {{
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 0.5px solid rgba(200,168,233,0.3) !important;
}}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border: none !important;
}}

/* ── Chip Tags ── */
.chip {{
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    background: rgba(200,168,233,0.15); color: #534AB7;
    font-size: 11px; font-weight: 500; margin: 2px 3px;
    border: 0.5px solid rgba(200,168,233,0.4);
}}

/* ── Sidebar dataset card ── */
.sidebar-dataset-card {{
    background: rgba(200,168,233,0.08);
    border: 0.5px solid rgba(200,168,233,0.3);
    border-radius: 10px; padding: 10px 12px;
    margin-bottom: 6px;
}}
.sidebar-dataset-name {{
    font-size: 13px; font-weight: 500; color: #26215C;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.sidebar-dataset-meta {{
    font-size: 11px; color: #AFA9EC; margin-top: 2px;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],          # list of {role, content, tool_trace?, charts?}
        "thread_id": str(uuid.uuid4()),
        "agent": None,
        "data_loaded": False,
        "df_name": None,
        "data_changes": [],
        "pending_question": None,
        "voice_mode": False,
        "pending_audio": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.agent is None:
        st.session_state.agent = build_agent()


_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Charting helpers — dataset-agnostic
# ─────────────────────────────────────────────────────────────────────────────
PALETTE_BAR = ["#7F77DD", "#AFA9EC", "#534AB7", "#C8A8E9", "#D4537E", "#F9C4D2"]
PALETTE_PIE = ["#7F77DD", "#534AB7", "#AFA9EC", "#C8A8E9", "#D4537E", "#F9C4D2", "#b8873a"]


def _theme(fig: go.Figure, title: str = "", height: int = 280) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color=INK, family="Inter", weight=600),
            x=0, xanchor="left", y=0.98,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MUTED, size=11),
        margin=dict(l=8, r=8, t=44, b=12),
        height=height,
        xaxis=dict(showgrid=False, color=MUTED, tickfont=dict(size=10), linecolor=LINE),
        yaxis=dict(showgrid=True, gridcolor=LINE, gridwidth=1, color=MUTED,
                   zeroline=False, tickfont=dict(size=10)),
        hoverlabel=dict(
            bgcolor=CARD_BG, bordercolor=LINE,
            font=dict(family="Inter", size=12, color=INK),
        ),
    )
    return fig


def chart_bar(df: pd.DataFrame, x: str, y: str, title: str = "", height: int = 300, horizontal: bool = False) -> go.Figure:
    if horizontal:
        fig = px.bar(df, x=y, y=x, orientation="h",
                     color_discrete_sequence=PALETTE_BAR)
    else:
        fig = px.bar(df, x=x, y=y, color_discrete_sequence=PALETTE_BAR)
    fig.update_traces(marker_line_width=0, marker_cornerradius=4)
    return _theme(fig, title, height)


def chart_line(df: pd.DataFrame, x: str, y: str, title: str = "", height: int = 300) -> go.Figure:
    fig = px.line(df, x=x, y=y, markers=True,
                  color_discrete_sequence=["#7F77DD"])
    fig.update_traces(
        line_width=2.5, marker_size=6,
        marker_color=CARD_BG, marker_line_color="#7F77DD",
        marker_line_width=2,
        fill="tozeroy",
        fillcolor="rgba(127,119,221,0.08)",
    )
    return _theme(fig, title, height)


def chart_hist(series: pd.Series, title: str = "", height: int = 260) -> go.Figure:
    fig = px.histogram(series.dropna(), nbins=24,
                       color_discrete_sequence=[ACCENT_VIVID])
    fig.update_traces(marker_line_width=0, marker_cornerradius=3)
    fig.update_layout(bargap=0.04, showlegend=False)
    return _theme(fig, title, height)


def chart_pie(df: pd.DataFrame, names: str, values: str, title: str = "", height: int = 300) -> go.Figure:
    fig = px.pie(df, names=names, values=values, hole=0.58,
                 color_discrete_sequence=PALETTE_PIE)
    fig.update_traces(
        textposition="outside", textinfo="label+percent",
        marker=dict(line=dict(color=PAGE_BG, width=2)),
    )
    return _theme(fig, title, height)


# ─────────────────────────────────────────────────────────────────────────────
# Number formatting
# ─────────────────────────────────────────────────────────────────────────────
def fmt_num(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return "—"
    if pd.isna(v):
        return "—"
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.2f}B"
    if a >= 1e6:
        return f"{v/1e6:.2f}M"
    if a >= 1e3:
        return f"{v/1e3:.1f}K"
    if a >= 1 or v == 0:
        return f"{v:,.0f}" if a >= 10 else f"{v:.2f}"
    return f"{v:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — upload + dataset summary + filters + quick prompts
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        # Brand
        st.markdown(
            f"""
            <div class="sidebar-brand">
              <div class="sidebar-brand-mark">🌿</div>
              <div>
                <div class="sidebar-brand-name">Sage</div>
                <div class="sidebar-brand-tag">Every dataset has a story. Sage tells it.</div>
              </div>
            </div>
            <div style="height:1px;background:rgba(255,255,255,0.07);margin:0.9rem 0 1rem"></div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.data_loaded:
            return

        # Dataset card
        df = get_df()
        profile = get_profile() or {}
        st.markdown('<div class="sidebar-h">Active Dataset</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="sidebar-dataset-card">
                <div class="sidebar-dataset-name">📄 {st.session_state.df_name}</div>
                <div class="sidebar-dataset-meta">{len(df):,} rows · {len(df.columns)} columns</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.session_state.data_changes:
            with st.expander(f"🧹  Cleaned {len(st.session_state.data_changes)} thing(s)", expanded=False):
                for c in st.session_state.data_changes:
                    st.markdown(f"• {c}")

        # Filters
        cats = (profile.get("classes", {}) or {}).get("categorical", [])
        usable = [c for c in cats if df[c].nunique(dropna=True) <= 30][:4]
        if usable:
            st.markdown('<div class="sidebar-h">Filters</div>', unsafe_allow_html=True)
        for c in usable:
            opts = ["All"] + sorted([str(x) for x in df[c].dropna().unique().tolist()])
            st.selectbox(c, opts, key=f"filter_{c}")
        st.session_state["_filter_cols"] = usable

        # Quick prompts
        prompts = quick_prompts_for_dataset()
        if prompts:
            st.markdown('<div class="sidebar-h">Quick Questions</div>', unsafe_allow_html=True)
            for p in prompts:
                if st.button(p, key=f"qp_{p}"):
                    st.session_state.pending_question = p
                    st.rerun()

        st.markdown(
            '<div style="height:1px;background:rgba(255,255,255,0.07);margin:1.1rem 0 0.8rem"></div>',
            unsafe_allow_html=True,
        )
        if st.button("↺  New conversation", key="new_conv"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()


render_sidebar()


# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────
def render_topbar() -> None:
    df = get_df()
    rows_meta = ""
    if df is not None:
        rows_meta = (
            f'<div class="topbar-meta">'
            f'<span class="topbar-dot"></span>'
            f"<b>{len(df):,}</b>&nbsp;rows&nbsp;·&nbsp;<b>{len(df.columns)}</b>&nbsp;cols&nbsp;·&nbsp;"
            f"<b>{st.session_state.df_name}</b>"
            f"</div>"
        )
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="brand-mark">🌿</div>
            <div>
              <div class="brand-name">Sage</div>
              <div class="brand-tag">Every dataset has a story. Sage tells it.</div>
            </div>
          </div>
          {rows_meta}
        </div>
        """,
        unsafe_allow_html=True,
    )


render_topbar()


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:
    # ── Hero section ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
    /* Landing page layout */
    .landing-hero {{
        text-align: center;
        padding: 3rem 1rem 2rem;
        animation: fadeUp 0.5s ease;
    }}
    .landing-icon {{
        width: 80px; height: 80px; border-radius: 22px;
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_VIVID} 100%);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.4rem;
        font-size: 2.2rem; line-height: 1;
        box-shadow: 0 12px 32px rgba(34,160,90,0.25);
    }}
    .landing-title {{
        font-family: 'Fraunces', serif;
        font-size: 2.8rem; font-weight: 700;
        color: {INK}; letter-spacing: -0.03em;
        line-height: 1.1; margin-bottom: 0.75rem;
    }}
    .landing-sub {{
        font-size: 1rem; color: {MUTED};
        line-height: 1.65; max-width: 520px;
        margin: 0 auto 2rem;
    }}
    .upload-zone {{
        background: {CARD_BG};
        border: 2px dashed {LINE};
        border-radius: 20px;
        padding: 2.2rem 2rem;
        max-width: 560px;
        margin: 0 auto;
        box-shadow: {SHADOW_MD};
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .upload-zone:hover {{
        border-color: {ACCENT_VIVID};
        box-shadow: 0 8px 32px rgba(34,160,90,0.12);
    }}
    .upload-label {{
        font-size: 0.75rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: {MUTED}; margin-bottom: 0.6rem;
    }}
    .feature-row {{
        display: flex; justify-content: center;
        gap: 1.5rem; margin-top: 2.5rem; flex-wrap: wrap;
    }}
    .feature-item {{
        display: flex; align-items: center; gap: 0.45rem;
        font-size: 0.82rem; color: {MUTED}; font-weight: 500;
    }}
    .feature-dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: {ACCENT_VIVID}; flex-shrink: 0;
    }}
    </style>

    <div class="landing-hero">
      <div class="landing-icon">🌿</div>
      <div class="landing-title">Every dataset has a story.<br>Sage tells it.</div>
      <div class="landing-sub">Upload any CSV or Excel file. Ask questions out loud. Hear the answers spoken back with the reasoning behind them — no SQL, no dashboard, no training needed.</div>
      <div class="feature-row">
        <div class="feature-item"><div class="feature-dot"></div>Voice-first analytics</div>
        <div class="feature-item"><div class="feature-dot"></div>8 reasoning tools</div>
        <div class="feature-item"><div class="feature-dot"></div>Any dataset, instant</div>
        <div class="feature-item"><div class="feature-dot"></div>Spoken insights</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload zone (centred, prominent) ──────────────────────────────────────
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        st.markdown('<div class="upload-label">Upload your data</div>', unsafe_allow_html=True)
        uploaded_main = st.file_uploader(
            "Drop a CSV or Excel file here, or click to browse",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="main_uploader",
        )
        st.markdown(
            f'<div style="text-align:center;margin-top:0.6rem;font-size:0.77rem;color:{MUTED}">'
            f'CSV · XLSX · XLS &nbsp;·&nbsp; Up to 200 MB</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_main is not None:
            try:
                if uploaded_main.name.lower().endswith(".csv"):
                    df_up = pd.read_csv(uploaded_main)
                else:
                    df_up = smart_read_excel(uploaded_main)
                changes = load_dataframe(df_up, dataset_name=uploaded_main.name)
                st.session_state.data_loaded = True
                st.session_state.df_name = uploaded_main.name
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.data_changes = changes
                st.session_state.agent = build_agent()
                st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")

        # Sample dataset shortcut
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sales_data.csv")
        if os.path.exists(sample_path):
            st.markdown('<div style="text-align:center;margin-top:0.9rem">', unsafe_allow_html=True)
            if st.button("⚡  Try with sample dataset instead", key="load_sample_main"):
                df_s = pd.read_csv(sample_path)
                changes = load_dataframe(df_s, dataset_name="sample_sales_data.csv")
                st.session_state.data_loaded = True
                st.session_state.df_name = "sample_sales_data.csv"
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.data_changes = changes
                st.session_state.agent = build_agent()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Feature pills ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="feature-row">
      <div class="feature-item"><div class="feature-dot"></div> Auto dashboard</div>
      <div class="feature-item"><div class="feature-dot"></div> AI analyst chat</div>
      <div class="feature-item"><div class="feature-dot"></div> Trend & breakdown charts</div>
      <div class="feature-item"><div class="feature-dot"></div> Plain-English insights</div>
      <div class="feature-item"><div class="feature-dot"></div> Smart recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────────────────────
def filtered_df() -> pd.DataFrame:
    df = get_df().copy()
    for c in st.session_state.get("_filter_cols", []):
        sel = st.session_state.get(f"filter_{c}", "All")
        if sel and sel != "All":
            df = df[df[c].astype(str) == sel]
    return df


df = filtered_df()
profile = get_profile() or {}
classes = profile.get("classes", {}) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_dash, tab_chat, tab_data = st.tabs(["📊  Dashboard", "💬  Chat with Sage", "🗂  Data preview"])


def explain_chart_with_voice(title: str, description: str) -> None:
    """Generate and play a voice explanation for a chart."""
    from voice import speak, autoplay_audio, _strip_markdown
    prompt = f"In exactly 2 sentences, explain this chart called '{title}': {description}. Be specific with the key insight."
    try:
        agent = st.session_state.get("agent")
        if agent:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
            msgs = result.get("messages", [])
            text = ""
            for msg in reversed(msgs):
                content = str(getattr(msg, "content", ""))
                if content and not getattr(msg, "tool_calls", None):
                    text = content
                    break
            if text:
                audio_b64 = speak(text)
                if audio_b64:
                    st.markdown(autoplay_audio(audio_b64), unsafe_allow_html=True)
                    st.caption(f"🔊 {text[:120]}...")
    except Exception as e:
        st.error(f"Voice error: {e}")


def _speak_chart(label: str) -> None:
    from voice import speak, autoplay_audio
    try:
        agent = st.session_state.get("agent")
        if not agent:
            st.caption("No agent loaded yet.")
            return
        config = {"configurable": {"thread_id": st.session_state.get("thread_id", "default")}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"In 2 sentences, explain the key insight from this chart: {label}. Be specific with numbers from the data."}]},
            config=config,
        )
        msgs = result.get("messages", [])
        text = ""
        for msg in reversed(msgs):
            content = str(getattr(msg, "content", ""))
            if content and not getattr(msg, "tool_calls", None):
                text = content
                break
        if text:
            audio = speak(text)
            if audio:
                st.markdown(autoplay_audio(audio), unsafe_allow_html=True)
                st.caption(f"🔊 {text[:180]}")
    except Exception as e:
        st.caption(f"Voice error: {e}")


def voice_chart_button(label: str, key: str) -> None:
    if st.button("🔊 Explain", key=key, help="Hear Sage explain this chart"):
        with st.spinner("Sage is thinking..."):
            _speak_chart(label)


# ── DASHBOARD ───────────────────────────────────────────────────────────────
def render_dashboard(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No rows match the current filters.")
        return

    numeric_cols = classes.get("numeric", [])
    categorical_cols = classes.get("categorical", [])
    datetime_cols = classes.get("datetime", [])

    # Pick the "primary metric" — numeric column with the largest sum
    # (good proxy for the most-meaningful column to highlight).
    primary_metric: Optional[str] = None
    if numeric_cols:
        sums = {c: pd.to_numeric(df[c], errors="coerce").abs().sum() for c in numeric_cols}
        primary_metric = max(sums, key=sums.get) if sums else numeric_cols[0]

    # ── KPI strip ───────────────────────────────────────────────────────────
    kpi_targets: List[Dict[str, Any]] = []
    kpi_targets.append({
        "label": "Rows",
        "value": fmt_num(len(df)),
        "sub": f"{len(df.columns)} columns",
    })
    if primary_metric:
        s = pd.to_numeric(df[primary_metric], errors="coerce")
        kpi_targets.append({
            "label": f"Total {primary_metric}",
            "value": fmt_num(s.sum()),
            "sub": f"avg {fmt_num(s.mean())}",
        })
    # Other top numeric columns
    other_nums = [c for c in numeric_cols if c != primary_metric][:3]
    for c in other_nums:
        s = pd.to_numeric(df[c], errors="coerce")
        kpi_targets.append({
            "label": f"Avg {c}",
            "value": fmt_num(s.mean()),
            "sub": f"min {fmt_num(s.min())} · max {fmt_num(s.max())}",
        })
    if categorical_cols:
        c = categorical_cols[0]
        kpi_targets.append({
            "label": f"Unique {c}",
            "value": fmt_num(df[c].nunique(dropna=True)),
            "sub": f"top: {df[c].mode().iloc[0] if df[c].notna().any() else '—'}",
        })

    cols = st.columns(min(len(kpi_targets), 6))
    for i, t in enumerate(kpi_targets[:6]):
        with cols[i]:
            st.markdown(
                f"""<div class="kpi-card">
                    <div class="kpi-label">{t['label']}</div>
                    <div class="kpi-value">{t['value']}</div>
                    <div class="kpi-sub">{t['sub']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Time series (if any datetime + numeric) ─────────────────────────────
    if datetime_cols and primary_metric:
        st.markdown('<div class="h-section">📈 Trend over time</div>', unsafe_allow_html=True)
        date_col = datetime_cols[0]
        d = df[[date_col, primary_metric]].copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d[primary_metric] = pd.to_numeric(d[primary_metric], errors="coerce")
        d = d.dropna()
        if not d.empty:
            span_days = (d[date_col].max() - d[date_col].min()).days
            freq = "D" if span_days <= 90 else "W" if span_days <= 365 * 2 else "M"
            ts = d.set_index(date_col)[primary_metric].resample(freq).sum().reset_index()
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(
                chart_line(ts, date_col, primary_metric, f"{primary_metric} over time"),
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)
            voice_chart_button(f"{primary_metric} over time showing monthly trends", "vc_trend")

    # ── Categorical breakdowns ──────────────────────────────────────────────
    breakdown_cats = [c for c in categorical_cols if df[c].nunique(dropna=True) <= 25][:4]
    if breakdown_cats and primary_metric:
        st.markdown('<div class="h-section">🔍 Breakdown</div>', unsafe_allow_html=True)
        rows = [breakdown_cats[i:i + 2] for i in range(0, len(breakdown_cats), 2)]
        for row in rows:
            cs = st.columns(len(row))
            for i, cat in enumerate(row):
                with cs[i]:
                    g = (
                        pd.to_numeric(df[primary_metric], errors="coerce")
                        .groupby(df[cat]).sum()
                        .sort_values(ascending=False).head(10)
                        .reset_index()
                    )
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    if g[cat].nunique() <= 6:
                        st.plotly_chart(
                            chart_pie(g, cat, primary_metric, f"{primary_metric} by {cat}"),
                            use_container_width=True,
                        )
                    else:
                        st.plotly_chart(
                            chart_bar(g, cat, primary_metric, f"{primary_metric} by {cat}"),
                            use_container_width=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    voice_chart_button(f"{primary_metric} by {cat} breakdown", f"vc_1_{cat}")

    # ── Distributions ───────────────────────────────────────────────────────
    if numeric_cols:
        st.markdown('<div class="h-section">〜 Distributions</div>', unsafe_allow_html=True)
        show_cols = numeric_cols[:3]
        cs = st.columns(len(show_cols))
        for i, c in enumerate(show_cols):
            with cs[i]:
                s = pd.to_numeric(df[c], errors="coerce")
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.plotly_chart(chart_hist(s, f"{c} distribution"), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                voice_chart_button(f"{c} distribution histogram", f"vc_2_{c}")

    # ── Correlations heatmap ───────────────────────────────────────────────
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] >= 3:
        st.markdown('<div class="h-section">⬡ Numeric correlations</div>', unsafe_allow_html=True)
        corr = nums.corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale=[[0, "#f0f9f4"], [0.5, "#4dc886"], [1, ACCENT]],
            zmin=-1, zmax=1,
        )
        _theme(fig, "", height=max(280, 30 * len(corr) + 80))
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        voice_chart_button("numeric correlations heatmap showing relationships between columns", "vc_3")


with tab_dash:
    render_dashboard(df)


# ── DATA PREVIEW ─────────────────────────────────────────────────────────
with tab_data:
    st.markdown('<div class="h-section">⬡ First 200 rows</div>', unsafe_allow_html=True)
    st.dataframe(
        df.head(200), use_container_width=True, hide_index=True,
        column_config={c: st.column_config.Column(width="medium") for c in df.columns[:10]},
    )

    if classes:
        st.markdown('<div class="h-section">◈ Detected column types</div>', unsafe_allow_html=True)
        chips = []
        for kind, cols_list in classes.items():
            for c in cols_list:
                chips.append(f'<span class="chip">{c} · {kind}</span>')
        st.markdown("<div style='line-height:2.2'>" + " ".join(chips) + "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────────────────────────────────────
def _content_to_str(content: Any) -> str:
    if isinstance(content, list):
        out = []
        for c in content:
            if isinstance(c, dict):
                out.append(c.get("text", ""))
            else:
                out.append(str(c))
        return " ".join(out)
    return str(content) if content is not None else ""


def _build_inline_chart_for_tool(name: str, args: Dict[str, Any]) -> Optional[go.Figure]:
    """Re-execute a tool's intent in the UI to render a small chart."""
    cur = get_df()
    if cur is None or cur.empty:
        return None
    try:
        if name == "top_n":
            group_by = args.get("group_by")
            metric = args.get("metric")
            n = int(args.get("n", 5))
            asc = bool(args.get("ascending", False))
            if group_by in cur.columns and metric in cur.columns:
                s = pd.to_numeric(cur[metric], errors="coerce")
                g = (
                    s.groupby(cur[group_by]).sum()
                    .sort_values(ascending=asc).head(n)
                    .reset_index()
                )
                return chart_bar(g, group_by, metric, f"{group_by} by {metric}", height=240)
        if name == "value_counts":
            col = args.get("column")
            top = int(args.get("top_n", 10))
            if col in cur.columns:
                vc = cur[col].value_counts(dropna=True).head(top).reset_index()
                vc.columns = [col, "count"]
                return chart_bar(vc, col, "count", f"Top values in {col}", height=240)
        if name == "time_series":
            date_col = args.get("date_column")
            metric = args.get("metric")
            freq = args.get("freq", "M")
            if date_col in cur.columns and metric in cur.columns:
                d = cur[[date_col, metric]].copy()
                d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
                d[metric] = pd.to_numeric(d[metric], errors="coerce")
                d = d.dropna()
                if not d.empty:
                    ts = d.set_index(date_col)[metric].resample(freq).sum().reset_index()
                    return chart_line(ts, date_col, metric, f"{metric} over time", height=240)
    except Exception:
        return None
    return None


def run_agent_with_trace(question: str) -> Dict[str, Any]:
    """
    Invoke the agent and return:
      { content: str, trace: [ {tool, args, result} ], charts: [Figure] }
    """
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}, config=config
        )
    except Exception as e:
        return {"content": f"Error: {e}", "trace": [], "charts": []}

    msgs = result.get("messages", [])
    trace: List[Dict[str, Any]] = []
    charts: List[go.Figure] = []
    final_text = ""

    pending_tool_calls: Dict[str, Dict[str, Any]] = {}
    for msg in msgs:
        # AI message with tool calls
        tcs = getattr(msg, "tool_calls", None) or []
        if tcs:
            for tc in tcs:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                if tc_id:
                    pending_tool_calls[tc_id] = {"name": tc_name, "args": tc_args or {}}
        # Tool result message
        if msg.__class__.__name__ == "ToolMessage":
            tc_id = getattr(msg, "tool_call_id", None)
            content = _content_to_str(msg.content)
            meta = pending_tool_calls.get(tc_id, {"name": getattr(msg, "name", "tool"), "args": {}})
            trace.append({"tool": meta["name"], "args": meta["args"], "result": content[:800]})
            chart = _build_inline_chart_for_tool(meta["name"], meta["args"])
            if chart is not None:
                charts.append(chart)

    # Final assistant text — last AI message with content and no further tool calls
    for msg in reversed(msgs):
        content = _content_to_str(getattr(msg, "content", ""))
        if content and not (getattr(msg, "tool_calls", None)):
            final_text = content
            break
    if not final_text and msgs:
        final_text = _content_to_str(getattr(msgs[-1], "content", ""))

    return {"content": final_text, "trace": trace, "charts": charts}


def _parse_row_count(tool_name: str, result: str) -> Optional[int]:
    if not result:
        return None
    try:
        data = json.loads(result)
        if isinstance(data, dict):
            if "outlier_count" in data:
                return int(data["outlier_count"])
            if "rows" in data:
                return int(data["rows"])
        return None
    except Exception:
        pass
    lines = [l for l in result.strip().split("\n") if l.strip()]
    return max(0, len(lines) - 1) if len(lines) > 1 else None


def _extract_key_finding(content: str) -> str:
    m = re.search(r"(?:🔍[^\n]*Key Finding|Key Finding[^\n]*🔍)[^\n]*\n+(.+?)(?:\n|$)", content, re.IGNORECASE)
    if m:
        line = re.sub(r"\*+", "", m.group(1)).strip()
        if len(line) > 20:
            return line
    for line in content.split("\n"):
        clean = re.sub(r"\*+|#+|^[-•]\s*", "", line).strip()
        if clean and len(clean) > 30 and not any(clean.startswith(e) for e in ["🔍", "📊", "⚡", "🎯", "💬"]):
            return clean
    return ""


def _render_message(msg: Dict[str, Any]) -> None:
    if msg["role"] == "user":
        st.markdown(
            f"""<div class="bubble-user">
                <div class="bubble-meta">You</div>
                {msg['content']}
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Tool trace (collapsed)
    trace = msg.get("trace") or []
    if trace:
        with st.expander(f"⚙️  Sage ran {len(trace)} tool call{'s' if len(trace)!=1 else ''}", expanded=False):
            for step in trace:
                tool_name = step["tool"]
                args = step.get("args") or {}
                result = step.get("result", "")
                row_count = _parse_row_count(tool_name, result)
                count_badge = f"  ·  {row_count} rows" if row_count is not None and row_count > 0 else ""
                st.markdown(
                    f"<div class='tool-trace'><b>{tool_name}</b>{count_badge}</div>",
                    unsafe_allow_html=True,
                )
                if tool_name == "run_sql" and args.get("query"):
                    st.code(args["query"], language="sql")
                if result:
                    st.code(result[:600], language="text")

    # Render label then body; use st.markdown for native markdown rendering
    st.markdown(
        f'<div class="bubble-meta agent" style="margin:0.6rem 0 0.2rem">🌿 Sage · Analyst</div>',
        unsafe_allow_html=True,
    )
    content = msg.get("content") or ""
    # Render as native Streamlit markdown so **bold**, bullets, headers all work
    st.markdown(content)

    # One-sentence plain-English interpretation
    interp = _extract_key_finding(content)
    if interp:
        st.markdown(
            f'<div style="font-size:0.82rem;color:{MUTED};font-style:italic;'
            f'margin:0.3rem 0 0.6rem;padding:0.5rem 0.75rem;'
            f'border-left:3px solid {ACCENT_VIVID};background:{ACCENT_SOFT};'
            f'border-radius:0 8px 8px 0;">{interp}</div>',
            unsafe_allow_html=True,
        )

    # Anomaly detect warning
    for step in trace:
        if step["tool"] == "anomaly_detect" and step.get("result"):
            try:
                data = json.loads(step["result"])
                count = data.get("outlier_count", 0)
                if count > 0:
                    col_name = data.get("column", "column")
                    st.warning(f"⚠️  {count} outlier{'s' if count != 1 else ''} detected in **{col_name}** (IQR method).")
            except Exception:
                pass

    # Inline charts inside a card
    for fig in msg.get("charts") or []:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def _suggest_followups(profile: Dict[str, Any], asked: List[str]) -> List[str]:
    """Pick 3 dataset-aware follow-ups not yet asked."""
    pool = quick_prompts_for_dataset()
    asked_set = {a.lower() for a in asked}
    out = [p for p in pool if p.lower() not in asked_set]
    return out[:3]


with tab_chat:
    st.markdown(
        f"""<div class="chat-banner">
          <span class="chat-banner-icon">✦</span>
          <span>Every dataset has a story — ask Sage what yours is saying right now.</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Process pending question (from sidebar quick prompt)
    if st.session_state.pending_question:
        pq = st.session_state.pending_question
        st.session_state.pending_question = None
        st.session_state.messages.append({"role": "user", "content": pq})
        with st.spinner("Sage is thinking…"):
            r = run_agent_with_trace(pq)
        st.session_state.messages.append({
            "role": "assistant",
            "content": r["content"],
            "trace": r["trace"],
            "charts": r["charts"],
        })

    messages_area = st.container()

    # Voice + text input row
    user_input = st.chat_input("Ask Sage about your data…")

    st.markdown("""
<style>
div[data-testid="stChatInput"] > div {
    border: 0.5px solid rgba(200,168,233,0.5) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.85) !important;
}
</style>
""", unsafe_allow_html=True)

    audio_value = st.audio_input("🎤", key="voice_recorder", label_visibility="collapsed")
    if audio_value is not None:
        audio_bytes = audio_value.read()
        audio_hash = hash(audio_bytes)
        if audio_hash != st.session_state.get("_last_audio_hash"):
            st.session_state["_last_audio_hash"] = audio_hash
            with st.spinner("Sage is listening..."):
                transcript = transcribe_audio(audio_bytes)
            if transcript and not transcript.startswith("Transcription error"):
                st.info(f'🎤 You said: "{transcript}"')
                st.session_state.messages.append({"role": "user", "content": transcript})
                thinking_b64 = speak_thinking()
                if thinking_b64:
                    st.markdown(autoplay_audio(thinking_b64), unsafe_allow_html=True)
                with st.spinner("Sage is thinking..."):
                    r = run_agent_with_trace(transcript)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": r["content"],
                    "trace": r["trace"],
                    "charts": r["charts"],
                })
                if r["content"]:
                    speech_text = r["content"][:500]
                    answer_b64 = speak(speech_text)
                    if answer_b64:
                        st.session_state["pending_audio"] = answer_b64
                st.rerun()

    # Autoplay pending audio answer
    if st.session_state.get("pending_audio"):
        st.markdown(
            autoplay_audio(st.session_state["pending_audio"]),
            unsafe_allow_html=True,
        )
        st.session_state["pending_audio"] = None

    if user_input and user_input.strip():
        q = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Sage is thinking…"):
            r = run_agent_with_trace(q)
        st.session_state.messages.append({
            "role": "assistant",
            "content": r["content"],
            "trace": r["trace"],
            "charts": r["charts"],
        })

    with messages_area:
        if not st.session_state.messages:
            prompts = quick_prompts_for_dataset()
            st.markdown(
                f"""<div class="empty" style="padding:2.8rem 2rem 1.8rem">
                    <span class="empty-icon">✦</span>
                    <div class="empty-title">What do you want to know?</div>
                    <div class="empty-desc">Ask anything about your data — Sage will analyse it and respond like a senior analyst, with real numbers and clear recommendations.</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if prompts:
                st.markdown(
                    f'<div style="font-size:0.7rem;color:{MUTED};text-transform:uppercase;'
                    f'letter-spacing:0.1em;font-weight:700;margin:0.5rem 0 0.6rem;text-align:center">'
                    f'Try one of these</div>',
                    unsafe_allow_html=True,
                )
                # Render as actual clickable buttons in a grid
                btn_cols = st.columns(min(len(prompts[:4]), 2))
                for qi, prompt_text in enumerate(prompts[:4]):
                    with btn_cols[qi % 2]:
                        if st.button(prompt_text, key=f"starter_{qi}"):
                            st.session_state.pending_question = prompt_text
                            st.rerun()

        for i, msg in enumerate(st.session_state.messages):
            _render_message(msg)

        # Suggested follow-ups after the LAST assistant message
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            asked = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
            follow = _suggest_followups(profile, asked)
            if follow:
                st.markdown('<div class="follow-label">Suggested next</div>', unsafe_allow_html=True)
                cols = st.columns(len(follow))
                for j, q in enumerate(follow):
                    with cols[j]:
                        if st.button(q, key=f"follow_{len(st.session_state.messages)}_{j}"):
                            st.session_state.pending_question = q
                            st.rerun()
