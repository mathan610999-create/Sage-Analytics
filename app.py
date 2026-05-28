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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: {PAGE_BG};
    color: {INK};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1300px; }}

/* ── Keyframe Animations ── */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}
@keyframes pulse-dot {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%       {{ opacity: 0.5; transform: scale(0.75); }}
}}

/* ── Topbar ── */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 1.2rem 0;
    border-bottom: 1px solid {LINE};
    margin-bottom: 1.6rem;
    animation: fadeIn 0.4s ease;
}}
.brand {{ display: flex; align-items: center; gap: 0.75rem; }}
.brand-mark {{
    width: 38px; height: 38px; border-radius: 11px;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_VIVID} 100%);
    color: white; display: flex; align-items: center;
    justify-content: center; font-size: 1.15rem;
    box-shadow: 0 2px 8px rgba(34,160,90,0.35);
}}
.brand-name {{
    font-family: 'Fraunces', serif; font-size: 1.4rem;
    font-weight: 700; color: {INK}; letter-spacing: -0.02em;
}}
.brand-tag {{ font-size: 0.76rem; color: {MUTED}; margin-top: 1px; letter-spacing: 0.01em; }}
.topbar-meta {{
    display: flex; align-items: center; gap: 0.5rem;
    background: {CARD_BG}; border: 1px solid {LINE};
    border-radius: 999px; padding: 0.35rem 0.85rem;
    font-size: 0.78rem; color: {MUTED};
    box-shadow: {SHADOW_SM};
}}
.topbar-meta b {{ color: {ACCENT}; font-weight: 600; }}
.topbar-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {ACCENT_VIVID};
    display: inline-block; margin-right: 4px;
    animation: pulse-dot 2s ease-in-out infinite;
}}

/* ── KPI Cards ── */
.kpi {{
    background: {CARD_BG};
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 1.1rem 1.25rem 1rem;
    position: relative; overflow: hidden;
    box-shadow: {SHADOW_SM};
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
    animation: fadeUp 0.4s ease both;
}}
.kpi::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, {ACCENT} 0%, {ACCENT_VIVID} 100%);
    border-radius: 16px 16px 0 0;
}}
.kpi:hover {{
    transform: translateY(-2px);
    box-shadow: {SHADOW_MD};
}}
.kpi-label {{
    font-size: 0.68rem; color: {MUTED};
    text-transform: uppercase; letter-spacing: 0.08em;
    font-weight: 600; margin-bottom: 0.5rem;
}}
.kpi-value {{
    font-family: 'Fraunces', serif;
    font-size: 1.7rem; font-weight: 700;
    color: {INK}; line-height: 1; letter-spacing: -0.02em;
}}
.kpi-sub {{
    font-size: 0.72rem; color: {MUTED}; margin-top: 0.45rem;
    padding-top: 0.45rem; border-top: 1px solid {LINE};
}}

/* ── Section Headings ── */
.h-section {{
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.72rem; color: {MUTED};
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700; margin: 2rem 0 0.9rem 0;
}}
.h-section::after {{
    content: ''; flex: 1;
    height: 1px; background: {LINE};
}}

/* ── Chart Card Wrapper ── */
.chart-card {{
    background: {CARD_BG}; border: 1px solid {LINE};
    border-radius: 16px; padding: 1rem 1rem 0.5rem;
    box-shadow: {SHADOW_SM}; margin-bottom: 0.5rem;
    animation: fadeUp 0.5s ease both;
}}

/* ── Empty State ── */
.empty {{
    background: linear-gradient(135deg, {CARD_BG} 0%, #f0f6f2 100%);
    border: 1px solid {LINE}; border-radius: 20px;
    padding: 4rem 2.5rem; text-align: center;
    box-shadow: {SHADOW_SM};
    animation: fadeUp 0.5s ease;
}}
.empty-icon {{
    font-size: 3rem; margin-bottom: 1rem;
    display: flex; align-items: center; justify-content: center;
    width: 72px; height: 72px; border-radius: 20px;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_VIVID} 100%);
    color: white; margin: 0 auto 1.2rem;
    box-shadow: 0 8px 24px rgba(34,160,90,0.3);
    font-size: 2rem; line-height: 1;
}}
.empty-title {{
    font-family: 'Fraunces', serif; font-size: 1.5rem;
    font-weight: 700; color: {INK}; margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}}
.empty-desc {{ font-size: 0.9rem; color: {MUTED}; line-height: 1.6; max-width: 420px; margin: 0 auto; }}

/* ── Sidebar — Dark Premium ── */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG} !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}}
section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.6rem;
}}
section[data-testid="stSidebar"] * {{
    color: rgba(255,255,255,0.85) !important;
}}
/* File uploader — dark sidebar */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {{
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px dashed rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"]:hover {{
    border-color: {ACCENT_VIVID} !important;
    background: rgba(34,160,90,0.1) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] * {{
    color: rgba(255,255,255,0.65) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] svg {{
    fill: rgba(255,255,255,0.4) !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {{
    background: rgba(34,160,90,0.15) !important;
    border: 1px solid rgba(34,160,90,0.3) !important;
    border-radius: 8px !important;
    color: #7ddba0 !important;
}}
/* Selectbox dark */
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.9) !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div:hover {{
    border-color: {ACCENT_VIVID} !important;
}}
section[data-testid="stSidebar"] .stSelectbox svg {{
    fill: rgba(255,255,255,0.5) !important;
}}
.sidebar-h {{
    font-size: 0.65rem; color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin: 1.3rem 0 0.5rem 0;
}}
.sidebar-brand {{
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem;
}}
.sidebar-brand-mark {{
    width: 32px; height: 32px; border-radius: 9px;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_VIVID} 100%);
    display: flex; align-items: center; justify-content: center; font-size: 1rem;
    box-shadow: 0 2px 8px rgba(34,160,90,0.4);
    flex-shrink: 0;
}}
.sidebar-brand-name {{
    font-family: 'Fraunces', serif; font-size: 1.2rem;
    font-weight: 700; color: white !important; letter-spacing: -0.01em;
}}
.sidebar-brand-tag {{
    font-size: 0.75rem; color: rgba(255,255,255,0.4) !important; margin-top: 0px;
}}
.sidebar-dataset-card {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 0.8rem 0.9rem;
    margin-bottom: 0.3rem;
}}
.sidebar-dataset-name {{
    font-size: 0.84rem; font-weight: 600;
    color: rgba(255,255,255,0.92) !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.sidebar-dataset-meta {{
    font-size: 0.72rem; color: rgba(255,255,255,0.4) !important; margin-top: 2px;
}}

/* ── Sidebar Buttons ── */
section[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.82) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.9rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    backdrop-filter: blur(4px) !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(34,160,90,0.18) !important;
    border-color: {ACCENT_VIVID} !important;
    color: #7ddba0 !important;
    transform: translateX(2px) !important;
}}

/* ── Main Buttons ── */
.stButton > button {{
    background: {CARD_BG} !important;
    color: {INK} !important;
    border: 1px solid {LINE} !important;
    border-radius: 10px !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 0.9rem !important;
    transition: all 0.18s ease !important;
    box-shadow: {SHADOW_SM} !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    background: {ACCENT_SOFT} !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px {ACCENT_GLOW} !important;
}}

/* ── Tabs — Pill Style ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD_BG};
    border: 1px solid {LINE}; border-radius: 12px;
    padding: 4px; gap: 2px;
    box-shadow: {SHADOW_SM};
    margin-bottom: 1.2rem;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {MUTED} !important;
    font-weight: 500 !important;
    border-radius: 9px !important;
    padding: 0.5rem 1.1rem !important;
    font-size: 0.875rem !important;
    transition: all 0.18s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {ACCENT_SOFT} !important;
    color: {ACCENT} !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_VIVID} 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px {ACCENT_GLOW} !important;
}}

/* ── Chat Bubbles ── */
.bubble-user {{
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_VIVID} 100%);
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.15rem;
    margin: 0.6rem 0 0.6rem 20%;
    color: white !important; font-size: 0.91rem; line-height: 1.6;
    box-shadow: 0 4px 16px {ACCENT_GLOW};
    animation: fadeUp 0.3s ease;
}}
.bubble-agent {{
    background: {CARD_BG};
    border: 1px solid {LINE};
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 20% 0.6rem 0;
    color: {INK}; font-size: 0.91rem; line-height: 1.7;
    box-shadow: {SHADOW_SM};
    animation: fadeUp 0.3s ease;
}}
.bubble-meta {{
    font-size: 0.64rem; color: rgba(255,255,255,0.7);
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700; margin-bottom: 0.3rem;
}}
.bubble-meta.agent {{
    color: {ACCENT}; font-size: 0.64rem;
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700; margin-bottom: 0.3rem;
    display: flex; align-items: center; gap: 0.3rem;
}}

/* ── Tool Trace ── */
.tool-trace {{
    font-size: 0.76rem; color: #a8c4a8;
    background: #0d1f10; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 0.6rem 0.9rem;
    margin: 0.4rem 20% 0.4rem 0;
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, monospace;
    line-height: 1.5;
}}
.tool-trace b {{ color: {ACCENT_VIVID}; font-weight: 600; }}

/* ── Chat info banner ── */
.chat-banner {{
    background: linear-gradient(135deg, {ACCENT_SOFT} 0%, #f0f9f4 100%);
    border: 1px solid rgba(34,160,90,0.2);
    border-radius: 14px; padding: 0.85rem 1.1rem;
    margin-bottom: 1rem; font-size: 0.85rem; color: {MUTED};
    display: flex; align-items: center; gap: 0.6rem;
    box-shadow: {SHADOW_SM};
}}
.chat-banner-icon {{ font-size: 1.1rem; flex-shrink: 0; }}

/* ── Chip Tags ── */
.chip {{
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    background: {ACCENT_SOFT}; color: {ACCENT};
    font-size: 0.72rem; font-weight: 600; margin: 2px 3px;
    border: 1px solid rgba(34,160,90,0.2);
    letter-spacing: 0.01em;
}}

/* ── Follow-up suggestion pills ── */
.follow-label {{
    font-size: 0.65rem; color: {MUTED};
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700; margin: 0.8rem 0 0.5rem 0;
    display: flex; align-items: center; gap: 0.5rem;
}}
.follow-label::before {{
    content: '✦'; font-size: 0.55rem; color: {GOLD};
}}

/* ── Divider ── */
hr {{ border: none; border-top: 1px solid {LINE}; margin: 1.2rem 0; }}

/* ── Expander styling ── */
.streamlit-expanderHeader {{
    font-size: 0.8rem !important; font-weight: 600 !important;
    color: {MUTED} !important;
    border-radius: 8px !important;
}}

/* ── Success/Error banners ── */
.stSuccess {{ border-radius: 10px !important; }}
.stError {{ border-radius: 10px !important; }}

/* ── Main page file uploader (landing) ── */
.upload-zone [data-testid="stFileUploadDropzone"] {{
    background: {ACCENT_SOFT} !important;
    border: 2px dashed rgba(34,160,90,0.35) !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
    transition: all 0.2s ease !important;
}}
.upload-zone [data-testid="stFileUploadDropzone"]:hover {{
    background: rgba(34,160,90,0.12) !important;
    border-color: {ACCENT_VIVID} !important;
}}
.upload-zone [data-testid="stFileUploadDropzone"] * {{
    color: {ACCENT} !important;
}}
.upload-zone [data-testid="stFileUploadDropzone"] svg {{
    fill: {ACCENT} !important;
}}
/* Sample dataset button — centred subtle link style */
.upload-zone .stButton > button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: {MUTED} !important;
    font-size: 0.82rem !important;
    text-decoration: underline !important;
    padding: 0 !important;
    width: auto !important;
}}
.upload-zone .stButton > button:hover {{
    color: {ACCENT} !important;
    background: transparent !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {LINE}; border-radius: 999px; }}
::-webkit-scrollbar-thumb:hover {{ background: #c5cfc4; }}

/* ── Native markdown styled as agent bubble ── */
.bubble-meta.agent + div[data-testid="stMarkdownContainer"] {{
    background: {CARD_BG};
    border: 1px solid {LINE};
    border-radius: 0 18px 18px 18px;
    padding: 1rem 1.25rem;
    margin: 0 20% 0.8rem 0;
    box-shadow: {SHADOW_SM};
    font-size: 0.91rem;
    line-height: 1.75;
    animation: fadeUp 0.35s ease;
}}
.bubble-meta.agent + div[data-testid="stMarkdownContainer"] strong {{
    color: {ACCENT};
}}
.bubble-meta.agent + div[data-testid="stMarkdownContainer"] p {{
    margin: 0.3rem 0;
}}
.bubble-meta.agent + div[data-testid="stMarkdownContainer"] ul {{
    margin: 0.3rem 0; padding-left: 1.2rem;
}}
.bubble-meta.agent + div[data-testid="stMarkdownContainer"] li {{
    margin: 0.2rem 0;
}}

/* ── Starter prompt buttons (chat empty state) ── */
div[data-testid="stHorizontalBlock"] .stButton > button {{
    background: {CARD_BG} !important;
    border: 1px solid {LINE} !important;
    border-radius: 12px !important;
    font-size: 0.84rem !important;
    color: {INK} !important;
    text-align: left !important;
    padding: 0.7rem 1rem !important;
    height: auto !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    box-shadow: {SHADOW_SM} !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {{
    border-color: {ACCENT_VIVID} !important;
    background: {ACCENT_SOFT} !important;
    color: {ACCENT} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px {ACCENT_GLOW} !important;
}}

/* ── KPI card accent colours by index ── */
.kpi:nth-child(1)::before {{ background: linear-gradient(90deg, #1a5c35, #22a05a); }}
.kpi:nth-child(2)::before {{ background: linear-gradient(90deg, #b8873a, #e0b870); }}
.kpi:nth-child(3)::before {{ background: linear-gradient(90deg, #2563eb, #60a5fa); }}
.kpi:nth-child(4)::before {{ background: linear-gradient(90deg, #7c3aed, #a78bfa); }}
.kpi:nth-child(5)::before {{ background: linear-gradient(90deg, #dc2626, #f87171); }}
.kpi:nth-child(6)::before {{ background: linear-gradient(90deg, #0891b2, #67e8f9); }}

/* ── Topbar pill polish ── */
.topbar-meta {{
    animation: fadeIn 0.6s ease 0.2s both;
}}

/* ── Page background texture ── */
.main .block-container {{
    background: {PAGE_BG};
}}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {{
    border-radius: 14px !important;
    border: 1.5px solid {LINE} !important;
    font-size: 0.92rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease !important;
    background: {CARD_BG} !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {ACCENT_VIVID} !important;
    box-shadow: 0 0 0 3px {ACCENT_GLOW} !important;
}}

/* ── Expander (tool trace) ── */
details summary {{
    border-radius: 10px !important;
    padding: 0.45rem 0.8rem !important;
    font-size: 0.8rem !important;
    background: #f4f7f4 !important;
    border: 1px solid {LINE} !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: background 0.15s !important;
}}
details summary:hover {{ background: {ACCENT_SOFT} !important; }}
details[open] summary {{ border-radius: 10px 10px 0 0 !important; }}

/* ── Dataframe table ── */
[data-testid="stDataFrame"] {{
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid {LINE} !important;
    box-shadow: {SHADOW_SM} !important;
}}

/* ── Success/info alerts ── */
[data-testid="stAlert"] {{
    border-radius: 12px !important;
    border: none !important;
}}

/* ── Follow-up buttons ── */
.follow-label + div .stButton > button {{
    background: {CARD_BG} !important;
    border: 1px solid {LINE} !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    padding: 0.4rem 0.85rem !important;
    color: {MUTED} !important;
    font-weight: 500 !important;
}}
.follow-label + div .stButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    background: {ACCENT_SOFT} !important;
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
PALETTE_BAR = ["#22a05a", "#1a5c35", "#4dc886", "#0d3d22", "#6fd99a", "#b8e8c8"]
PALETTE_PIE = ["#1a5c35", "#22a05a", "#4dc886", "#6fd99a", "#b8e8c8", "#b8873a", "#e0b870"]


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
                  color_discrete_sequence=[ACCENT_VIVID])
    fig.update_traces(
        line_width=2.5, marker_size=6,
        marker_color=CARD_BG, marker_line_color=ACCENT_VIVID,
        marker_line_width=2,
        fill="tozeroy",
        fillcolor="rgba(34,160,90,0.08)",
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
                <div class="sidebar-brand-tag">Wise advice from any dataset</div>
              </div>
            </div>
            <div style="height:1px;background:rgba(255,255,255,0.07);margin:0.9rem 0 1rem"></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-h">Upload Data</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded is not None and uploaded.name != st.session_state.get("df_name"):
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = smart_read_excel(uploaded)

                changes = load_dataframe(df_up, dataset_name=uploaded.name)
                st.session_state.data_loaded = True
                st.session_state.df_name = uploaded.name
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.data_changes = changes
                st.session_state.agent = build_agent()
                st.success(f"✓  Loaded {len(df_up):,} rows")
                st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")

        if not st.session_state.data_loaded and os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sales_data.csv")
        ):
            if st.button("⚡  Try sample dataset", key="load_sample"):
                df_s = pd.read_csv(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sales_data.csv")
                )
                changes = load_dataframe(df_s, dataset_name="sample_sales_data.csv")
                st.session_state.data_loaded = True
                st.session_state.df_name = "sample_sales_data.csv"
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.data_changes = changes
                st.session_state.agent = build_agent()
                st.rerun()

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
              <div class="brand-tag">Plain-English insights from any dataset</div>
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
        box-shadow: 0 12px 32px rgba(34,160,90,0.35);
    }}
    .landing-title {{
        font-family: 'Fraunces', serif;
        font-size: 2.6rem; font-weight: 700;
        color: {INK}; letter-spacing: -0.03em;
        line-height: 1.1; margin-bottom: 0.75rem;
    }}
    .landing-sub {{
        font-size: 1rem; color: {MUTED};
        line-height: 1.65; max-width: 480px;
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
      <div class="landing-title">Wise advice from<br>any dataset</div>
      <div class="landing-sub">Upload a spreadsheet and Sage instantly builds your dashboard, profiles your data, and answers questions like a senior analyst — with real numbers and clear recommendations.</div>
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

    KPI_ICONS = ["◈", "◉", "◇", "◆", "○", "●"]
    cols = st.columns(min(len(kpi_targets), 6))
    for i, t in enumerate(kpi_targets[:6]):
        with cols[i]:
            icon = KPI_ICONS[i % len(KPI_ICONS)]
            st.markdown(
                f"""<div class="kpi" style="animation-delay:{i*0.07}s">
                    <div class="kpi-label">{icon} &nbsp;{t['label']}</div>
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
          <span>Ask anything in plain English — Sage profiles your data, runs analysis tools, and explains results with the actual numbers it found.</span>
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

    # Voice input using native Streamlit audio input
    with st.expander("🎤 Speak your question", expanded=False):
        audio_value = st.audio_input("Click to record your question", key="voice_recorder")
        if audio_value is not None:
            audio_bytes = audio_value.read()
            audio_hash = hash(audio_bytes)
            if audio_hash != st.session_state.get("_last_audio_hash"):
                st.session_state["_last_audio_hash"] = audio_hash
                with st.spinner("Sage is listening..."):
                    transcript = transcribe_audio(audio_bytes)
                st.write(f"DEBUG transcript: {transcript}")
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
