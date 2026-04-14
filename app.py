"""
app.py - Sage Streamlit UI (Phase 2 — Dashboard + Chat)
Two tabs: full visual dashboard + AI chat interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import uuid
from dotenv import load_dotenv
from tools import load_dataframe, get_df, smart_read_excel
from agent import build_agent

# Load .env from the same folder as this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

st.set_page_config(
    page_title="Sage — Wise advice from your data",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f5f0;
    color: #1a1a1a;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }

.sage-header {
    background: #1a2e1a;
    border-radius: 12px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.2rem;
}
.sage-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #a8d5a2;
    margin: 0;
}
.sage-subtitle {
    font-size: 0.78rem;
    color: #6a9e64;
    margin-top: 0.2rem;
    letter-spacing: 0.5px;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e0e8e0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1a2e1a;
    line-height: 1.2;
}
.kpi-label {
    font-size: 0.68rem;
    color: #6a9e64;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}
.kpi-delta {
    font-size: 0.75rem;
    margin-top: 0.2rem;
}

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #1a2e1a;
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #c8e0c4;
}

.chat-user {
    background: #e8f0e8;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.9rem;
    color: #1a2e1a;
    border: 1px solid #c8d8c8;
}
.chat-agent {
    background: #ffffff;
    border-radius: 4px 12px 12px 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.9rem;
    color: #1a1a1a;
    border: 1px solid #e0e8e0;
    border-left: 3px solid #4a8a44;
    line-height: 1.7;
}
.chat-label {
    font-size: 0.62rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    font-weight: 500;
}
.user-label { color: #4a8a44; }
.agent-label { color: #2d5e2d; }

.stButton > button {
    background: #1a2e1a !important;
    color: #a8d5a2 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
}
.stButton > button:hover {
    background: #2d5e2d !important;
}

section[data-testid="stSidebar"] {
    background: #f0ede6 !important;
    border-right: 1px solid #ddd8cc !important;
}

.insight-box {
    background: #f0f7ee;
    border: 1px solid #c8dcc4;
    border-left: 4px solid #4a8a44;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #1a2e1a;
}

.upload-area {
    background: #ffffff;
    border: 2px dashed #c8d8c8;
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #e8f0e8;
    border-radius: 8px 8px 0 0;
    color: #2d5e2d;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1a2e1a !important;
    color: #a8d5a2 !important;
}
hr { border-color: #e0e8e0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

if "messages"     not in st.session_state: st.session_state.messages     = []
if "thread_id"    not in st.session_state: st.session_state.thread_id    = str(uuid.uuid4())
if "agent"        not in st.session_state: st.session_state.agent        = build_agent()
if "data_loaded"  not in st.session_state: st.session_state.data_loaded  = False
if "quick_action" not in st.session_state: st.session_state.quick_action = None
if "df_name"      not in st.session_state: st.session_state.df_name      = None
if "data_changes" not in st.session_state: st.session_state.data_changes = []


# ── Helpers ────────────────────────────────────────────────────────────────────
SAGE_GREEN  = ["#c8e8c2", "#4a8a44", "#1a2e1a"]
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#1a1a1a"),
    margin=dict(l=10, r=10, t=30, b=10),
)

def sage_bar(df_agg, x, y, title="", color_col=None):
    fig = px.bar(df_agg, x=x, y=y,
                 color=color_col or y,
                 color_continuous_scale=SAGE_GREEN,
                 title=title, height=260)
    fig.update_layout(**CHART_THEME, coloraxis_showscale=False,
                      title_font_size=13)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8f0e8")
    return fig

def sage_line(df_agg, x, y, title=""):
    fig = px.line(df_agg, x=x, y=y, title=title, height=260,
                  markers=True, color_discrete_sequence=["#4a8a44"])
    fig.update_layout(**CHART_THEME, title_font_size=13)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8f0e8")
    fig.update_traces(line_width=2.5, marker_size=6)
    return fig

def sage_pie(df_agg, names, values, title=""):
    fig = px.pie(df_agg, names=names, values=values, title=title,
                 color_discrete_sequence=px.colors.sequential.Greens_r,
                 height=260)
    fig.update_layout(**CHART_THEME, title_font_size=13,
                      showlegend=True,
                      legend=dict(orientation="v", x=1, y=0.5))
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig

def fmt_currency(v):
    try:
        v = float(v)
        if v >= 1_000_000: return f"${v/1_000_000:.1f}M"
        if v >= 1_000:     return f"${v/1_000:.0f}K"
        return f"${v:.0f}"
    except:
        return "$0"

def safe_groupby(df, group_col, val_col, agg="sum"):
    """Safe groupby that converts to numeric first"""
    d = df.copy()
    d[val_col] = pd.to_numeric(d[val_col], errors='coerce').fillna(0)
    if agg == "sum":
        return d.groupby(group_col)[val_col].sum().reset_index()
    return d.groupby(group_col)[val_col].mean().reset_index()

def run_agent(question):
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        result = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )
        # Get last message
        messages = result["messages"]
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    return msg.content
        return messages[-1].content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def auto_insights(df):
    """Generate 3-5 plain English insights from the data automatically."""
    insights = []
    try:
        if "region" in df.columns and "revenue" in df.columns:
            reg = pd.to_numeric(df["revenue"], errors="coerce").groupby(df["region"]).sum().sort_values()
            if len(reg) >= 2 and reg.iloc[-1] > 0:
                top, bot = reg.index[-1], reg.index[0]
                gap = round((1 - reg[bot] / reg[top]) * 100)
                insights.append(f"📍 {top} is your strongest region. {bot} is {gap}% behind — worth investigating.")
    except: pass
    try:
        if "category" in df.columns and "revenue" in df.columns:
            cat = pd.to_numeric(df["revenue"], errors="coerce").groupby(df["category"]).sum().sort_values(ascending=False)
            if len(cat) >= 2:
                insights.append(f"🏆 {cat.index[0]} drives the most revenue ({fmt_currency(cat.iloc[0])}). {cat.index[-1]} is your smallest category.")
    except: pass
    try:
        if "channel" in df.columns and "revenue" in df.columns:
            ch = pd.to_numeric(df["revenue"], errors="coerce").groupby(df["channel"]).sum().sort_values(ascending=False)
            if len(ch) >= 1:
                insights.append(f"🛒 {ch.index[0]} channel leads with {fmt_currency(ch.iloc[0])} in revenue.")
    except: pass
    try:
        if "margin_pct" in df.columns:
            avg_margin = pd.to_numeric(df["margin_pct"], errors='coerce').mean()
            if "product" in df.columns:
                low = pd.to_numeric(df["margin_pct"], errors="coerce").groupby(df["product"]).mean().sort_values().head(1)
                if len(low) > 0:
                    insights.append(f"📊 Average margin is {avg_margin:.1f}%. '{low.index[0]}' has the lowest margin at {low.iloc[0]:.1f}%.")
    except: pass
    try:
        if "discount_pct" in df.columns and "revenue" in df.columns:
            disc = pd.to_numeric(df["discount_pct"], errors="coerce")
            heavy_disc = disc[disc >= 15]
            if len(heavy_disc) > 0:
                pct = round(len(heavy_disc) / len(df) * 100)
                insights.append(f"💡 {pct}% of orders use discounts of 15%+. Consider if this is intentional strategy.")
    except: pass
    return insights


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Sage")
    st.markdown("*Wise advice from your data*")
    st.markdown("---")

    st.markdown("**Upload your data**")
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") \
                    else smart_read_excel(uploaded)
            changes = load_dataframe(df_up)
            st.session_state.data_loaded  = True
            st.session_state.df_name      = uploaded.name
            st.session_state.messages     = []
            st.session_state.thread_id    = str(uuid.uuid4())
            st.session_state.data_changes = changes
            # Rebuild agent so it reads the new data
            st.session_state.agent = build_agent()
            st.success(f"✅ Loaded {len(df_up):,} rows — data auto-cleaned and ready")
            if changes:
                with st.expander(f"🤖 AI cleaned {len(changes)} things automatically — click to see"):
                    for c in changes:
                        st.markdown(f"✅ {c}")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if not st.session_state.data_loaded:
        if st.button("Load sample sales data"):
            try:
                df_s = pd.read_csv("sample_sales_data.csv")
                changes = load_dataframe(df_s)
                st.session_state.data_loaded  = True
                st.session_state.df_name      = "sample_sales_data.csv"
                st.session_state.messages     = []
                st.session_state.thread_id    = str(uuid.uuid4())
                st.session_state.data_changes = changes
                st.session_state.agent        = build_agent()
                st.rerun()
            except:
                st.error("Run generate_data.py first")

    st.markdown("---")

    if st.session_state.data_loaded:
        # Filters
        df_now = get_df()
        st.markdown("**Filters**")

        regions = ["All"] + sorted(df_now["region"].dropna().astype(str).unique().tolist()) \
                  if "region" in df_now.columns else ["All"]
        sel_region = st.selectbox("Region", regions)

        categories = ["All"] + sorted(df_now["category"].dropna().astype(str).unique().tolist()) \
                     if "category" in df_now.columns else ["All"]
        sel_cat = st.selectbox("Category", categories)

        channels = ["All"] + sorted(df_now["channel"].dropna().astype(str).unique().tolist()) \
                   if "channel" in df_now.columns else ["All"]
        sel_channel = st.selectbox("Channel", channels)

        st.session_state["filter_region"]   = sel_region
        st.session_state["filter_category"] = sel_cat
        st.session_state["filter_channel"]  = sel_channel

        st.markdown("---")
        st.markdown("**Ask Sage**")
        quick_qs = [
            "Give me a full business briefing",
            "What is working well?",
            "What needs attention?",
            "Which region underperforms?",
            "What are my top products?",
            "Where should I focus next quarter?",
        ]
        for q in quick_qs:
            if st.button(q, key=f"qa_{q}"):
                st.session_state.quick_action = q

        st.markdown("---")
        if st.button("New conversation"):
            st.session_state.messages    = []
            st.session_state.thread_id   = str(uuid.uuid4())
            st.rerun()

        st.markdown(f"""
        <div style="font-size:0.7rem;color:#6a9e64;margin-top:0.5rem">
        {st.session_state.df_name}<br>
        {len(df_now):,} rows · {len(df_now.columns)} columns
        </div>""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sage-header">
    <div class="sage-title">🌿 Sage</div>
    <div class="sage-subtitle">Wise advice from your data — helping you make better decisions</div>
</div>
""", unsafe_allow_html=True)

# ── No data state ──────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:
    st.markdown("""
    <div class="upload-area">
        <div style="font-size:2.5rem;margin-bottom:0.8rem">🌿</div>
        <div style="font-size:1rem;font-weight:500;color:#1a2e1a;margin-bottom:0.4rem">
            Upload your data to get started
        </div>
        <div style="font-size:0.85rem;color:#6a9e64">
            Upload a CSV or Excel file — or load the sample sales dataset from the sidebar
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ── Apply filters ──────────────────────────────────────────────────────────────
df = get_df().copy()
if st.session_state.get("filter_region", "All") != "All":
    df = df[df["region"] == st.session_state["filter_region"]]
if st.session_state.get("filter_category", "All") != "All":
    df = df[df["category"] == st.session_state["filter_category"]]
if st.session_state.get("filter_channel", "All") != "All":
    df = df[df["channel"] == st.session_state["filter_channel"]]


# ── TWO TABS ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Dashboard", "💬 Chat with Sage"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPI cards ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def safe_sum(df, col):
        if col not in df.columns: return 0
        return pd.to_numeric(df[col], errors='coerce').sum() or 0

    def safe_mean(df, col):
        if col not in df.columns: return 0
        return pd.to_numeric(df[col], errors='coerce').mean() or 0

    total_rev   = safe_sum(df, "revenue")
    total_prof  = safe_sum(df, "profit")
    avg_margin  = (total_prof / total_rev * 100) if total_rev > 0 else 0
    total_units = safe_sum(df, "units_sold")
    total_orders = len(df)
    avg_disc    = safe_mean(df, "discount_pct")

    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{fmt_currency(total_rev)}</div>
            <div class="kpi-label">Total Revenue</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{fmt_currency(total_prof)}</div>
            <div class="kpi-label">Total Profit</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{avg_margin:.1f}%</div>
            <div class="kpi-label">Avg Margin</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{total_units:,}</div>
            <div class="kpi-label">Units Sold</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{total_orders:,}</div>
            <div class="kpi-label">Total Orders</div>
        </div>""", unsafe_allow_html=True)
    with k6:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{avg_disc:.1f}%</div>
            <div class="kpi-label">Avg Discount</div>
        </div>""", unsafe_allow_html=True)

    # ── Auto Insights ──────────────────────────────────────────────────────────
    insights = auto_insights(df)
    if insights:
        st.markdown('<div class="section-header">🌿 Key Insights</div>',
                    unsafe_allow_html=True)
        for ins in insights:
            st.markdown(f'<div class="insight-box">{ins}</div>',
                        unsafe_allow_html=True)

    # ── Row 1: Revenue by Region + Monthly Trend ───────────────────────────────
    st.markdown('<div class="section-header">Revenue Performance</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        if "region" in df.columns and "revenue" in df.columns:
            reg_df = safe_groupby(df, "region", "revenue").sort_values("revenue", ascending=False)
            st.plotly_chart(sage_bar(reg_df, "region", "revenue", "Revenue by Region"),
                            use_container_width=True)

    with c2:
        if "month" in df.columns and "revenue" in df.columns:
            month_order = ["January","February","March","April","May","June",
                           "July","August","September","October","November","December"]
            mon_df = safe_groupby(df, "month", "revenue")
            mon_df["month"] = pd.Categorical(mon_df["month"], categories=month_order, ordered=True)
            mon_df = mon_df.sort_values("month")
            st.plotly_chart(sage_line(mon_df, "month", "revenue", "Monthly Revenue Trend"),
                            use_container_width=True)

    # ── Row 2: Category + Channel ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Category & Channel Breakdown</div>',
                unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        # Use category if available, fall back to product
        cat_col = "category" if "category" in df.columns else \
                  "product" if "product" in df.columns else None
        if cat_col and "revenue" in df.columns:
            cat_df = safe_groupby(df, cat_col, "revenue")
            # If too many values (product), show top 8
            if len(cat_df) > 8:
                cat_df = cat_df.sort_values("revenue", ascending=False).head(8)
            label = "Revenue by Category" if cat_col == "category" else "Revenue by Product"
            st.plotly_chart(sage_pie(cat_df, cat_col, "revenue", label),
                            use_container_width=True)

    with c4:
        if "channel" in df.columns and "revenue" in df.columns:
            ch_df = safe_groupby(df, "channel", "revenue").sort_values("revenue", ascending=False)
            st.plotly_chart(sage_bar(ch_df, "channel", "revenue", "Revenue by Channel"),
                            use_container_width=True)

    # ── Row 3: Top Products + Margin by Category ───────────────────────────────
    st.markdown('<div class="section-header">Product Performance</div>',
                unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        if "product" in df.columns and "revenue" in df.columns:
            prod_df = safe_groupby(df, "product", "revenue").sort_values("revenue", ascending=True).tail(8)
            fig = px.bar(prod_df, x="revenue", y="product", orientation="h",
                         title="Top Products by Revenue", height=300,
                         color="revenue",
                         color_continuous_scale=SAGE_GREEN)
            fig.update_layout(**CHART_THEME, coloraxis_showscale=False,
                              title_font_size=13)
            fig.update_xaxes(showgrid=True, gridcolor="#e8f0e8")
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    with c6:
        cat_col2 = "category" if "category" in df.columns else \
                   "product" if "product" in df.columns else None
        if cat_col2 and "margin_pct" in df.columns:
            mar_df = safe_groupby(df, cat_col2, "margin_pct", "mean").sort_values("margin_pct", ascending=False)
            mar_df["margin_pct"] = pd.to_numeric(mar_df["margin_pct"], errors="coerce").round(1)
            label2 = "Avg Margin % by Category" if cat_col2 == "category" else "Avg Margin % by Product"
            fig = px.bar(mar_df, x=cat_col2, y="margin_pct",
                         title=label2, height=300,
                         color="margin_pct",
                         color_continuous_scale=SAGE_GREEN)
            fig.update_layout(**CHART_THEME, coloraxis_showscale=False,
                              title_font_size=13)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#e8f0e8")
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Quarterly + Units by Region ────────────────────────────────────
    st.markdown('<div class="section-header">Volume & Quarterly View</div>',
                unsafe_allow_html=True)
    c7, c8 = st.columns(2)

    with c7:
        if "quarter" in df.columns and "revenue" in df.columns:
            df_q = df.copy()
            df_q["revenue"] = pd.to_numeric(df_q["revenue"], errors="coerce").fillna(0)
            df_q["profit"] = pd.to_numeric(df_q["profit"], errors="coerce").fillna(0)
            q_df = df_q.groupby("quarter")[["revenue","profit"]].sum().reset_index()
            fig = go.Figure()
            fig.add_bar(x=q_df["quarter"], y=q_df["revenue"],
                        name="Revenue", marker_color="#4a8a44")
            fig.add_bar(x=q_df["quarter"], y=q_df["profit"],
                        name="Profit", marker_color="#a8d5a2")
            fig.update_layout(**CHART_THEME, title="Revenue vs Profit by Quarter",
                              title_font_size=13, barmode="group", height=280,
                              legend=dict(orientation="h", y=1.1))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#e8f0e8")
            st.plotly_chart(fig, use_container_width=True)

    with c8:
        if "region" in df.columns and "units_sold" in df.columns:
            u_df = safe_groupby(df, "region", "units_sold").sort_values("units_sold", ascending=False)
            st.plotly_chart(sage_bar(u_df, "region", "units_sold", "Units Sold by Region"),
                            use_container_width=True)

    # ── Row 5: Discount analysis ───────────────────────────────────────────────
    if "discount_pct" in df.columns and "revenue" in df.columns:
        st.markdown('<div class="section-header">Discount Analysis</div>',
                    unsafe_allow_html=True)
        c9, c10 = st.columns(2)

        with c9:
            disc_col = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0)
            disc_bins = pd.cut(disc_col,
                               bins=[-1, 0, 5, 10, 15, 20, 100],
                               labels=["No discount","1-5%","6-10%","11-15%","16-20%","20%+"])
            df["_rev_num"] = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
            disc_df = df.groupby(disc_bins, observed=True)["_rev_num"].sum().reset_index()
            disc_df.columns = ["discount_band", "revenue"]
            st.plotly_chart(sage_bar(disc_df, "discount_band", "revenue",
                                     "Revenue by Discount Band"),
                            use_container_width=True)

        with c10:
            if "region" in df.columns:
                disc_reg = safe_groupby(df, "region", "discount_pct", "mean")
                disc_reg["discount_pct"] = disc_reg["discount_pct"].round(1)
                st.plotly_chart(sage_bar(disc_reg, "region", "discount_pct",
                                         "Avg Discount % by Region"),
                                use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="font-size:0.88rem;color:#4a8a44;margin-bottom:1rem;padding:0.8rem 1rem;
    background:#f0f7ee;border-radius:8px;border:1px solid #c8dcc4">
    💬 Ask Sage anything about your data in plain English.
    Type your question below and press Enter.
    </div>""", unsafe_allow_html=True)

    def _content_to_str(content) -> str:
        """Normalise LangChain content — handles both str and list-of-blocks."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts)
        return str(content)

    # Reserve the messages area ABOVE the input
    messages_area = st.container()

    # Process quick action from sidebar
    if st.session_state.get("quick_action"):
        pending = st.session_state.quick_action
        st.session_state.quick_action = None
        st.session_state.messages.append({"role": "user", "content": pending})
        with st.spinner("Sage is thinking..."):
            answer = run_agent(pending)
        st.session_state.messages.append(
            {"role": "assistant", "content": _content_to_str(answer)}
        )

    # Chat input at the bottom — st.chat_input handles its own state
    user_input = st.chat_input("Ask Sage about your data...")

    # Process typed input — no st.rerun() needed
    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Sage is thinking..."):
            answer = run_agent(user_input.strip())
        st.session_state.messages.append(
            {"role": "assistant", "content": _content_to_str(answer)}
        )

    # Render all messages into the reserved container
    with messages_area:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:2rem 1rem;opacity:0.5">
                <div style="font-size:2rem;margin-bottom:0.8rem">🌿</div>
                <div style="font-size:0.88rem;color:#4a8a44">
                    Type a question below or click a quick question in the sidebar
                </div>
            </div>""", unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-user">
                    <div class="chat-label user-label">You</div>
                    {_content_to_str(msg["content"])}
                </div>""", unsafe_allow_html=True)
            else:
                content = _content_to_str(msg["content"]).replace("\n", "<br>")
                st.markdown(f"""
                <div class="chat-agent">
                    <div class="chat-label agent-label">Sage</div>
                    {content}
                </div>""", unsafe_allow_html=True)
