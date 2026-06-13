"""
Predictive Maintenance Dashboard — HMI-inspired Streamlit interface.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.api_client import (
    call_predict,
    check_api_health,
    get_feature_importance,
    get_metrics,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictMaint | Compressor AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #050f1a !important;
    font-family: 'Inter', system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background: #081524 !important;
    border-right: 1px solid #112030;
}
[data-testid="stHeader"]     { display: none; }
.block-container              { padding: 20px 28px 40px !important; }
section[data-testid="stSidebar"] > div { padding-top: 20px !important; }

/* ── Typography ── */
h1,h2,h3,h4 { color: #d0e8f5 !important; font-weight: 700 !important; }
p, li       { color: #5a7a8a; }

/* ── Top topbar ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #081524;
    border: 1px solid #112030;
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 20px;
}
.topbar-logo   { font-size: 20px; font-weight: 800; color: #00c8c0; letter-spacing: -0.5px; }
.topbar-sub    { font-size: 11px; color: #3a5a6a; text-transform: uppercase; letter-spacing: 1.5px; }
.topbar-spacer { flex: 1; }
.topbar-chip {
    background: #0c1e30;
    border: 1px solid #1a3244;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #7a9ab0;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* ── HMI Panel card ── */
.panel {
    background: #0c1e30;
    border: 1px solid #1a3244;
    border-radius: 12px;
    padding: 16px 20px 20px;
    position: relative;
    height: 100%;
}
.panel-header {
    font-size: 10px;
    font-weight: 700;
    color: #3a5a6a;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #112030;
    padding-bottom: 10px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-header-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00c8c0;
    box-shadow: 0 0 6px #00c8c0;
    display: inline-block;
}

/* ── KPI tiles (like HMI stat boxes) ── */
.kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px; }
.kpi-tile {
    background: #081524;
    border: 1px solid #112030;
    border-radius: 8px;
    padding: 12px 14px;
}
.kpi-tile.accent-teal  { border-top: 2px solid #00c8c0; }
.kpi-tile.accent-blue  { border-top: 2px solid #0094c6; }
.kpi-tile.accent-red   { border-top: 2px solid #ff3d5a; }
.kpi-tile.accent-green { border-top: 2px solid #00e096; }
.kpi-tile.accent-amber { border-top: 2px solid #ffb300; }
.kpi-big   { font-size: 28px; font-weight: 800; color: #d0e8f5; line-height: 1.1; margin: 2px 0; }
.kpi-unit  { font-size: 11px; color: #3a5a6a; font-weight: 600; }
.kpi-label { font-size: 9px; color: #3a5a6a; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

/* ── Wide metric bar (4-col) ── */
.metric-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 14px 0; }
.metric-cell {
    background: #0c1e30;
    border: 1px solid #1a3244;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
}
.metric-cell:hover { border-color: #264a60; }
.metric-val   { font-size: 22px; font-weight: 700; color: #d0e8f5; line-height: 1.2; }
.metric-label { font-size: 10px; color: #3a5a6a; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }

/* ── Alert banners ── */
.banner {
    border-radius: 10px;
    padding: 14px 20px;
    font-weight: 600;
    font-size: 15px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.banner-fault  { background:rgba(255,61,90,.08);  border:1px solid #ff3d5a; border-left:4px solid #ff3d5a; color:#ff3d5a; }
.banner-normal { background:rgba(0,224,150,.07);  border:1px solid #00e096; border-left:4px solid #00e096; color:#00e096; }

/* ── Sensor section labels ── */
.sensor-section {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3a5a6a;
    margin: 16px 0 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #112030;
}

/* ── Digital signal toggle chips ── */
.dig-label {
    text-align: center;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3a5a6a;
    margin-bottom: 2px;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #006e6a 0%, #00c8c0 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    height: 52px !important;
    letter-spacing: 0.6px;
    transition: opacity .2s, box-shadow .2s;
    box-shadow: 0 4px 18px rgba(0,200,192,0.25);
}
.stButton > button[kind="primary"]:hover {
    opacity: .9 !important;
    box-shadow: 0 6px 24px rgba(0,200,192,0.4) !important;
}

/* ── Sidebar ── */
.sidebar-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #00c8c0;
    margin: 22px 0 10px;
    padding-bottom: 7px;
    border-bottom: 1px solid #112030;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #0c1e30;
    font-size: 12px;
}
.stat-key { color: #3a5a6a; }
.stat-val { color: #d0e8f5; font-weight: 600; }

/* ── Status badge ── */
.sbadge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.sbadge-online  { background:rgba(0,200,192,.12); color:#00c8c0; border:1px solid #00c8c0; }
.sbadge-offline { background:rgba(255,61,90,.12);  color:#ff3d5a; border:1px solid #ff3d5a; }

/* ── Tabs ── */
[data-testid="stTabs"] button            { color:#3a5a6a !important; font-weight:600 !important; font-size:13px !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#00c8c0 !important;
    border-bottom-color:#00c8c0 !important;
}

/* ── Table ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #112030 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #081524; }
::-webkit-scrollbar-thumb { background: #1a3244; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Plot theme helpers ────────────────────────────────────────────────────────
_GRID = dict(gridcolor="#112030", zerolinecolor="#112030", linecolor="#112030")
_BG   = dict(plot_bgcolor="#0c1e30", paper_bgcolor="#050f1a")
_FONT = dict(color="#5a7a8a", family="Inter, system-ui, sans-serif")


def _layout(height: int = 300, **kw) -> dict:
    """Merge dark-theme base with caller overrides — no duplicate-key risk."""
    base: dict = {
        **_BG,
        "font":   _FONT,
        "xaxis":  kw.pop("xaxis",  dict(**_GRID)),
        "yaxis":  kw.pop("yaxis",  dict(**_GRID)),
        "margin": kw.pop("margin", dict(t=50, b=40, l=50, r=20)),
        "height": height,
    }
    base.update(kw)
    return base


# ─── Colour palette ───────────────────────────────────────────────────────────
C_TEAL   = "#00c8c0"
C_BLUE   = "#0094c6"
C_GREEN  = "#00e096"
C_RED    = "#ff3d5a"
C_AMBER  = "#ffb300"
C_PURPLE = "#9b72cf"

# ─── Sensor config ────────────────────────────────────────────────────────────
ANALOGUE = {
    "TP2":            (-0.5,  1.0,  -0.012, "bar"),
    "TP3":            (0.0,  12.0,   9.358, "bar"),
    "H1":             (0.0,  12.0,   9.340, "bar"),
    "DV_pressure":    (-0.5,  0.5,  -0.024, "bar"),
    "Reservoirs":     (0.0,  12.0,   9.358, "bar"),
    "Oil_temperature":(30.0,100.0,   53.6,  "°C"),
    "Motor_current":  (0.0,  10.0,   0.04,  "A"),
}
DIGITAL_DEFAULTS = {
    "COMP": True, "DV_eletric": False, "Towers": True, "MPG": True,
    "LPS": False, "Pressure_switch": True, "Oil_level": True, "Caudal_impulses": True,
}

# ─── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─── Cached API calls ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def _fetch_metrics(api_url: str) -> dict | None:
    return get_metrics(api_url)


@st.cache_data(ttl=60)
def _fetch_fi(api_url: str) -> dict | None:
    return get_feature_importance(api_url)


# ─── Charts ───────────────────────────────────────────────────────────────────
def _donut_gauge(prob_fault: float, alert: bool) -> go.Figure:
    """Ring-style fault probability indicator (HMI-inspired donut)."""
    pct     = round(prob_fault * 100, 1)
    fg_col  = C_RED if alert else C_TEAL
    bg_col  = "rgba(255,255,255,0.04)"

    fig = go.Figure(go.Pie(
        values=[pct, max(0.01, 100 - pct)],
        hole=0.72,
        marker=dict(
            colors=[fg_col, bg_col],
            line=dict(width=0),
        ),
        textinfo="none",
        hoverinfo="skip",
        sort=False,
        direction="clockwise",
        rotation=90,
    ))
    fig.add_annotation(
        text=(
            f"<span style='font-size:32px;font-weight:800;color:{fg_col};'>{pct:.0f}%</span>"
            f"<br><span style='font-size:11px;color:#3a5a6a;'>FAULT PROB</span>"
        ),
        x=0.5, y=0.5,
        showarrow=False,
        align="center",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=220,
        paper_bgcolor="#0c1e30",
        plot_bgcolor="#0c1e30",
    )
    return fig


def _radar(payload: dict) -> go.Figure:
    """Normalised pressure profile radar."""
    sensors  = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"]
    lo_vals  = [-0.5,  0.0, 0.0, -0.5,  0.0]
    hi_vals  = [ 1.0, 12.0,12.0,  0.5, 12.0]
    norm = [
        max(0.0, min(1.0, (payload[s] - lo) / (hi - lo + 1e-9)))
        for s, lo, hi in zip(sensors, lo_vals, hi_vals)
    ]
    r_closed     = norm + [norm[0]]
    theta_closed = sensors + [sensors[0]]

    fig = go.Figure(go.Scatterpolar(
        r=r_closed, theta=theta_closed, fill="toself",
        fillcolor="rgba(0,200,192,0.10)",
        line=dict(color=C_TEAL, width=2),
        marker=dict(size=5, color=C_TEAL),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="#1a3244", tickfont=dict(color="#3a5a6a", size=8),
                tickvals=[0.25, 0.5, 0.75, 1.0],
                tickformat=".0%",
            ),
            angularaxis=dict(gridcolor="#1a3244", tickfont=dict(color="#7a9ab0", size=10)),
            bgcolor="#0c1e30",
        ),
        showlegend=False,
        title=dict(text="Pressure Profile", font=dict(size=11, color="#3a5a6a")),
        height=250,
        paper_bgcolor="#0c1e30",
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


def _health_bars(payload: dict) -> go.Figure:
    """Derived health indicators as horizontal bars."""
    labels = ["ΔP (TP3−TP2)", "Oil Temp °C", "Motor A", "Temp×Current", "Res−TP3"]
    values = [
        round(payload["TP3"] - payload["TP2"], 3),
        payload["Oil_temperature"],
        payload["Motor_current"],
        round(payload["Oil_temperature"] * payload["Motor_current"], 3),
        round(payload["Reservoirs"] - payload["TP3"], 3),
    ]
    max_abs = max(abs(v) for v in values) or 1.0
    fill_pcts = [min(abs(v) / max_abs, 1.0) for v in values]
    bar_colors = [
        C_TEAL if v >= 0 else C_RED
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.75)" for c in bar_colors],
            line=dict(width=0),
        ),
        text=[f"<b>{v:.3f}</b>" for v in values],
        textposition="outside",
        textfont=dict(color="#7a9ab0", size=10),
    ))
    fig.update_layout(**_layout(
        height=250,
        title=dict(text="Health Indicators", font=dict(size=11, color="#3a5a6a")),
        xaxis=dict(**_GRID),
        yaxis=dict(**_GRID),
        margin=dict(t=40, b=10, l=10, r=70),
    ))
    return fig


def _model_comparison(metrics: dict) -> go.Figure:
    keys  = [k for k in metrics if isinstance(metrics[k], dict) and "validation" in metrics[k]]
    names = [k.replace("_", " ").title() for k in keys]
    series = [
        ("F1",        C_TEAL,   [metrics[k]["validation"]["f1"]        for k in keys]),
        ("Recall",    C_GREEN,  [metrics[k]["validation"]["recall"]     for k in keys]),
        ("Precision", C_AMBER,  [metrics[k]["validation"]["precision"]  for k in keys]),
        ("ROC AUC",   C_PURPLE, [metrics[k]["validation"]["roc_auc"]   for k in keys]),
    ]
    fig = go.Figure()
    for label, color, vals in series:
        fig.add_trace(go.Bar(
            name=label, x=names, y=vals,
            marker=dict(color=color, opacity=0.82, line=dict(width=0)),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=10, color="#5a7a8a"),
        ))
    fig.update_layout(**_layout(
        height=380,
        barmode="group",
        title=dict(text="Validation Metrics — All Models", font=dict(size=13, color="#d0e8f5")),
        legend=dict(bgcolor="#0c1e30", bordercolor="#1a3244", font=dict(color="#7a9ab0")),
        yaxis=dict(range=[0, 1.18], **_GRID),
        margin=dict(t=55, b=40, l=50, r=20),
    ))
    return fig


def _confusion_heatmap(metrics: dict, model_key: str) -> go.Figure:
    cm    = metrics[model_key]["validation"]["confusion_matrix"]
    z     = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
    total = sum(sum(row) for row in z)
    annotations = [
        dict(
            text=(
                f"<b>{z[i][j]:,}</b><br>"
                f"<span style='font-size:9px'>{z[i][j]/total:.1%}</span>"
            ),
            x=j, y=i, showarrow=False,
            font=dict(size=13, color="white"),
        )
        for i in range(2) for j in range(2)
    ]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Pred Normal", "Pred Fault"],
        y=["Actual Normal", "Actual Fault"],
        colorscale=[[0.0,"#0c1e30"],[0.5,"#0d3d5c"],[1.0,C_TEAL]],
        showscale=False,
    ))
    fig.update_layout(
        annotations=annotations,
        title=dict(text=model_key.replace("_"," ").title(), font=dict(size=12, color="#d0e8f5")),
        **_layout(height=270, margin=dict(t=45, b=40, l=100, r=20)),
    )
    return fig


def _feature_importance_chart(fi_data: dict) -> go.Figure:
    pairs  = fi_data.get("importances", [])[:15]
    names  = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]
    max_v  = max(values) if values else 1.0
    colors = [f"rgba(0,200,192,{0.25 + 0.75*v/max_v:.2f})" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color="#5a7a8a", size=9),
    ))
    model_name = fi_data.get("model", "").replace("_", " ").title()
    fig.update_layout(**_layout(
        height=430,
        title=dict(text=f"Feature Importances — {model_name}", font=dict(size=13, color="#d0e8f5")),
        yaxis=dict(**_GRID),
        margin=dict(t=55, b=20, l=185, r=75),
    ))
    return fig


def _history_chart(history: list) -> go.Figure | None:
    if not history:
        return None
    df = pd.DataFrame(history)
    df["n"] = range(1, len(df) + 1)

    fig = go.Figure()
    # Filled area under the line
    fig.add_trace(go.Scatter(
        x=df["n"], y=df["probability_fault"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,200,192,0.06)",
        line=dict(color="rgba(0,200,192,0.35)", width=1.5),
        showlegend=False,
    ))
    # Coloured markers
    fig.add_trace(go.Scatter(
        x=df["n"], y=df["probability_fault"],
        mode="markers",
        marker=dict(
            color=[C_RED if a else C_TEAL for a in df["alert"]],
            size=9,
            line=dict(color="#050f1a", width=1.5),
        ),
        text=df["timestamp"],
        hovertemplate="<b>%{y:.1%}</b> at %{text}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(
        y=0.5, line=dict(color=C_AMBER, dash="dash", width=1.5),
        annotation_text=" Alert (50%)",
        annotation_font=dict(color=C_AMBER, size=10),
    )
    fig.update_layout(**_layout(
        height=255,
        title=dict(text="Session Prediction History", font=dict(size=12, color="#d0e8f5")),
        xaxis=dict(title="Prediction #", **_GRID),
        yaxis=dict(
            title="Fault Probability",
            range=[-0.05, 1.05],
            tickformat=".0%",
            **_GRID,
        ),
        margin=dict(t=50, b=40, l=60, r=20),
    ))
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='padding:8px 0 12px;'>"
            "<div style='font-size:20px;font-weight:800;color:#00c8c0;letter-spacing:-0.5px;'>⬡ PredictMaint</div>"
            "<div style='font-size:10px;color:#1a3a4a;text-transform:uppercase;"
            "letter-spacing:2px;margin-top:3px;'>MetroPT-3 · Compressor AI</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        health = check_api_health(API_URL)
        st.markdown("<div class='sidebar-label'>System Status</div>", unsafe_allow_html=True)

        if health and health.get("model_loaded"):
            st.markdown("<span class='sbadge sbadge-online'>⬤ API Online</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for key, val in [
                ("Model",   health.get("model_name","—").replace("_"," ").title()),
                ("Val F1",  f"{health.get('model_val_f1',0):.4f}"),
                ("Served",  f"{health.get('predictions_served',0):,}"),
            ]:
                st.markdown(
                    f"<div class='stat-row'><span class='stat-key'>{key}</span>"
                    f"<span class='stat-val'>{val}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<span class='sbadge sbadge-offline'>⬤ API Offline</span>", unsafe_allow_html=True)
            st.caption(f"`{API_URL}`")

        st.markdown("<div class='sidebar-label'>Configuration</div>", unsafe_allow_html=True)
        for key, val in [
            ("Alert threshold", "≥ 50%"),
            ("Horizon",         "6 hr pre-fault"),
            ("Dataset",         "MetroPT-3"),
            ("Class balance",   "class_weight=balanced"),
        ]:
            st.markdown(
                f"<div class='stat-row'><span class='stat-key'>{key}</span>"
                f"<span class='stat-val'>{val}</span></div>",
                unsafe_allow_html=True,
            )

        n      = len(st.session_state.history)
        faults = sum(1 for p in st.session_state.history if p["alert"])

        st.markdown("<div class='sidebar-label'>Session</div>", unsafe_allow_html=True)
        for key, val, color in [
            ("Predictions", str(n),      "#d0e8f5"),
            ("Alerts",      str(faults), C_RED if faults else "#d0e8f5"),
        ]:
            st.markdown(
                f"<div class='stat-row'><span class='stat-key'>{key}</span>"
                f"<span class='stat-val' style='color:{color};'>{val}</span></div>",
                unsafe_allow_html=True,
            )

        if n > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⟳  Clear Session", use_container_width=True):
                st.session_state.history = []
                st.rerun()


# ─── Sensor input panel ───────────────────────────────────────────────────────
def collect_inputs() -> dict:
    st.markdown("<div class='panel'><div class='panel-header'><span class='panel-header-dot'></span>Sensor Inputs</div>", unsafe_allow_html=True)

    st.markdown("<div class='sensor-section'>Pressure Sensors (bar)</div>", unsafe_allow_html=True)
    p_keys = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"]
    p_cols = st.columns(5)
    pv: dict[str, float] = {}
    for col, key in zip(p_cols, p_keys):
        lo, hi, default, unit = ANALOGUE[key]
        with col:
            pv[key] = st.slider(
                f"{key}", min_value=float(lo), max_value=float(hi),
                value=float(default),
                step=0.001 if abs(hi - lo) <= 2 else 0.01,
                format="%.3f",
            )

    st.markdown("<div class='sensor-section'>Temperature & Current</div>", unsafe_allow_html=True)
    tc = st.columns(2)
    with tc[0]:
        oil_temp = st.slider("Oil Temperature (°C)", 30.0, 100.0, 53.6, 0.1)
    with tc[1]:
        motor_curr = st.slider("Motor Current (A)", 0.0, 10.0, 0.04, 0.01)

    st.markdown("<div class='sensor-section'>Digital Valve Signals</div>", unsafe_allow_html=True)
    d_cols = st.columns(8)
    dv: dict[str, float] = {}
    for col, (sensor, default) in zip(d_cols, DIGITAL_DEFAULTS.items()):
        with col:
            st.markdown(f"<div class='dig-label'>{sensor}</div>", unsafe_allow_html=True)
            dv[sensor] = float(st.toggle(f"_{sensor}", value=default, label_visibility="collapsed"))

    st.markdown("</div>", unsafe_allow_html=True)

    return {**pv, "Oil_temperature": oil_temp, "Motor_current": motor_curr, **dv}


# ─── Result panel ─────────────────────────────────────────────────────────────
def render_result(result: dict, payload: dict, fi_data: dict | None) -> None:
    alert = result["alert"]

    # Alert banner
    cls  = "banner-fault" if alert else "banner-normal"
    icon = "⚠" if alert else "✔"
    st.markdown(
        f"<div class='banner {cls}'>{icon}&nbsp;&nbsp;{result['alert_message']}</div>",
        unsafe_allow_html=True,
    )

    # 4 metric cells
    pct_fault  = f"{result['probability_fault']:.1%}"
    pct_normal = f"{result['probability_normal']:.1%}"
    model_name = result["model_used"].replace("_", " ").title()
    status_col = C_RED if alert else C_GREEN

    st.markdown(
        f"<div class='metric-row'>"
        f"<div class='metric-cell'><div class='metric-val' style='color:{status_col};'>{result['status']}</div><div class='metric-label'>Status</div></div>"
        f"<div class='metric-cell'><div class='metric-val' style='color:{C_RED};'>{pct_fault}</div><div class='metric-label'>Fault Probability</div></div>"
        f"<div class='metric-cell'><div class='metric-val' style='color:{C_GREEN};'>{pct_normal}</div><div class='metric-label'>Normal Probability</div></div>"
        f"<div class='metric-cell'><div class='metric-val' style='color:{C_TEAL};font-size:16px;'>{model_name}</div><div class='metric-label'>Model Used</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Donut | Radar | Health bars — three equal panels
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='panel'><div class='panel-header'><span class='panel-header-dot'></span>Fault Status</div>", unsafe_allow_html=True)
        st.plotly_chart(_donut_gauge(result["probability_fault"], alert), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='panel'><div class='panel-header'><span class='panel-header-dot'></span>Pressure Profile</div>", unsafe_allow_html=True)
        st.plotly_chart(_radar(payload), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='panel'><div class='panel-header'><span class='panel-header-dot'></span>Health Indicators</div>", unsafe_allow_html=True)
        st.plotly_chart(_health_bars(payload), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # KPI tile grid for live derived values (HMI-style stat boxes)
    st.markdown("<br>", unsafe_allow_html=True)
    kp = payload
    pressure_drop = round(kp["TP3"] - kp["TP2"], 3)
    temp_x_cur    = round(kp["Oil_temperature"] * kp["Motor_current"], 3)
    res_delta     = round(kp["Reservoirs"] - kp["TP3"], 3)

    tiles = [
        ("Pressure Drop",   f"{pressure_drop:.3f}",            "bar",  "accent-teal"),
        ("Oil Temperature", f"{kp['Oil_temperature']:.1f}",    "°C",   "accent-amber"),
        ("Motor Current",   f"{kp['Motor_current']:.2f}",      "A",    "accent-blue"),
        ("Temp × Current",  f"{temp_x_cur:.2f}",               "kW",   "accent-purple"),
        ("Reservoir Δ",     f"{res_delta:.3f}",                 "bar",  "accent-green"),
        ("Fault Prob",      f"{result['probability_fault']*100:.1f}", "%", "accent-red" if alert else "accent-green"),
    ]

    # 6 tiles in a single row using CSS grid via columns
    cols = st.columns(6)
    for col, (label, val, unit, accent) in zip(cols, tiles):
        a_color = {"accent-teal": C_TEAL, "accent-amber": C_AMBER, "accent-blue": C_BLUE,
                   "accent-purple": C_PURPLE, "accent-green": C_GREEN, "accent-red": C_RED}.get(accent, C_TEAL)
        with col:
            st.markdown(
                f"<div class='kpi-tile {accent}'>"
                f"<div class='kpi-big' style='color:{a_color};'>{val}</div>"
                f"<div class='kpi-unit'>{unit}</div>"
                f"<div class='kpi-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Top features (if available)
    if result.get("top_features"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='panel'><div class='panel-header'><span class='panel-header-dot'></span>Top Model Features</div>", unsafe_allow_html=True)
        tf_df = pd.DataFrame(result["top_features"])
        tf_fig = go.Figure(go.Bar(
            x=tf_df["importance"], y=tf_df["feature"], orientation="h",
            marker=dict(color=C_TEAL, opacity=0.8, line=dict(width=0)),
            text=[f"{v:.5f}" for v in tf_df["importance"]],
            textposition="outside",
            textfont=dict(size=10, color="#5a7a8a"),
        ))
        tf_fig.update_layout(**_layout(
            height=200,
            yaxis=dict(autorange="reversed", **_GRID),
            margin=dict(t=20, b=20, l=160, r=70),
        ))
        st.plotly_chart(tf_fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Model performance tab ────────────────────────────────────────────────────
def render_performance(metrics: dict | None) -> None:
    if not metrics:
        st.info("Model evaluation data unavailable — ensure the API is running and `evaluation_results.json` exists.")
        return

    keys = [k for k in metrics if isinstance(metrics[k], dict) and "validation" in metrics[k]]
    if not keys:
        st.warning("No evaluation data found.")
        return

    best_key = max(keys, key=lambda k: metrics[k]["validation"]["f1"])
    best     = metrics[best_key]

    # Best model KPI row
    st.markdown("### 🏆 Best Model")
    highlights = [
        ("Model",     best_key.replace("_"," ").title(), C_TEAL,   "accent-teal"),
        ("Val F1",    f"{best['validation']['f1']:.4f}",  C_GREEN,  "accent-green"),
        ("Val Recall",f"{best['validation']['recall']:.4f}", C_AMBER,"accent-amber"),
        ("Val AUC",   f"{best['validation']['roc_auc']:.4f}",C_PURPLE,"accent-purple"),
    ]
    for col, (label, value, color, accent) in zip(st.columns(4), highlights):
        with col:
            st.markdown(
                f"<div class='kpi-tile {accent}' style='text-align:center;padding:16px;'>"
                f"<div class='kpi-big' style='color:{color};font-size:22px;'>{value}</div>"
                f"<div class='kpi-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(_model_comparison(metrics), width="stretch")

    # Confusion matrices
    st.markdown("### 🔢 Confusion Matrices — Validation Set")
    cm_cols = st.columns(len(keys))
    for col, k in zip(cm_cols, keys):
        with col:
            st.plotly_chart(_confusion_heatmap(metrics, k), width="stretch")

    # Detailed table
    st.markdown("### 📋 Full Metrics")
    rows = []
    for k in keys:
        v = metrics[k]["validation"]
        t = metrics[k]["test"]
        rows.append({
            "Model":      k.replace("_"," ").title(),
            "Val F1":     v["f1"],    "Val Recall": v["recall"],
            "Val Prec":   v["precision"], "Val AUC": v["roc_auc"],
            "Test F1":    t["f1"],    "Test Recall": t["recall"],
            "Test Prec":  t["precision"], "Test AUC": t["roc_auc"],
        })
    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        df.style.format("{:.4f}").highlight_max(color="rgba(0,200,192,0.15)", axis=0),
        width="stretch",
    )


# ─── System info tab ──────────────────────────────────────────────────────────
def render_system_info(fi_data: dict | None) -> None:
    st.markdown("### ℹ️ System & Pipeline")
    health = check_api_health(API_URL)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='panel'><div class='panel-header'><span class='panel-header-dot'></span>API Health</div>", unsafe_allow_html=True)
        if health:
            st.json(health)
        else:
            st.error(f"API unreachable at `{API_URL}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='panel'><div class='panel-header'><span class='panel-header-dot'></span>Dataset & Pipeline</div>", unsafe_allow_html=True)
        st.markdown("""
| Property | Value |
|---|---|
| Dataset | MetroPT-3 (industrial compressor) |
| Raw rows | ~1.5 M |
| After windowing | ~100 K |
| Fault rate | ~2.5% |
| Horizons | 1 hr / 6 hr / 24 hr |
| Split | 70 / 15 / 15 stratified |
| Scaler | StandardScaler |
| Imbalance | class_weight="balanced" |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    if fi_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Feature Importances")
        st.plotly_chart(_feature_importance_chart(fi_data), width="stretch")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    render_sidebar()

    # Top bar
    health     = check_api_health(API_URL)
    model_chip = health.get("model_name","—").replace("_"," ").title() if health else "Offline"
    served     = health.get("predictions_served",0) if health else 0
    now_str    = datetime.now().strftime("%H:%M:%S")
    status_dot = "🟢" if (health and health.get("model_loaded")) else "🔴"

    st.markdown(
        f"<div class='topbar'>"
        f"<div><div class='topbar-logo'>⬡ PredictMaint</div>"
        f"<div class='topbar-sub'>Compressor Predictive Maintenance · MetroPT-3</div></div>"
        f"<div class='topbar-spacer'></div>"
        f"<span class='topbar-chip'>{status_dot} {model_chip}</span>"
        f"<span class='topbar-chip'>📡 {served:,} served</span>"
        f"<span class='topbar-chip'>🕐 {now_str}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    metrics = _fetch_metrics(API_URL)
    fi_data = _fetch_fi(API_URL)

    tab1, tab2, tab3 = st.tabs(["⚡  Live Prediction", "📊  Model Performance", "ℹ️  System"])

    with tab1:
        payload = collect_inputs()
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⚡  Run Fault Analysis", type="primary", use_container_width=True):
            with st.spinner("Analysing sensor readings…"):
                result = call_predict(API_URL, payload)

            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                st.session_state.history.append({
                    "timestamp":        datetime.now().strftime("%H:%M:%S"),
                    "probability_fault":result["probability_fault"],
                    "status":           result["status"],
                    "alert":            result["alert"],
                })
                st.markdown("<hr>", unsafe_allow_html=True)
                render_result(result, payload, fi_data)

        if st.session_state.history:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 📈 Session History")
            fig_h = _history_chart(st.session_state.history)
            if fig_h:
                st.plotly_chart(fig_h, width="stretch")

            hist_df = pd.DataFrame(st.session_state.history[::-1])
            hist_df["alert"]            = hist_df["alert"].map({True: "⚠ Alert", False: "✔ Normal"})
            hist_df["probability_fault"] = hist_df["probability_fault"].map("{:.1%}".format)
            st.dataframe(hist_df, width="stretch", height=180)

    with tab2:
        render_performance(metrics)

    with tab3:
        render_system_info(fi_data)

    st.markdown(
        "<hr><div style='text-align:center;color:#1a3244;font-size:11px;padding:8px 0;'>"
        "MetroPT-3 &nbsp;·&nbsp; FastAPI 3.0 &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; "
        "Predictive Maintenance Intelligence Platform"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
