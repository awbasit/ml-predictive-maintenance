"""
Predictive Maintenance Dashboard — HMI-inspired Streamlit interface.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_app_dir = Path(__file__).resolve().parent
# Monorepo: dashboard/app.py → repo root has utils/
# HF Space: app.py at Space root with utils/ alongside
PROJECT_ROOT = _app_dir if (_app_dir / "utils").is_dir() else _app_dir.parent
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ══════════════════════════════════════════════════
   STREAMLIT CHROME — keep sidebar toggle, hide clutter
   ══════════════════════════════════════════════════ */
#MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stMainMenuPopover"],
[data-testid="manage-app-button"] { display: none !important; }

/* Minimal header — keeps the sidebar open/close control visible */
[data-testid="stHeader"] {
    background: #050f1a !important;
    border-bottom: 1px solid #112030;
    height: 3rem !important;
    min-height: 3rem !important;
    z-index: 1000 !important;
}
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    color: #00c8c0 !important;
    z-index: 1001 !important;
}

/* ══════════════════════════════════════
   SIDEBAR — fixed width, styled
   ══════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #081524 !important;
    border-right: 1px solid #112030;
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
}
/* Collapse the default Streamlit sidebar top spacer to zero */
[data-testid="stSidebarHeader"] {
    min-height: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}
section[data-testid="stSidebar"] > div {
    background: #081524 !important;
    padding: 16px 14px 20px !important;
    box-sizing: border-box !important;
}
/* Scrollbar */
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 5px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: #1a3244; border-radius: 3px;
}

/* ── Sidebar text ── */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small {
    color: #c8dce8 !important;
}
section[data-testid="stSidebar"] .stat-key { color: #6a8a9a !important; }
section[data-testid="stSidebar"] .stat-val { color: #d0e8f5 !important; }
section[data-testid="stSidebar"] .sidebar-label { color: #00c8c0 !important; }

/* ── "Active profile: Normal" — force correct size ── */
section[data-testid="stSidebar"] .scenario-active,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p.scenario-active {
    font-size: 11px !important;
    color: #8aaaba !important;
    margin: 4px 0 8px !important;
    line-height: 1.5 !important;
}
section[data-testid="stSidebar"] .scenario-active strong {
    color: #d0e8f5 !important;
    font-size: 11px !important;
}

/* ── Scenario buttons: Normal = green, Faulty = red ── */
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button {
    background: rgba(0,224,150,.10) !important;
    border: 1px solid rgba(0,224,150,.55) !important;
    color: #00e096 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button:hover {
    background: rgba(0,224,150,.20) !important;
    border-color: #00e096 !important;
}
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button {
    background: rgba(255,61,90,.10) !important;
    border: 1px solid rgba(255,61,90,.55) !important;
    color: #ff3d5a !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child .stButton > button:hover {
    background: rgba(255,61,90,.20) !important;
    border-color: #ff3d5a !important;
}

/* ── Generic sidebar buttons (Clear Session, expander default) ── */
section[data-testid="stSidebar"] .stButton > button {
    color: #d0e8f5 !important;
    background: #0c1e30 !important;
    border: 1px solid #1a3244 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #c8dce8 !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] [data-testid="stElementContainer"] {
    margin-bottom: 2px !important;
}
section[data-testid="stSidebar"] [data-testid="column"] {
    padding: 0 3px !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #8aaaba !important;
}

/* Main content padding */
section[data-testid="stMain"] > div.block-container {
    padding: 3rem 1.75rem 2.5rem !important;
}

/* ══════════════════════════════════════
   BASE
   ══════════════════════════════════════ */
html, body, [data-testid="stAppViewContainer"] {
    background: #050f1a !important;
    font-family: 'Inter', system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background: #081524 !important;
    border-right: 1px solid #112030;
}
.block-container { max-width: 100% !important; }

/* ══════════════════════════════════════
   TYPOGRAPHY
   ══════════════════════════════════════ */
h1,h2,h3,h4 { color: #d0e8f5 !important; font-weight: 700 !important; }
p, li       { color: #5a7a8a; }

/* ══════════════════════════════════════
   TOPBAR
   ══════════════════════════════════════ */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    background: #081524;
    border: 1px solid #112030;
    border-radius: 10px;
    padding: 10px 20px;
    margin-bottom: 18px;
}
.topbar-sub-inline {
    font-size: 10px;
    color: #3a5a6a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    white-space: nowrap;
    flex: 1;
}
.topbar-chip {
    background: #0c1e30;
    border: 1px solid #1a3244;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    color: #7a9ab0;
    white-space: nowrap;
    flex: 1;
    text-align: center;
}

/* ══════════════════════════════════════
   PANEL CARDS
   ══════════════════════════════════════ */
.panel {
    background: #0c1e30;
    border: 1px solid #1a3244;
    border-radius: 12px;
    padding: 14px 18px 18px;
}
.panel-header {
    font-size: 9px;
    font-weight: 700;
    color: #3a5a6a;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #112030;
    padding-bottom: 8px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.panel-header-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00c8c0;
    box-shadow: 0 0 5px #00c8c0;
    flex-shrink: 0;
}

/* ══════════════════════════════════════
   KPI TILES
   ══════════════════════════════════════ */
.kpi-tile {
    background: #081524;
    border: 1px solid #112030;
    border-radius: 8px;
    padding: 10px 12px;
    height: 100%;
    box-sizing: border-box;
}
.kpi-tile.accent-teal   { border-top: 2px solid #00c8c0; }
.kpi-tile.accent-blue   { border-top: 2px solid #0094c6; }
.kpi-tile.accent-red    { border-top: 2px solid #ff3d5a; }
.kpi-tile.accent-green  { border-top: 2px solid #00e096; }
.kpi-tile.accent-amber  { border-top: 2px solid #ffb300; }
.kpi-tile.accent-purple { border-top: 2px solid #9b72cf; }
.kpi-big   { font-size: clamp(18px,2.5vw,28px); font-weight: 800; color: #d0e8f5; line-height: 1.1; margin: 2px 0; }
.kpi-unit  { font-size: 10px; color: #3a5a6a; font-weight: 600; }
.kpi-label { font-size: 8px; color: #3a5a6a; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 3px; }

/* ══════════════════════════════════════
   4-CELL METRIC ROW
   ══════════════════════════════════════ */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 12px 0;
}
.metric-cell {
    background: #0c1e30;
    border: 1px solid #1a3244;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    min-width: 0;
}
.metric-cell:hover { border-color: #264a60; }
.metric-val   { font-size: clamp(14px,1.8vw,22px); font-weight: 700; color: #d0e8f5; line-height: 1.2; word-break: break-word; }
.metric-label { font-size: 9px; color: #3a5a6a; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ══════════════════════════════════════
   ALERT BANNERS
   ══════════════════════════════════════ */
.banner {
    border-radius: 10px;
    padding: 12px 18px;
    font-weight: 600;
    font-size: clamp(13px,1.5vw,15px);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.banner-fault  { background:rgba(255,61,90,.08);  border:1px solid #ff3d5a; border-left:4px solid #ff3d5a; color:#ff3d5a; }
.banner-normal { background:rgba(0,224,150,.07);  border:1px solid #00e096; border-left:4px solid #00e096; color:#00e096; }

/* ══════════════════════════════════════
   HERO TITLE BLOCK
   ══════════════════════════════════════ */
.hero {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 4px 0 14px;
    margin-top: 0;
}
.hero-icon {
    font-size: clamp(28px, 3.5vw, 42px);
    line-height: 1;
    color: #00c8c0;
    filter: drop-shadow(0 0 14px rgba(0,200,192,0.50));
    flex-shrink: 0;
}
.hero-title {
    font-size: clamp(20px, 2.8vw, 34px);
    font-weight: 900;
    color: #d0e8f5;
    letter-spacing: -1px;
    line-height: 1.15;
}
.hero-title span { color: #00c8c0; }
.hero-sub {
    font-size: clamp(10px, 1.1vw, 12px);
    color: #3a5a6a;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-top: 5px;
}

/* ══════════════════════════════════════
   SENSOR PANEL — CONTAINER OVERRIDES
   ══════════════════════════════════════ */

/* Bordered st.container() → our dark card style */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #1a3244 !important;
    border-radius: 12px !important;
    background-color: #0c1e30 !important;
    padding: 6px 10px 12px !important;
    transition: border-color .2s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #264a60 !important;
}

/* Section heading inside a sensor container */
.sensor-group-head {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 9px;
    font-weight: 800;
    color: #00c8c0;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    padding-bottom: 10px;
    margin-bottom: 4px;
    border-bottom: 1px solid #112030;
}
.sensor-group-head::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 11px;
    background: #00c8c0;
    border-radius: 2px;
    box-shadow: 0 0 6px rgba(0,200,192,.5);
    flex-shrink: 0;
}

/* Individual slider value display */
.sensor-val-card {
    background: #081524;
    border: 1px solid #112030;
    border-radius: 8px;
    padding: 8px 12px 4px;
    margin-bottom: 4px;
    text-align: center;
}
.sensor-val-card .sv-name  { font-size: 8px; font-weight: 700; color: #3a5a6a; text-transform: uppercase; letter-spacing: 1.5px; }
.sensor-val-card .sv-value { font-size: clamp(16px,1.8vw,22px); font-weight: 800; color: #00c8c0; line-height: 1.1; margin: 2px 0; }
.sensor-val-card .sv-unit  { font-size: 9px; color: #3a5a6a; }

/* Digital toggle row */
.dig-chip-row { display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 0; }

/* ══════════════════════════════════════
   SENSOR SECTION LABELS (legacy fallback)
   ══════════════════════════════════════ */
.sensor-section {
    font-size: 8px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #3a5a6a;
    margin: 14px 0 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #112030;
}

/* ══════════════════════════════════════
   RUN FAULT ANALYSIS — HMI ACTION BUTTON
   ══════════════════════════════════════ */
@keyframes hmi-shimmer {
    0%   { background-position: 200% center; }
    100% { background-position: -200% center; }
}
@keyframes hmi-pulse {
    0%, 100% { box-shadow: 0 0 18px rgba(0,200,192,.35), 0 0 0 0 rgba(0,200,192,.2); }
    50%       { box-shadow: 0 0 30px rgba(0,200,192,.55), 0 0 0 6px rgba(0,200,192,.0); }
}

.stButton > button[kind="primary"] {
    width: 100% !important;
    background: linear-gradient(90deg,
        #003d3b 0%, #006b68 25%, #00c8c0 50%, #006b68 75%, #003d3b 100%) !important;
    background-size: 300% 100% !important;
    animation: hmi-shimmer 5s linear infinite, hmi-pulse 3s ease-in-out infinite !important;
    border: 1px solid rgba(0,200,192,0.5) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: clamp(13px,1.3vw,16px) !important;
    font-weight: 900 !important;
    height: 60px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
}
.stButton > button[kind="primary"]:hover {
    animation-play-state: paused !important;
    background: linear-gradient(90deg, #00a89f, #00c8c0) !important;
    box-shadow: 0 0 40px rgba(0,200,192,.65) !important;
    border-color: #00c8c0 !important;
}
/* Force white text on all child elements Streamlit creates */
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: clamp(13px,1.3vw,16px) !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    background: transparent !important;
}

/* Secondary (Clear Session) button */
.stButton > button:not([kind="primary"]) {
    background: #0c1e30 !important;
    border: 1px solid #1a3244 !important;
    border-radius: 8px !important;
    color: #5a7a8a !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #264a60 !important;
    color: #d0e8f5 !important;
}

/* ══════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════ */
.sidebar-label {
    font-size: 8px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #00c8c0;
    margin: 12px 0 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #112030;
}
.sidebar-label:first-of-type { margin-top: 6px; }
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #0c1e30;
    font-size: 12px;
    gap: 8px;
}
.stat-key { color: #3a5a6a; white-space: nowrap; }
.stat-val { color: #d0e8f5; font-weight: 600; text-align: right; word-break: break-word; }

/* ── Demo scenario sidebar ── */
.scenario-values {
    background: #081524;
    border: 1px solid #112030;
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 10px;
    max-height: 280px;
    overflow-y: auto;
}
.scenario-values-title {
    font-size: 9px;
    font-weight: 700;
    color: #00c8c0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.scenario-values-divider {
    border-top: 1px solid #112030;
    margin: 8px 0;
}
.scenario-active {
    font-size: 11px;
    color: #7a9ab0;
    margin: 6px 0 2px;
}
.scenario-active strong { color: #d0e8f5; }
div[data-testid="stSidebar"] .stButton > button.scenario-normal-btn {
    background: rgba(0,224,150,.12) !important;
    border: 1px solid #00e096 !important;
    color: #00e096 !important;
    font-weight: 700 !important;
}
div[data-testid="stSidebar"] .stButton > button.scenario-fault-btn {
    background: rgba(255,61,90,.12) !important;
    border: 1px solid #ff3d5a !important;
    color: #ff3d5a !important;
    font-weight: 700 !important;
}

/* ── Status badge ── */
.sbadge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}
.sbadge-online  { background:rgba(0,200,192,.12); color:#00c8c0; border:1px solid #00c8c0; }
.sbadge-offline { background:rgba(255,61,90,.12);  color:#ff3d5a; border:1px solid #ff3d5a; }

/* ── Sidebar status bar ── */
.sidebar-status-bar {
    background: #0c1e30;
    border: 1px solid #112030;
    border-radius: 8px;
    padding: 8px 10px;
    margin: 4px 0 8px;
}
.sidebar-status-bar .sbadge {
    display: flex;
    width: 100%;
    justify-content: center;
    margin-bottom: 6px;
    padding: 3px 8px;
    font-size: 10px;
}
.sidebar-status-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
}
.sidebar-status-cell {
    background: #081524;
    border: 1px solid #1a3244;
    border-radius: 5px;
    padding: 4px 7px;
    font-size: 9px;
    color: #6a8a9a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    line-height: 1.2;
}
.sidebar-status-cell strong {
    display: block;
    color: #d0e8f5;
    font-size: 11px;
    font-weight: 600;
    text-transform: none;
    letter-spacing: 0;
    margin-top: 1px;
}

/* ══════════════════════════════════════
   TABS
   ══════════════════════════════════════ */
[data-testid="stTabs"] button { color:#3a5a6a !important; font-weight:600 !important; font-size:clamp(11px,1.2vw,13px) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#00c8c0 !important;
    border-bottom-color:#00c8c0 !important;
}

/* ══════════════════════════════════════
   MISC
   ══════════════════════════════════════ */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
hr { border-color: #112030 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #081524; }
::-webkit-scrollbar-thumb { background: #1a3244; border-radius: 3px; }

/* ══════════════════════════════════════
   RESPONSIVE BREAKPOINTS
   ══════════════════════════════════════ */
@media screen and (max-width: 1100px) {
    .metric-row { grid-template-columns: repeat(2, 1fr); }
    .topbar-sub-inline { display: none; }
}
@media screen and (max-width: 768px) {
    .metric-row { grid-template-columns: repeat(2, 1fr); gap: 6px; }
    .block-container { padding: 12px 10px 30px !important; }
    .topbar { padding: 8px 12px; flex-direction: column; align-items: flex-start; }
    .topbar-chip, .topbar-sub-inline { display: none; }
    .kpi-big { font-size: 18px !important; }
    .panel { padding: 10px 12px 14px; }
    .metric-cell { padding: 10px 10px; }
}
@media screen and (max-width: 480px) {
    .metric-row { grid-template-columns: 1fr 1fr; gap: 5px; }
    .banner { font-size: 12px; padding: 10px 12px; }
    .kpi-big { font-size: 16px !important; }
}
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
        "font":     _FONT,
        "autosize": True,
        "xaxis":    kw.pop("xaxis",  dict(**_GRID)),
        "yaxis":    kw.pop("yaxis",  dict(**_GRID)),
        "margin":   kw.pop("margin", dict(t=50, b=40, l=50, r=20)),
        "height":   height,
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
    "TP2":            (-0.05, 11.0,  -0.012, "bar"),
    "TP3":            (7.0,  12.0,   9.358, "bar"),
    "H1":             (-0.05, 11.0,   9.340, "bar"),
    "DV_pressure":    (-0.05,  7.0,  -0.024, "bar"),
    "Reservoirs":     (7.0,  12.0,   9.358, "bar"),
    "Oil_temperature":(30.0, 100.0,   53.6,  "°C"),
    "Motor_current":  (0.0,  10.0,   0.04,  "A"),
}
DIGITAL_DEFAULTS = {
    "COMP": True, "DV_eletric": False, "Towers": True, "MPG": True,
    "LPS": False, "Pressure_switch": True, "Oil_level": True, "Caudal_impulses": True,
}

# Demo profiles — Normal = healthy baseline; Faulty = pre-fault window sample (6 hr horizon)
SCENARIO_PRESETS: dict[str, dict[str, float | bool]] = {
    "Normal": {
        "TP2": -0.018, "TP3": 9.984, "H1": 9.976, "DV_pressure": -0.020,
        "Reservoirs": 9.982, "Oil_temperature": 73.625, "Motor_current": 3.748,
        "COMP": True, "DV_eletric": False, "Towers": True, "MPG": True,
        "LPS": False, "Pressure_switch": True, "Oil_level": False, "Caudal_impulses": True,
    },
    "Faulty": {
        "TP2": 7.584, "TP3": 8.106, "H1": -0.006, "DV_pressure": 2.036,
        "Reservoirs": 8.108, "Oil_temperature": 76.15, "Motor_current": 5.57,
        "COMP": False, "DV_eletric": True, "Towers": False, "MPG": False,
        "LPS": False, "Pressure_switch": False, "Oil_level": True, "Caudal_impulses": True,
    },
}

def _input_key(sensor: str) -> str:
    return f"in_{sensor}"


def _init_sensor_state() -> None:
    """Seed session state once with the Normal operating profile."""
    if st.session_state.get("sensors_initialized"):
        return
    for key, val in SCENARIO_PRESETS["Normal"].items():
        st.session_state[_input_key(key)] = val
    st.session_state.active_scenario = "Normal"
    st.session_state.sensors_initialized = True


def apply_scenario(name: str) -> None:
    """Load a demo profile into all sensor widgets and queue an auto-prediction."""
    preset = SCENARIO_PRESETS[name]
    for key, val in preset.items():
        st.session_state[_input_key(key)] = val
    st.session_state.active_scenario = name
    st.session_state.run_prediction = True


def _format_preset_value(key: str, val: float | bool) -> str:
    if isinstance(val, bool):
        return "ON" if val else "OFF"
    if key in ("Oil_temperature",):
        return f"{float(val):.1f}"
    if key in ("Motor_current",):
        return f"{float(val):.2f}"
    return f"{float(val):.3f}"


def _render_scenario_values(name: str) -> None:
    """Show the sensor values for the selected demo profile."""
    preset = SCENARIO_PRESETS[name]
    analogue_keys = list(ANALOGUE.keys())
    rows_html = "".join(
        f"<div class='stat-row'>"
        f"<span class='stat-key'>{key}</span>"
        f"<span class='stat-val'>{_format_preset_value(key, preset[key])}</span>"
        f"</div>"
        for key in analogue_keys
    )
    digital_html = "".join(
        f"<div class='stat-row'>"
        f"<span class='stat-key'>{key}</span>"
        f"<span class='stat-val'>{_format_preset_value(key, preset[key])}</span>"
        f"</div>"
        for key in DIGITAL_DEFAULTS
    )
    st.markdown(
        f"<div class='scenario-values'>"
        f"<div class='scenario-values-title'>{name} profile</div>"
        f"{rows_html}"
        f"<div class='scenario-values-divider'></div>"
        f"{digital_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

# ─── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─── Cached API calls ─────────────────────────────────────────────────────────
def _load_eval_results_from_disk() -> dict | None:
    """Read evaluation_results.json directly — used when the API is offline."""
    candidates = [
        PROJECT_ROOT / "models" / "evaluation_results.json",
        PROJECT_ROOT / "evaluation_results.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    return None


@st.cache_data(ttl=60)
def _fetch_metrics(api_url: str) -> dict | None:
    data = get_metrics(api_url)
    if data is None:
        data = _load_eval_results_from_disk()
    return data


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
    sensors = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"]
    norm = [
        max(0.0, min(1.0, (payload[s] - ANALOGUE[s][0]) / (ANALOGUE[s][1] - ANALOGUE[s][0] + 1e-9)))
        for s in sensors
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
def _render_sidebar_status_bar(health: dict | None) -> None:
    online = health and health.get("model_loaded")
    now_str = datetime.now().strftime("%H:%M:%S")
    if online:
        model = health.get("model_name", "—").replace("_", " ").title()
        val_f1 = f"{health.get('model_val_f1', 0):.4f}"
        served = f"{health.get('predictions_served', 0):,}"
        st.markdown(
            "<div class='sidebar-status-bar'>"
            "<span class='sbadge sbadge-online'>⬤ API Online</span>"
            "<div class='sidebar-status-grid'>"
            f"<div class='sidebar-status-cell'>Model<strong>{model}</strong></div>"
            f"<div class='sidebar-status-cell'>Val F1<strong>{val_f1}</strong></div>"
            f"<div class='sidebar-status-cell'>Served<strong>{served}</strong></div>"
            f"<div class='sidebar-status-cell'>Time<strong>{now_str}</strong></div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='sidebar-status-bar'>"
            "<span class='sbadge sbadge-offline'>⬤ API Offline</span>"
            f"<div class='sidebar-status-grid'>"
            f"<div class='sidebar-status-cell'>Time<strong>{now_str}</strong></div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"`{API_URL}`")


def render_sidebar(health: dict | None = None) -> None:
    if health is None:
        health = check_api_health(API_URL)

    with st.sidebar:
        st.markdown(
            "<div style='padding:4px 0 4px;'>"
            "<div style='font-size:18px;font-weight:800;color:#00c8c0;letter-spacing:-0.5px;'>⬡ PredictMaint</div>"
            "<div style='font-size:10px;color:#5a8a9a;text-transform:uppercase;"
            "letter-spacing:2px;margin-top:3px;'>MetroPT-3 · Compressor AI</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        _render_sidebar_status_bar(health)

        st.markdown(
            "<div class='sidebar-label' style='margin-top:6px;margin-bottom:2px;'>Demo Scenarios</div>"
            "<p style='font-size:11px;color:#8aaaba;margin:0 0 4px;line-height:1.3;'>"
            "Load a sensor profile and run inference automatically.</p>"
            f"<p class='scenario-active' style='margin:0 0 6px;'>Active profile: "
            f"<strong>{st.session_state.get('active_scenario', 'Normal')}</strong></p>",
            unsafe_allow_html=True,
        )

        active = st.session_state.get("active_scenario", "Normal")
        sc1, sc2 = st.columns(2, gap="small")
        with sc1:
            if st.button("🟢 Normal", use_container_width=True, key="btn_scenario_normal"):
                apply_scenario("Normal")
                st.rerun()
        with sc2:
            if st.button("🔴 Faulty", use_container_width=True, key="btn_scenario_faulty"):
                apply_scenario("Faulty")
                st.rerun()

        with st.expander("View profile values", expanded=False):
            _render_scenario_values(active)

        with st.expander("Configuration", expanded=False):
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

    # ── Pressure Sensors ────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            "<div class='sensor-group-head'>Pressure Sensors</div>",
            unsafe_allow_html=True,
        )
        p_keys = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs"]
        p_cols = st.columns(5)
        pv: dict[str, float] = {}
        for col, key in zip(p_cols, p_keys):
            lo, hi, default, unit = ANALOGUE[key]
            with col:
                val = st.slider(
                    f"{key}",
                    min_value=float(lo), max_value=float(hi),
                    step=0.001 if abs(hi - lo) <= 2 else 0.01,
                    format="%.3f",
                    label_visibility="collapsed",
                    key=_input_key(key),
                )
                pv[key] = val
                st.markdown(
                    f"<div class='sensor-val-card'>"
                    f"<div class='sv-name'>{key}</div>"
                    f"<div class='sv-value'>{val:.3f}</div>"
                    f"<div class='sv-unit'>{unit}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Temperature & Current ────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            "<div class='sensor-group-head'>Temperature &amp; Current</div>",
            unsafe_allow_html=True,
        )
        tc = st.columns(2)
        with tc[0]:
            oil_temp = st.slider(
                "Oil Temperature", min_value=30.0, max_value=100.0,
                step=0.1, label_visibility="collapsed",
                key=_input_key("Oil_temperature"),
            )
            st.markdown(
                f"<div class='sensor-val-card'>"
                f"<div class='sv-name'>Oil Temperature</div>"
                f"<div class='sv-value'>{oil_temp:.1f}</div>"
                f"<div class='sv-unit'>°C</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with tc[1]:
            motor_curr = st.slider(
                "Motor Current", min_value=0.0, max_value=10.0,
                step=0.01, label_visibility="collapsed",
                key=_input_key("Motor_current"),
            )
            st.markdown(
                f"<div class='sensor-val-card'>"
                f"<div class='sv-name'>Motor Current</div>"
                f"<div class='sv-value'>{motor_curr:.2f}</div>"
                f"<div class='sv-unit'>A</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Digital Valve Signals ────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown(
            "<div class='sensor-group-head'>Digital Valve Signals</div>",
            unsafe_allow_html=True,
        )
        digital_items = list(DIGITAL_DEFAULTS.items())
        row1, row2 = digital_items[:4], digital_items[4:]
        dv: dict[str, float] = {}
        for row in (row1, row2):
            d_cols = st.columns(4)
            for col, (sensor, default) in zip(d_cols, row):
                with col:
                    dv[sensor] = float(st.toggle(sensor, key=_input_key(sensor)))

    return {**pv, "Oil_temperature": oil_temp, "Motor_current": motor_curr, **dv}


def _handle_prediction(payload: dict, fi_data: dict | None) -> None:
    """Call the API and render the prediction panel."""
    with st.spinner("Analysing sensor readings…"):
        result = call_predict(API_URL, payload)

    if "error" in result:
        st.error(f"Prediction failed: {result['error']}")
        return

    st.session_state.history.append({
        "timestamp":         datetime.now().strftime("%H:%M:%S"),
        "probability_fault": result["probability_fault"],
        "status":            result["status"],
        "alert":             result["alert"],
    })
    st.markdown("<hr>", unsafe_allow_html=True)
    render_result(result, payload, fi_data)


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


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    _init_sensor_state()
    health = check_api_health(API_URL)
    render_sidebar(health)

    # ── Hero title ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='hero'>"
        "<div class='hero-icon'>⬡</div>"
        "<div>"
        "<div class='hero-title'>Compressor <span>Predictive</span><br>Maintenance</div>"
        "<div class='hero-sub'>MetroPT-3 Industrial Air Compressor · Real-time Fault Detection AI</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    metrics = _fetch_metrics(API_URL)
    fi_data = _fetch_fi(API_URL)

    tab1, tab2 = st.tabs(["⚡  Live Prediction", "📊  Model Performance"])

    with tab1:
        payload = collect_inputs()
        st.markdown("<br>", unsafe_allow_html=True)

        run_clicked = st.button("⚡  ANALYSE SENSOR READINGS", type="primary", use_container_width=True)
        auto_run = st.session_state.pop("run_prediction", False)

        if run_clicked or auto_run:
            _handle_prediction(payload, fi_data)

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

    st.markdown(
        "<hr><div style='text-align:center;color:#1a3244;font-size:11px;padding:8px 0;'>"
        "MetroPT-3 &nbsp;·&nbsp; FastAPI 3.0 &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; "
        "Predictive Maintenance Intelligence Platform"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
