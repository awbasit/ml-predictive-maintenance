"""
Layer 5a: Streamlit dashboard.
Provides a polished UI for live predictions and result visualisation.
Communicates with the FastAPI service via HTTP.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow running as script: `streamlit run dashboard/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.api_client import call_predict, check_api_health

API_URL = os.getenv("API_URL", "http://localhost:8000")
ALERT_THRESHOLD = 0.5

SENSOR_DEFAULTS = {
    "rpm": 1499.0,
    "motor_power": 6984.0,
    "torque": 49.19,
    "outlet_pressure_bar": 4.05,
    "air_flow": 754.67,
    "noise_db": 53.41,
    "outlet_temp": 118.86,
    "wpump_outlet_press": 2.80,
    "water_inlet_temp": 83.02,
    "water_outlet_temp": 96.64,
    "wpump_power": 222.19,
    "water_flow": 53.71,
    "oilpump_power": 300.48,
    "oil_tank_temp": 46.24,
    "gaccx": 0.60,
    "gaccy": 0.35,
    "gaccz": 3.92,
    "haccx": 1.10,
    "haccy": 1.35,
    "haccz": 3.50,
}

PRESET_MULTIPLIERS = {
    "Normal baseline": 1.00,
    "High load": 1.08,
    "Stress test": 1.18,
}


def inject_styles() -> None:
    """Apply custom styling for a cleaner visual dashboard."""
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #081021 0%, #0b1427 100%);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101c3f 0%, #152959 100%) !important;
            border-right: 1px solid rgba(126, 156, 236, 0.28);
        }
        .hero {
            border: 1px solid rgba(71, 89, 130, 0.5);
            background: linear-gradient(120deg, rgba(18, 28, 52, 0.96), rgba(29, 38, 90, 0.9));
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 12px;
            box-shadow: 0 0 24px rgba(54, 120, 255, 0.25);
        }
        .hero h2 {
            color: #f3f7ff;
            margin: 0 0 6px 0;
            font-size: 1.55rem;
        }
        .hero p {
            margin: 0;
            color: #bdd2ff;
            font-size: 0.95rem;
        }
        .section-card {
            border: 1px solid rgba(84, 98, 132, 0.45);
            background: transparent;
            border-radius: 12px;
            padding: 10px 14px;
            margin-bottom: 12px;
            box-shadow: inset 0 0 0 1px rgba(107, 144, 255, 0.07);
        }
        .kpi-tile {
            border: 1px solid rgba(109, 123, 160, 0.45);
            background: linear-gradient(140deg, rgba(16, 26, 48, 0.95), rgba(19, 31, 73, 0.9));
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            box-shadow: 0 0 16px rgba(82, 122, 255, 0.18);
        }
        .kpi-label {
            color: #9cb1e2;
            font-size: 0.78rem;
            margin-bottom: 2px;
        }
        .kpi-value {
            color: #f5f8ff;
            font-size: 1.45rem;
            font-weight: 650;
            line-height: 1.1;
        }
        .kpi-sub {
            color: #7f94c6;
            font-size: 0.78rem;
            margin-top: 4px;
        }
        .mini-note {
            color: #9bb0d7;
            font-size: 0.86rem;
        }
        .title-temp {
            color: #35d7ff;
            text-shadow: 0 0 12px rgba(53, 215, 255, 0.35);
        }
        .title-live {
            color: #8ef7c1;
            text-shadow: 0 0 12px rgba(142, 247, 193, 0.3);
        }
        .title-result {
            color: #caa4ff;
            text-shadow: 0 0 14px rgba(202, 164, 255, 0.35);
        }
        .title-prob {
            color: #6cd9ff;
        }
        div[data-testid="stNumberInput"] label p {
            color: #e9f1ff !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stNumberInput"] input {
            background: rgba(8, 14, 32, 0.28) !important;
            color: #f5f8ff !important;
            border: 1px solid rgba(118, 154, 245, 0.7) !important;
            border-radius: 8px !important;
            font-size: 1.01rem !important;
        }
        div[data-testid="stNumberInput"] input:focus {
            border-color: #7ca6ff !important;
            box-shadow: 0 0 0 0.2rem rgba(78, 138, 255, 0.25) !important;
        }
        div[data-testid="stPlotlyChart"] > div {
            background: transparent !important;
        }
        button[kind="primary"] {
            background: linear-gradient(90deg, #2b63ff, #874dff) !important;
            color: #f8fbff !important;
            border: 1px solid rgba(183, 202, 255, 0.4) !important;
        }
        button[kind="primary"]:hover {
            filter: brightness(1.08);
        }
        .sidebar-card {
            border: 1px solid rgba(88, 107, 150, 0.55);
            background: linear-gradient(135deg, rgba(18, 27, 51, 0.96), rgba(18, 33, 71, 0.9));
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
            box-shadow: 0 0 16px rgba(64, 118, 255, 0.2);
        }
        .sidebar-title {
            color: #dbe7ff;
            font-size: 0.95rem;
            font-weight: 600;
            margin: 0 0 5px 0;
        }
        .sidebar-note {
            color: #9eb4e8;
            font-size: 0.82rem;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_sensor_state() -> None:
    """Seed streamlit session state with default sensor values."""
    for key, value in SENSOR_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def apply_preset(preset_name: str) -> None:
    """Apply predefined operating profile to sensor state."""
    multiplier = PRESET_MULTIPLIERS[preset_name]
    for key, value in SENSOR_DEFAULTS.items():
        st.session_state[key] = round(value * multiplier, 2)


def render_sidebar(api_url: str) -> None:
    """Render system status and controls."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <p class="sidebar-title">Control Sidebar</p>
                <p class="sidebar-note">Live API health, scenario profiles, and alert controls.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("System")
        health = check_api_health(api_url)
        if health and health.get("model_loaded"):
            st.success("API connected")
            st.metric("Model", health.get("model_name", "N/A").replace("_", " ").title())
            st.metric("Validation F1", f"{health.get('model_val_f1', 0):.4f}")
        elif health:
            st.warning("API up, model not loaded")
            st.caption(f"Endpoint: {api_url}")
        else:
            st.error("API not reachable")
            st.caption(f"Endpoint: {api_url}")

        st.divider()
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("Input profiles")
        preset = st.selectbox("Scenario preset", list(PRESET_MULTIPLIERS.keys()), index=0)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply", use_container_width=True):
                apply_preset(preset)
                st.rerun()
        with col2:
            if st.button("Reset", use_container_width=True):
                apply_preset("Normal baseline")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.caption(f"Alert threshold: noisy probability >= {ALERT_THRESHOLD:.0%}")


def render_header() -> None:
    """Render main hero header."""
    st.markdown(
        """
        <div class="hero">
            <h2>Neural Maintenance Command Center</h2>
            <p>Interactive compressor monitoring with real-time bearing risk analytics and visual diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_kpis(payload: dict) -> None:
    """Render compact KPI tiles inspired by analytics control panels."""
    water_delta = payload["water_outlet_temp"] - payload["water_inlet_temp"]
    outlet_oil_delta = payload["outlet_temp"] - payload["oil_tank_temp"]
    pressure_ratio = payload["outlet_pressure_bar"] / max(payload["air_flow"], 1.0)
    vib_ground = (payload["gaccx"] ** 2 + payload["gaccy"] ** 2 + payload["gaccz"] ** 2) ** 0.5
    vib_head = (payload["haccx"] ** 2 + payload["haccy"] ** 2 + payload["haccz"] ** 2) ** 0.5

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">Water Delta</div>
                <div class="kpi-value">{water_delta:.2f} C</div>
                <div class="kpi-sub">Cooling differential</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">Outlet vs Oil</div>
                <div class="kpi-value">{outlet_oil_delta:.2f} C</div>
                <div class="kpi-sub">Thermal stress marker</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">Pressure / Flow</div>
                <div class="kpi-value">{pressure_ratio:.4f}</div>
                <div class="kpi-sub">Mechanical loading ratio</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-tile">
                <div class="kpi-label">Vibration Delta</div>
                <div class="kpi-value">{(vib_head - vib_ground):.2f}</div>
                <div class="kpi-sub">Head-ground magnitude gap</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_temperature_inputs() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="title-temp">Temperature sensors</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Outlet temp (C)", step=0.1, key="outlet_temp")
    with c2:
        st.number_input("Water inlet temp (C)", step=0.1, key="water_inlet_temp")
    with c3:
        st.number_input("Water outlet temp (C)", step=0.1, key="water_outlet_temp")
    with c4:
        st.number_input("Oil tank temp (C)", step=0.1, key="oil_tank_temp")
    st.markdown("</div>", unsafe_allow_html=True)


def render_mechanical_inputs() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Mechanical sensors")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.number_input("RPM", step=1.0, key="rpm")
    with c2:
        st.number_input("Motor power (kW)", step=1.0, key="motor_power")
    with c3:
        st.number_input("Torque (Nm)", step=0.1, key="torque")
    with c4:
        st.number_input("Outlet pressure (bar)", step=0.01, key="outlet_pressure_bar")
    with c5:
        st.number_input("Air flow (m3/min)", step=0.1, key="air_flow")

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        st.number_input("Noise (dB)", step=0.1, key="noise_db")
    with c7:
        st.number_input("Water pump pressure (bar)", step=0.01, key="wpump_outlet_press")
    with c8:
        st.number_input("Water pump power (kW)", step=0.1, key="wpump_power")
    with c9:
        st.number_input("Water flow (m3/min)", step=0.1, key="water_flow")
    with c10:
        st.number_input("Oil pump power (kW)", step=0.1, key="oilpump_power")
    st.markdown("</div>", unsafe_allow_html=True)


def render_vibration_inputs() -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Vibration sensors")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.number_input("Ground acc X", step=0.01, key="gaccx")
    with c2:
        st.number_input("Ground acc Y", step=0.01, key="gaccy")
    with c3:
        st.number_input("Ground acc Z", step=0.01, key="gaccz")
    with c4:
        st.number_input("Head acc X", step=0.01, key="haccx")
    with c5:
        st.number_input("Head acc Y", step=0.01, key="haccy")
    with c6:
        st.number_input("Head acc Z", step=0.01, key="haccz")
    st.markdown("</div>", unsafe_allow_html=True)


def get_payload_from_state() -> dict:
    """Collect model payload from session state keys."""
    return {key: float(st.session_state[key]) for key in SENSOR_DEFAULTS}


def render_live_preview(payload: dict) -> None:
    """Render pre-inference sensor summaries."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="title-live">Live sensor snapshot</h3>', unsafe_allow_html=True)

    water_delta = payload["water_outlet_temp"] - payload["water_inlet_temp"]
    outlet_vs_oil = payload["outlet_temp"] - payload["oil_tank_temp"]
    vib_ground = (payload["gaccx"] ** 2 + payload["gaccy"] ** 2 + payload["gaccz"] ** 2) ** 0.5
    vib_head = (payload["haccx"] ** 2 + payload["haccy"] ** 2 + payload["haccz"] ** 2) ** 0.5

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Water Delta (C)", f"{water_delta:.2f}")
    c2.metric("Outlet vs Oil (C)", f"{outlet_vs_oil:.2f}")
    c3.metric("Ground Vib Magnitude", f"{vib_ground:.2f}")
    c4.metric("Head Vib Magnitude", f"{vib_head:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_result(result: dict, payload: dict) -> None:
    """Render polished prediction result area with multiple visuals."""
    st.markdown('<h2 class="title-result">Prediction result</h2>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        status_color = "🔴" if result["alert"] else "🟢"
        st.metric("Bearing status", f"{status_color} {result['status']}")
    with c2:
        st.metric("Fault probability", f"{result['probability_noisy']:.1%}")
    with c3:
        st.metric("Model", result["model_used"].replace("_", " ").title())

    if result["alert"]:
        st.error(result["alert_message"])
    else:
        st.success(result["alert_message"])

    chart_tab, signal_tab = st.tabs(["Risk view", "Sensor view"])

    with chart_tab:
        left, right = st.columns([2, 1])
        with left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["probability_noisy"] * 100,
                    title={"text": "Bearing fault probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ff4d5a" if result["alert"] else "#22c88a"},
                        "steps": [
                            {"range": [0, 50], "color": "#10293e"},
                            {"range": [50, 100], "color": "#3a1621"},
                        ],
                        "threshold": {
                            "line": {"color": "#ff4d5a", "width": 3},
                            "thickness": 0.75,
                            "value": ALERT_THRESHOLD * 100,
                        },
                    },
                )
            )
            gauge.update_layout(height=330, margin=dict(t=45, b=10))
            gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#eaf1ff"},
            )
            st.plotly_chart(gauge, use_container_width=True)

        with right:
            donut = go.Figure(
                data=[
                    go.Pie(
                        labels=["Noisy", "Ok"],
                        values=[result["probability_noisy"], result["probability_ok"]],
                        hole=0.62,
                        marker={"colors": ["#8d5bff", "#40e0d0"]},
                        textinfo="label+percent",
                    )
                ]
            )
            donut.update_layout(
                height=330,
                margin=dict(t=45, b=10),
                title={"text": "<span class='title-prob'>Probability split</span>"},
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#eaf1ff"},
            )
            st.plotly_chart(donut, use_container_width=True)

    with signal_tab:
        temp_df = pd.DataFrame(
            {
                "Sensor": ["Outlet", "Water Inlet", "Water Outlet", "Oil Tank"],
                "Value (C)": [
                    payload["outlet_temp"],
                    payload["water_inlet_temp"],
                    payload["water_outlet_temp"],
                    payload["oil_tank_temp"],
                ],
            }
        )
        fig_temp = px.bar(
            temp_df,
            x="Sensor",
            y="Value (C)",
            color="Value (C)",
            color_continuous_scale=["#2ec5ff", "#ffbf45", "#ff4d5a"],
            title="Temperature profile",
        )
        fig_temp.update_layout(height=330, margin=dict(t=45, b=10))
        fig_temp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eaf1ff"},
        )

        vib_df = pd.DataFrame(
            {
                "axis": ["X", "Y", "Z"],
                "ground": [payload["gaccx"], payload["gaccy"], payload["gaccz"]],
                "head": [payload["haccx"], payload["haccy"], payload["haccz"]],
            }
        )
        fig_vib = go.Figure()
        fig_vib.add_trace(
            go.Scatterpolar(
                r=vib_df["ground"],
                theta=vib_df["axis"],
                fill="toself",
                name="Ground",
                line_color="#2ec5ff",
            )
        )
        fig_vib.add_trace(
            go.Scatterpolar(
                r=vib_df["head"],
                theta=vib_df["axis"],
                fill="toself",
                name="Head",
                line_color="#ff4d5a",
            )
        )
        fig_vib.update_layout(
            title="Vibration comparison (ground vs head)",
            polar={"radialaxis": {"visible": True}},
            height=330,
            margin=dict(t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eaf1ff"},
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(fig_temp, use_container_width=True)
        with right:
            st.plotly_chart(fig_vib, use_container_width=True)

def main() -> None:
    st.set_page_config(
        page_title="AC Compressor Predictive Maintenance",
        page_icon="🌡️",
        layout="wide",
    )
    inject_styles()
    init_sensor_state()
    render_sidebar(API_URL)
    render_header()

    input_tab, preview_tab = st.tabs(["Input panel", "Live preview"])
    with input_tab:
        temp_tab, mech_tab, vib_tab = st.tabs(["Temperature", "Mechanical", "Vibration"])
        with temp_tab:
            render_temperature_inputs()
        with mech_tab:
            render_mechanical_inputs()
        with vib_tab:
            render_vibration_inputs()
    payload = get_payload_from_state()
    render_top_kpis(payload)
    with preview_tab:
        render_live_preview(payload)

    st.markdown('<p class="mini-note">Tip: use preset profiles in sidebar and then tweak only key signals.</p>', unsafe_allow_html=True)
    if st.button("Run prediction", type="primary", use_container_width=True):
        with st.spinner("Running prediction..."):
            result = call_predict(API_URL, payload)
        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            render_prediction_result(result, payload)

    st.caption("Powered by FastAPI + Streamlit")


if __name__ == "__main__":
    main()
