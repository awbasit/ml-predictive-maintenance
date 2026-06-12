"""
Layer 9: Streamlit dashboard for MetroPT-3 predictive maintenance.
"""

from __future__ import annotations

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

st.set_page_config(page_title="UMaT AC Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("⚙️ Predictive Maintenance System")
st.caption("Case study: UMaT Air Conditioning Compressors (MetroPT-3, 6hr horizon)")
st.divider()


def render_sidebar():
    with st.sidebar:
        st.subheader("System status")
        health = check_api_health(API_URL)
        if health and health.get("model_loaded"):
            st.success("API connected")
            st.metric("Model", health.get("model_name", "N/A").replace("_", " ").title())
            st.metric("Validation F1", f"{health.get('model_val_f1', 0):.4f}")
        else:
            st.error("API not reachable")
            st.caption(f"Endpoint: {API_URL}")

        st.divider()
        st.caption("Alert threshold: fault probability >= 50%")
        st.caption("Prediction horizon: 6 hours pre-fault")
        st.caption("MetroPT-3 real industrial compressor signals")


def collect_inputs() -> dict:
    st.subheader("Sensor input")

    st.markdown("**Pressure sensors (bar)**")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        TP2 = st.number_input("TP2", value=-0.012, step=0.001, format="%.3f")
    with c2:
        TP3 = st.number_input("TP3", value=9.358, step=0.001, format="%.3f")
    with c3:
        H1 = st.number_input("H1", value=9.340, step=0.001, format="%.3f")
    with c4:
        DV_pressure = st.number_input("DV pressure", value=-0.024, step=0.001, format="%.3f")
    with c5:
        Reservoirs = st.number_input("Reservoirs", value=9.358, step=0.001, format="%.3f")

    st.markdown("**Temperature & current**")
    c6, c7 = st.columns(2)
    with c6:
        Oil_temperature = st.number_input("Oil temperature (C)", value=53.6, step=0.1)
    with c7:
        Motor_current = st.number_input("Motor current (A)", value=0.04, step=0.01)

    st.markdown("**Digital valve signals (0/1)**")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        COMP = st.selectbox("COMP", [0.0, 1.0], index=1)
        MPG = st.selectbox("MPG", [0.0, 1.0], index=1)
    with d2:
        DV_eletric = st.selectbox("DV_eletric", [0.0, 1.0], index=0)
        LPS = st.selectbox("LPS", [0.0, 1.0], index=0)
    with d3:
        Towers = st.selectbox("Towers", [0.0, 1.0], index=1)
        Pressure_switch = st.selectbox("Pressure_switch", [0.0, 1.0], index=1)
    with d4:
        Oil_level = st.selectbox("Oil_level", [0.0, 1.0], index=1)
        Caudal_impulses = st.selectbox("Caudal_impulses", [0.0, 1.0], index=1)

    return {
        "TP2": TP2,
        "TP3": TP3,
        "H1": H1,
        "DV_pressure": DV_pressure,
        "Reservoirs": Reservoirs,
        "Oil_temperature": Oil_temperature,
        "Motor_current": Motor_current,
        "COMP": COMP,
        "DV_eletric": DV_eletric,
        "Towers": Towers,
        "MPG": MPG,
        "LPS": LPS,
        "Pressure_switch": Pressure_switch,
        "Oil_level": Oil_level,
        "Caudal_impulses": Caudal_impulses,
    }


def render_result(result: dict, payload: dict):
    st.subheader("Prediction result")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        icon = "🔴" if result["alert"] else "🟢"
        st.metric("Status", f"{icon} {result['status']}")
    with c2:
        st.metric("Fault probability", f"{result['probability_fault']:.1%}")
    with c3:
        st.metric("Normal probability", f"{result['probability_normal']:.1%}")
    with c4:
        st.metric("Model", result["model_used"].replace("_", " ").title())

    if result["alert"]:
        st.error(result["alert_message"])
    else:
        st.success(result["alert_message"])

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=result["probability_fault"] * 100,
            title={"text": "Fault probability (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e24b4a" if result["alert"] else "#1d9e75"},
                "steps": [
                    {"range": [0, 30], "color": "#E1F5EE"},
                    {"range": [30, 50], "color": "#FFF8E1"},
                    {"range": [50, 100], "color": "#FCEBEB"},
                ],
                "threshold": {"line": {"color": "#e24b4a", "width": 3}, "thickness": 0.75, "value": 50},
            },
        )
    )
    gauge.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(gauge, use_container_width=True)

    left, right = st.columns(2)
    with left:
        pressure_df = pd.DataFrame(
            {
                "Sensor": ["TP2", "TP3", "H1", "DV", "Reservoirs"],
                "Value": [
                    payload["TP2"],
                    payload["TP3"],
                    payload["H1"],
                    payload["DV_pressure"],
                    payload["Reservoirs"],
                ],
            }
        )
        fig_pressure = px.bar(pressure_df, x="Sensor", y="Value", title="Pressure readings", color="Value")
        fig_pressure.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_pressure, use_container_width=True)

    with right:
        indicators = pd.DataFrame(
            {
                "Indicator": ["Pressure drop", "Oil temperature", "Motor current", "Temp x Current"],
                "Value": [
                    round(payload["TP3"] - payload["TP2"], 3),
                    payload["Oil_temperature"],
                    payload["Motor_current"],
                    round(payload["Oil_temperature"] * payload["Motor_current"], 3),
                ],
            }
        )
        fig_derived = px.bar(indicators, x="Indicator", y="Value", title="Derived health indicators", color="Value")
        fig_derived.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_derived, use_container_width=True)


def main():
    render_sidebar()
    payload = collect_inputs()
    st.divider()

    if st.button("Run fault prediction", type="primary", use_container_width=True):
        with st.spinner("Analysing sensor readings..."):
            result = call_predict(API_URL, payload)
        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            render_result(result, payload)

    st.divider()
    st.caption("Trained on MetroPT-3 | Powered by FastAPI + Streamlit")


if __name__ == "__main__":
    main()
