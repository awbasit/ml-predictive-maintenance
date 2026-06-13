---
title: Compressor Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: dashboard/app.py
pinned: false
---

# Compressor Predictive Maintenance — Dashboard

Streamlit dashboard for the MetroPT-3 predictive maintenance system.

## Environment variables

| Variable | Description | Example |
|---|---|---|
| `API_URL` | Full URL of the deployed FastAPI service | `https://your-api.railway.app` |

Set this as a **Space secret** in the HuggingFace Space settings.  
If `API_URL` is not set the dashboard falls back to `http://localhost:8000`.
