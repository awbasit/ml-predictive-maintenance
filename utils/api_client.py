"""Reusable HTTP client helpers for the FastAPI service."""

from __future__ import annotations

import requests


def check_api_health(api_url: str):
    """Call /health and return parsed JSON when available."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def call_predict(api_url: str, payload: dict):
    """Call /predict and return API JSON or an error payload."""
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as exc:
        return {"error": str(exc)}

