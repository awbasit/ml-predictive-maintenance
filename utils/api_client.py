"""Reusable HTTP client helpers for the FastAPI service."""

from __future__ import annotations

import requests


def check_api_health(api_url: str) -> dict | None:
    """Call /health and return parsed JSON when available."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def call_predict(api_url: str, payload: dict) -> dict:
    """Call /predict and return API JSON or an error payload."""
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as exc:
        return {"error": str(exc)}


def get_metrics(api_url: str) -> dict | None:
    """Fetch model evaluation results from /metrics."""
    try:
        response = requests.get(f"{api_url}/metrics", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def get_feature_importance(api_url: str) -> dict | None:
    """Fetch feature importances from /feature-importance."""
    try:
        response = requests.get(f"{api_url}/feature-importance", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def get_prediction_history(api_url: str) -> dict | None:
    """Fetch server-side prediction history from /history."""
    try:
        response = requests.get(f"{api_url}/history", timeout=5)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None
