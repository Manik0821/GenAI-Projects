"""Vercel ASGI entrypoint for the Travel Planner app."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
TRAVEL_APP_PATH = ROOT / "Travel-planner" / "app.py"


def _load_travel_app_module():
    spec = importlib.util.spec_from_file_location("travel_planner_app", TRAVEL_APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Travel Planner app from {TRAVEL_APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


travel_app = _load_travel_app_module()
fastapi_app = FastAPI(title="Travel Planner")
app = gr.mount_gradio_app(
    fastapi_app,
    travel_app.demo,
    path="/",
    show_error=True,
)
