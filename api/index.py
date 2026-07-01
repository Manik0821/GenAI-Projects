"""Vercel ASGI entrypoint for the Gradio apps."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

ROOT = Path(__file__).resolve().parents[1]
TRAVEL_APP_PATH = ROOT / "Travel-planner" / "app.py"
RESTRAUNT_APP_PATH = ROOT / "Restraunt-Reccomendation-system" / "fastMCP_server" / "app.py"
os.environ.setdefault("TRAVEL_APP_ROUTE_PREFIX", "/travel")

if os.getenv("VERCEL"):
    # Keep the restaurant recommender serverless-safe by disabling the local Chroma path.
    os.environ.setdefault("RESTAURANT_ENABLE_CHROMA", "0")
    os.environ.setdefault("RESTAURANT_DIRECT_TOOLS", "1")


def _load_app_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


travel_app = _load_app_module("travel_planer_app", TRAVEL_APP_PATH)
restraunt_app = _load_app_module("restraunt_reccomendation_app", RESTRAUNT_APP_PATH)
fastapi_app = FastAPI(title="GenAI Projects")


@fastapi_app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/travel")


@fastapi_app.api_route("/gradio_api/{path:path}", methods=["GET", "HEAD"])
def gradio_file_redirect(path: str):
    return RedirectResponse(url=f"/travel/gradio_api/{path}", status_code=307)


app = gr.mount_gradio_app(
    fastapi_app,
    travel_app.demo,
    path="/travel",
    allowed_paths=[str(TRAVEL_APP_PATH.parent)],
    show_error=True,
)

app = gr.mount_gradio_app(
    app,
    restraunt_app.demo,
    path="/restraunt",
    allowed_paths=[str(RESTRAUNT_APP_PATH.parent)],
    show_error=True,
)