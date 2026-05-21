"""Model setup package for Travel-planner."""

from .ai_model import ask_with_failover, get_nvidia_api_keys, make_chat_model, make_openai_client

__all__ = [
    "ask_with_failover",
    "get_nvidia_api_keys",
    "make_chat_model",
    "make_openai_client",
]
