"""Travel planner model setup with .env API key loading and key failover."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI
from langchain_openai import ChatOpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

# Resolve the workspace-level .env (GenAI-Projects/.env) from this file:
# Travel-planner/Model-setup/ai_model.py -> parents[2] => GenAI-Projects
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def get_nvidia_api_keys() -> list[str]:
    """Return available NVIDIA keys in priority order from .env."""
    return _dedupe(
        [
            os.getenv("NVIDIA_API_KEY", ""),
            os.getenv("NVIDIA_API_KEY1", ""),
            os.getenv("NVIDIA_API_KEY2", ""),
            os.getenv("NVIDIA_API_KEY3", ""),
        ]
    )


def make_openai_client(api_key: str | None = None) -> OpenAI:
    """Build a raw OpenAI-compatible client for NVIDIA inference."""
    key = api_key or (get_nvidia_api_keys()[0] if get_nvidia_api_keys() else "")
    if not key:
        raise ValueError(
            "No NVIDIA API key found. Set NVIDIA_API_KEY/NVIDIA_API_KEY1..3 in .env"
        )
    return OpenAI(api_key=key, base_url=NVIDIA_BASE_URL)


def ask_with_failover(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "You are a helpful travel planning assistant.",
    temperature: float = 0.3,
    max_tokens: int = 400,
) -> str:
    """Run one chat completion and rotate to backup keys if needed."""
    keys = get_nvidia_api_keys()
    if not keys:
        raise ValueError(
            "No NVIDIA API keys found. Add NVIDIA_API_KEY/NVIDIA_API_KEY1..3 in .env"
        )

    errors: list[str] = []
    for idx, key in enumerate(keys, start=1):
        try:
            client = make_openai_client(api_key=key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            code = getattr(exc, "status_code", "unknown")
            errors.append(f"key#{idx} failed (status={code})")
        except Exception as exc:  # pragma: no cover - defensive catch for SDK/runtime issues
            errors.append(f"key#{idx} failed ({type(exc).__name__})")

    raise RuntimeError("All NVIDIA API keys failed: " + "; ".join(errors))


def make_chat_model(
    *, model: str = DEFAULT_MODEL, temperature: float = 0.3, api_key: str | None = None
) -> ChatOpenAI:
    """Return a LangChain chat model configured for NVIDIA endpoint."""
    key = api_key or (get_nvidia_api_keys()[0] if get_nvidia_api_keys() else "")
    if not key:
        raise ValueError(
            "No NVIDIA API key found. Set NVIDIA_API_KEY/NVIDIA_API_KEY1..3 in .env"
        )
    return ChatOpenAI(
        model=model,
        base_url=NVIDIA_BASE_URL,
        api_key=key,
        temperature=temperature,
    )


if __name__ == "__main__":
    # Lightweight smoke test only when run directly.
    try:
        reply = ask_with_failover("Suggest a 2-day weekend trip near Bangalore.")
        print(reply)
    except Exception as exc:
        print("Model setup check failed:", exc)
