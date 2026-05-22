"""Image-based place/landmark recognition using NVIDIA vision models."""

from __future__ import annotations

import ast
import base64
import os
from pathlib import Path

import httpx
import urllib3
from dotenv import load_dotenv
from openai import OpenAI

# Load workspace-level .env
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"

_SSL_VERIFY: bool = (
    os.getenv("REQUESTS_VERIFY_SSL", "true").strip().lower() != "false"
)

if not _SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_MEDIA_TYPE_MAP: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_PLACE_PROMPT = (
    "Identify the city, country, or famous landmark visible in this image. "
    "Reply with ONLY the place name — for example: 'Mumbai, India', "
    "'Gateway of India, Mumbai', 'Eiffel Tower, Paris, France'. "
    "If the location cannot be determined, reply: 'Unknown'."
)


def _get_nvidia_keys() -> list[str]:
    candidates = [
        os.getenv("NVIDIA_API_KEY", ""),
        os.getenv("NVIDIA_API_KEY1", ""),
        os.getenv("NVIDIA_API_KEY2", ""),
        os.getenv("NVIDIA_API_KEY3", ""),
    ]
    seen: set[str] = set()
    result: list[str] = []
    for k in candidates:
        if k and k not in seen:
            seen.add(k)
            result.append(k)
    return result


def _make_openai_client(api_key: str) -> OpenAI:
    """Build an OpenAI client, respecting the SSL setting from .env."""
    if not _SSL_VERIFY:
        http_client = httpx.Client(verify=False)
        return OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL, http_client=http_client)
    return OpenAI(api_key=api_key, base_url=NVIDIA_BASE_URL)


def _local_file_to_base64(image_path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    ext = Path(image_path).suffix.lower()
    media_type = _MEDIA_TYPE_MAP.get(ext, "image/jpeg")
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8"), media_type


def _normalized_image_source(image_source: str | dict[str, object]) -> str:
    """Coerce Gradio file payloads, URLs, data URIs, and local paths to one string."""
    if isinstance(image_source, dict):
        return str(image_source.get("path") or image_source.get("url") or "").strip()

    source = str(image_source or "").strip()
    if not source:
        return ""

    if source.startswith("{") and source.endswith("}"):
        try:
            payload = ast.literal_eval(source)
        except (SyntaxError, ValueError):
            payload = None
        if isinstance(payload, dict):
            return str(payload.get("path") or payload.get("url") or source).strip()

    return source


def _image_block_from_source(image_source: str | dict[str, object]) -> dict[str, dict[str, str] | str]:
    """Build the image content block for URLs, data URIs, or local files."""
    source = _normalized_image_source(image_source)
    if source.startswith("data:"):
        return {"type": "image_url", "image_url": {"url": source}}
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}

    b64_data, media_type = _local_file_to_base64(source)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
    }


def recognize_place_from_image(
    image_source: str | dict[str, object],
    *,
    model: str = VISION_MODEL,
) -> str:
    """Identify the city or landmark shown in an image.

    Args:
        image_source: Local file path, data URI, OR a public URL to the image.
                      Supported formats: JPEG, PNG, GIF, WEBP.
                      URLs are passed directly to the NVIDIA API — the API
                      must be able to reach the URL from its own network.
                  Local files and data URIs are base64-encoded or passed
                  inline before sending.
        model: NVIDIA vision model identifier. Default is Llama 3.2 11B Vision.

    Returns:
        Place name string, e.g. ``"Gateway of India, Mumbai, India"``.

    Raises:
        RuntimeError: if all available API keys fail.
        ValueError: if no API keys are configured in .env.
    """
    image_block = _image_block_from_source(image_source)
    messages = [
        {
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": _PLACE_PROMPT},
            ],
        }
    ]

    keys = _get_nvidia_keys()
    if not keys:
        raise ValueError(
            "No NVIDIA API keys found. Add NVIDIA_API_KEY/NVIDIA_API_KEY1..3 to .env"
        )

    errors: list[str] = []
    for idx, key in enumerate(keys, start=1):
        try:
            client = _make_openai_client(key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0.1,
            )
            return (resp.choices[0].message.content or "Unknown").strip()
        except Exception as exc:
            errors.append(f"key#{idx} ({type(exc).__name__}): {exc}")

    raise RuntimeError("All NVIDIA API keys failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    import sys

    # Usage: python image_recognizer.py <local_image_path>
    # Example: python image_recognizer.py C:\photos\mumbai.jpg
    if len(sys.argv) < 2:
        print("Usage: python image_recognizer.py <local_image_path>")
        print("Example: python image_recognizer.py C:\\photos\\gateway_of_india.jpg")
        sys.exit(0)

    source = sys.argv[1]
    print(f"Image : {source}")
    print("Recognizing place...")
    place = recognize_place_from_image(source)
    print(f"Result: {place}")
