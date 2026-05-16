import os
from dotenv import load_dotenv

load_dotenv()


def get_nvidia_api_key() -> str:
    key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NVIDIA_API_KEY1")
        or os.getenv("NVIDIA_API_KEY2")
        or os.getenv("NVIDIA_API_KEY3")
        or ""
    )
    if not key:
        raise ValueError("Missing NVIDIA API key. Set NVIDIA_API_KEY in your .env file.")
    return key


NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")