from openai import OpenAI
from langchain_openai import ChatOpenAI

from app.config import get_nvidia_api_key, NVIDIA_BASE_URL, NVIDIA_MODEL


def get_openai_compatible_client() -> OpenAI:
    return OpenAI(
        api_key=get_nvidia_api_key(),
        base_url=NVIDIA_BASE_URL,
    )


def ask_llm(prompt: str) -> str:
    client = get_openai_compatible_client()
    response = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for health and wellness product analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=300,
    )
    return response.choices[0].message.content or ""


def make_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=NVIDIA_MODEL,
        base_url=NVIDIA_BASE_URL,
        api_key=get_nvidia_api_key(),
        temperature=0.7,
    )