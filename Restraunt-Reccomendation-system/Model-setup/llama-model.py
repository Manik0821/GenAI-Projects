import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def _get_nvidia_api_key() -> str:
    return (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NVIDIA_API_KEY1")
        or os.getenv("NVIDIA_API_KEY2")
        or os.getenv("NVIDIA_API_KEY3")
        or ""
    )

client = OpenAI(
    api_key=_get_nvidia_api_key(),
    base_url="https://integrate.api.nvidia.com/v1"
)

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful food recommendation assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content

def make_model():
    return ChatOpenAI(
        model="meta/llama-3.1-70b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=_get_nvidia_api_key(),
        temperature=0.7,
    )

try:
    print(ask_llm("Suggest a restaurant in Bangalore"))
except Exception as e:
    print("Error:", e)