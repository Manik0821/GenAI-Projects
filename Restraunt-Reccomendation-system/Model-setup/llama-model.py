import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY2"),   # your nvapi- key
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

try:
    print(ask_llm("Suggest a restaurant in Bangalore"))
except Exception as e:
    print("Error:", e)