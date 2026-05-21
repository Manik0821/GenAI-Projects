# Travel Planner Model Setup

This module loads API keys from the workspace `.env` file and creates a ready-to-use NVIDIA-hosted Llama model client.

## Environment Variables

Add these keys in `GenAI-Projects/.env`:

- `NVIDIA_API_KEY` (optional primary)
- `NVIDIA_API_KEY1` (backup 1)
- `NVIDIA_API_KEY2` (backup 2)
- `NVIDIA_API_KEY3` (backup 3)

The code uses available keys in order and falls back automatically if one key fails.

## Files

- `ai_model.py`: Client/model builders and failover logic
- `requirements.txt`: Minimal dependencies for model setup

## Usage

```python
from ai_model import ask_with_failover, make_chat_model

text = ask_with_failover("Plan a 3-day Goa itinerary")
print(text)

llm = make_chat_model(temperature=0.2)
```

## Install

```bash
pip install -r requirements.txt
```
