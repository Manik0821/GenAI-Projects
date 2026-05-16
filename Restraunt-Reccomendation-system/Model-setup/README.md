# Model-Setup - LLM Configuration

Scripts for setting up and configuring the language model and embedding models.

## Files

- **llama_model.py** - NVIDIA Llama 3.1 70B setup
  - Configures base URL: https://integrate.api.nvidia.com/v1
  - Model ID: meta/llama-3.1-70b-instruct
  - Handles API authentication
  - Sets temperature and other parameters
  - Implements key failover logic

## Models Used

**LLM - NVIDIA Llama 3.1 70B**
- Purpose: Intent routing, ReAct reasoning, fallback responses
- Used: 5% of requests (heuristic handles 95%)
- Temperature: 0.0 for routing, 0.2-0.3 for responses
- Context window: 8K tokens
- Latency: ~1-2 seconds per call

**Text Embeddings - SentenceTransformer**
- Model: all-MiniLM-L6-v2
- Dimension: 384-d
- Purpose: Restaurant description embeddings
- Used in: Vector DB indexing and semantic search
- Local execution (no API calls)

**Image Embeddings - CLIP**
- Model: openai/clip-vit-base-patch32
- Dimension: 512-d
- Purpose: Restaurant & recipe image embeddings
- Used in: Multimodal vector search
- Local execution

## Configuration

Requires environment variables:
```
NVIDIA_API_KEY=primary-key
NVIDIA_API_KEY1=backup-key-1
NVIDIA_API_KEY2=backup-key-2
NVIDIA_API_KEY3=backup-key-3
```

## Usage

```bash
python llama_model.py
# Validates API keys
# Tests connectivity to NVIDIA inference endpoint
# Confirms model availability
```

## Key Failover Logic

If primary key fails with:
- 401 (Unauthorized)
- 403 (Forbidden)
- 429 (Rate Limited)
- Quota exceeded

→ Automatically try next key in sequence

## Notes

- Embeddings models run locally (no network calls)
- LLM calls only for intent routing (not critical path)
- Keys can be rotated without restart
- Graceful degradation if all keys fail
