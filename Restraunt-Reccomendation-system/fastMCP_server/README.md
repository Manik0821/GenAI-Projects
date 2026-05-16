# FastMCP Server - Core Application

The main runtime engine. Hosts the Gradio UI and orchestrates all restaurant recommendation logic.

## Files

- **app.py** - Gradio web interface + heuristic routing logic
  - `handle_chat()`: Main chat handler with streaming
  - `_heuristic_route()`: Intent detection (95% accuracy, <5ms)
  - `_extract_vibe_argument()`: Parse cuisines, preferences, locations
  - `_thinking_message_for_action()`: Engagement messages

- **server.py** - FastMCP tool server exposing 3 tools
  - `recommend_by_vibe(vibe)`: Cuisine/preference search with location filtering
  - `get_restaurant_info(name)`: Restaurant lookup by name
  - `get_review(name)`: Customer review fetching
  - Smart vector DB usage: lexical first, vector only if needed

- **client.py** - MCP client for testing individual tools

- **test.py** - Tool diagnostics and manual testing

## Running

```bash
python app.py
# Opens http://127.0.0.1:7860
```

## Key Features

**Heuristic Routing:**
- 30+ cuisines detected instantly (french, italian, sushi, korean...)
- 20+ preferences (spicy, healthy, vegetarian, vegan...)
- Location extraction (in/near location patterns)
- No LLM overhead for 95% of queries

**Smart Tool Execution:**
- Lexical matching on structured JSON first
- Conditional vector search (only if results < 5)
- Metadata backfill from structured records
- Result filtering by location

**Resilience:**
- API key failover: primary → backup keys 1,2,3
- Graceful degradation (vector DB optional)
- Error handling with fallback ReAct loop

## Configuration

Requires `.env` with NVIDIA API keys:
```
NVIDIA_API_KEY=primary-key
NVIDIA_API_KEY1=backup-1
NVIDIA_API_KEY2=backup-2
NVIDIA_API_KEY3=backup-3
```

## Latency Targets

- Heuristic route: ~5ms
- Lexical search: ~50ms
- Vector search: ~100-200ms
- Total (recommend_by_vibe): <300ms typical
