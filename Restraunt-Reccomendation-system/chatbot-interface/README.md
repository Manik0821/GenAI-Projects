# Chatbot-Interface - Legacy UI

Original Gradio-based web interface. **Currently superseded by `fastMCP_server/app.py`**.

## Files

- **chatbot_interface.py** - Legacy Gradio chat UI
  - Web-based restaurant chat interface
  - Basic message history
  - Simple tool calling

- **chatbot_interface_requirements.txt** - Legacy dependencies

## Status

⚠️ **Deprecated**: Use `fastMCP_server/app.py` instead.

The newer implementation offers:
- ✅ Heuristic fast-path (95% queries, <100ms)
- ✅ Better error handling
- ✅ API key failover
- ✅ Location filtering
- ✅ Engaging thinking messages
- ✅ Optimized vector DB integration

## Why This Folder Exists

Kept for reference and potential fallback. Shows evolution from basic chat → optimized fast-path architecture.

## Migration Notes

If reverting to this interface:
```bash
python chatbot_interface.py
# Opens http://127.0.0.1:7860 (same port as app.py)
```

All tools defined in `fastMCP_server/server.py` remain compatible.

## Future

This folder will likely be archived once fast-path system is fully validated in production.
