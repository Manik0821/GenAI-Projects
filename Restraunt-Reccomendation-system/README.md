# Restaurant Recommendation System

This project is a modular GenAI-powered restaurant and recipe recommendation system. Each folder is a self-contained component or service. Below is an overview of each folder and its purpose:

## Folders & Contents

- **chatbot-interface/**
  - Gradio-based web UI for user interaction.
  - Requirements: `chatbot_interface_requirements.txt`

- **fastMCP_server/**
  - Main backend server, agent orchestration, and tool protocol (MCP).
  - Key files: `app.py` (Gradio app), `server.py` (MCP tools), `client.py`, `test.py` (diagnostics).

- **Loading-Data/**
  - Scripts and data for loading and preprocessing raw restaurant/recipe data.
  - Example: `loading-data.py`, `California-Culinary-Map.txt`

- **Managing-data/**
  - Scripts for managing and updating structured restaurant data.
  - Example: `restraunt_data_management.py`

- **Model-setup/**
  - LLM and embedding model setup scripts.
  - Example: `llama-model.py`

- **processing-data/**
  - Data processing and feature extraction scripts.
  - Example: `process-data.py`, `review_image_placeholder.jpg`

- **recipe_images/**
  - Directory for storing recipe and synthetic images.

- **specialized-agents/**
  - Specialized agent definitions and multi-agent system logic.
  - Requirements: `specialized-agents-requirement.txt`

- **vectorDB-index/**
  - Scripts for building and managing the Chroma vector database.
  - Example: `multimodal-vector-index.py`, `structured_restaurant_data.json`

## Root Files
- `requirements.txt`: Core dependencies for the main system.
- `Recipes.json`, `augmented-user-review.json`, `structured_restaurant_data.json`, `Synthetic-User-Reviews.json`: Data files.

## Setup
- Install dependencies for each module using the requirements file in its folder.
- Use Python 3.12+ and create a virtual environment in the project root.
- See each folder's requirements file for specific dependencies.

## New Runtime Workflow (Fast Path)
The current app workflow in `fastMCP_server/app.py` is optimized for low latency:

1. **Specialized intent routing first**
  - The request is routed to a specialized action:
  - `recommend_by_vibe`
  - `get_restaurant_info`
  - `get_review`

2. **Direct MCP tool call**
  - The selected MCP tool is called directly from `fastMCP_server/server.py`.
  - Tool output is formatted and returned quickly.

3. **Fallback ReAct only when needed**
  - If routing is unclear, a reduced-turn ReAct loop is used as fallback.

4. **Automatic API key failover**
  - Key order: `NVIDIA_API_KEY`, `NVIDIA_API_KEY1`, `NVIDIA_API_KEY2`, `NVIDIA_API_KEY3`.
  - If one key fails (auth/quota/rate issues), the app retries with backup keys.

## Notes
- The `.gitignore` is configured to exclude virtual environments, cache, large data, and generated files.
- For more details, see comments in each script or requirements file.

## Git Guidance: What To Commit
- **Commit**:
  - Source code (`*.py`), docs (`README.md`), dependency files (`requirements*.txt`), and small config files.

- **Do not commit**:
  - Generated vector DB files under `chroma_db/` (including `.bin`, `.sqlite3`, and other index artifacts).
  - Virtual environments, cache files, and large generated assets.

Chroma `.bin` files are internal index structures used by the vector database runtime. They are generated artifacts, can be recreated, and should stay out of Git.
