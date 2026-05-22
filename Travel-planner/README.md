# Travel Planner

Travel Planner is a Gradio-based travel discovery app that lets you search for a destination by name or detect a place from an uploaded image, then explore nearby attractions, museums, restaurants, hotels, and parks.

## Features

- Search destinations by place name
- Detect a destination from an uploaded image
- Show current location information in separate Image, Details, and Map tabs
- Fetch nearby points of interest with Geoapify
- Use NVIDIA-hosted models with API-key failover for image-based place recognition

## Project Structure

- `app.py` - Gradio UI entrypoint
- `data-fetcher/places_fetcher.py` - Geoapify geocoding and nearby-place lookup
- `data-fetcher/image_recognizer.py` - image-to-place recognition flow
- `Model-setup/ai_model.py` - NVIDIA/OpenAI-compatible model client helpers
- `travel-1.jpg`, `travel-2.avif`, `travel-3.webp` - carousel assets

## Requirements

- Python 3.10+
- A Geoapify API key
- One or more NVIDIA API keys for image recognition

## Environment Variables

Create a `.env` file in the workspace root (`GenAI-Projects/.env`) with the values you need:

```env
GEOAPIFY_API_KEY=your_geoapify_key
NVIDIA_API_KEY1=your_primary_nvidia_key
NVIDIA_API_KEY2=your_backup_nvidia_key
NVIDIA_API_KEY3=your_backup_nvidia_key
REQUESTS_VERIFY_SSL=false
```

Notes:

- `REQUESTS_VERIFY_SSL=false` is optional and is only useful when your network intercepts SSL traffic.
- The app reads `.env` from the workspace root, not from inside `Travel-planner/`.

## Setup

Install dependencies from the workspace root:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r Travel-planner\Model-setup\requirements.txt
.venv\Scripts\pip install -r Travel-planner\data-fetcher\requirements.txt
.venv\Scripts\pip install gradio httpx
```

## Run

Start the app from the workspace root:

```powershell
.venv\Scripts\python.exe Travel-planner\app.py
```

The app runs on `http://127.0.0.1:7860` by default.

## Deploy To Vercel

This repository now includes Vercel deployment files at the workspace root:

- `api/index.py` - ASGI entrypoint that mounts the Gradio app
- `vercel.json` - routes all requests to the Python function
- `requirements.txt` - runtime dependencies for the deployed app
- `.vercelignore` - excludes unrelated projects and local artifacts from upload

Before deploying, add these environment variables in the Vercel project settings:

- `GEOAPIFY_API_KEY`
- `NVIDIA_API_KEY1`
- `NVIDIA_API_KEY2`
- `NVIDIA_API_KEY3`
- `REQUESTS_VERIFY_SSL`

Deployment notes:

- Keep the Vercel project root at the repository root, not inside `Travel-planner/`.
- The deployed app reads environment variables from Vercel project settings; it does not require a checked-in `.env` file.
- The Vercel entrypoint imports `Travel-planner/app.py` and serves the existing Gradio UI through FastAPI.

## What The UI Shows

- A top carousel with travel imagery
- A place search input and explore button
- An optional image-upload flow for detecting a place
- Current location tabs for image, details, and map
- Nearby results grouped into category tabs