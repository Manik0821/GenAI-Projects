# Connoisseur Companion - Restaurant Recommendation System

An intelligent, **low-latency restaurant discovery engine** powered by AI. Users ask natural language questions about restaurants, and the system instantly delivers personalized recommendations from California's culinary landscape.

## ⚡ Key Features

- **95% Fast-Path**: Most queries answered in <100ms without LLM calls
- **Cuisine Detection**: 30+ cuisines (french, italian, sushi, korean, etc.)
- **Preference Matching**: spicy, healthy, vegan, gluten-free, and more
- **Location Filtering**: "near pasadena" or "in santa monica" 
- **API Key Failover**: Automatic backup if primary key fails
- **Streaming UI**: Real-time thinking messages keep users engaged

## 🚀 Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate on Windows
pip install -r requirements.txt

# Run
cd Restraunt-Reccomendation-system
python fastMCP_server/app.py
# Open http://127.0.0.1:7860
```

## 📁 Project Structure

| Folder | Purpose |
|--------|---------|
| **fastMCP_server/** | Core app: `app.py` (UI), `server.py` (tools), `client.py` (MCP client) |
| **Loading-Data/** | Load & preprocess raw restaurant data |
| **Managing-data/** | Manage structured restaurant metadata |
| **Model-setup/** | LLM & embedding model configuration |
| **processing-data/** | Feature extraction & data enrichment |
| **specialized-agents/** | Multi-agent definitions (future expansion) |
| **vectorDB-index/** | Chroma vector DB setup & semantic search |
| **chatbot-interface/** | Legacy Gradio UI (superseded by app.py) |

## 🧠 How It Works

### User Query Flow
```
User: "I want italian dishes near pasadena"
  ↓
[Heuristic Router] Detects: Cuisine="italian", Location="pasadena"
  ↓
[Fast-Path] Lexical search → 2 matches found instantly
  ↓
[Conditional Vector Search] Need more? Query Chroma DB → +1 semantic match
  ↓
[Format & Stream] Return formatted results with thinking messages
```

### Three Core Tools

| Tool | Use Case | Speed |
|------|----------|-------|
| `recommend_by_vibe` | Cuisine/preference searches + location filter | <100ms |
| `get_restaurant_info` | Restaurant details by name | <50ms |
| `get_review` | Customer reviews & ratings | <50ms |

## 🎯 Supported Queries

✅ `"french restaurant"` → Cuisine search  
✅ `"spicy food in dtla"` → Preference + location  
✅ `"tell me about iron & embers"` → Restaurant details  
✅ `"reviews for sakura garden"` → Customer feedback  
✅ Complex/ambiguous → Fallback ReAct loop (LLM)  

## 📊 Tech Stack

- **LLM**: NVIDIA Llama 3.1 70B (https://integrate.api.nvidia.com/v1)
- **Vector DB**: Chroma (text: SentenceTransformer 384-d, images: CLIP 512-d)
- **UI Framework**: Gradio
- **Tool Protocol**: FastMCP
- **Language**: Python 3.12+

## ⚙️ Configuration

### Environment Variables (.env)
```
NVIDIA_API_KEY=your-primary-key
NVIDIA_API_KEY1=backup-key-1
NVIDIA_API_KEY2=backup-key-2
NVIDIA_API_KEY3=backup-key-3
```

### Key Optimizations
- **Lexical-first**: Check structured data before vector search
- **Conditional vector DB**: Only query embeddings if <5 structured results
- **Location filtering**: Precise geographic relevance
- **API failover**: Seamless backup key rotation

## 📝 Folder Details

Each folder has its own `README.md` describing:
- What the module does
- Key files and their roles
- Dependencies
- Example usage

See individual folder READMEs for deep dives.

## 🔧 Development

### Running Diagnostics
```bash
python mcp_diag_a.py  # MCP latency test
python mcp_diag_b.py  # Agent latency test
```

### Testing Specific Tools
```bash
python fastMCP_server/test.py
```

## 📚 Notes
- The `.gitignore` excludes venv, cache, and Chroma DB artifacts
- Always test after modifying heuristic routing rules
- Update docstrings when adding new cuisine/preference keywords
- Keep thinking messages under 100 characters for UI clarity
