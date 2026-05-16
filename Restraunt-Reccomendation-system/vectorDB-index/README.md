# VectorDB-Index - Semantic Search

Scripts for building and managing the Chroma vector database for semantic restaurant searches.

## Files

- **multimodal_vector_index.py** - Vector DB initialization
  - Embeds restaurant descriptions using SentenceTransformer (384-d)
  - Embeds restaurant images using CLIP (512-d)
  - Stores embeddings in Chroma persistent DB
  - Enables semantic search alongside lexical matching

- **structured_restaurant_data.json** - Input data
  - Restaurant records with descriptions and metadata
  - Used for embedding and indexing

## How It Works

**Text Embeddings** (SentenceTransformer all-MiniLM-L6-v2)
- Converts restaurant descriptions → 384-dimensional vectors
- Captures semantic meaning for restaurant search
- Enables queries like "cozy Italian spot" → find semantically similar restaurants

**Image Embeddings** (CLIP - Contrastive Learning Image-Text)
- Converts recipe/restaurant images → 512-dimensional vectors
- Bridges vision and text understanding
- Supports image-based discovery (future feature)

**Chroma Database**
- Persistent storage under `chroma_db/`
- Collection name: "restaurants"
- Supports similarity search and filtering
- Auto-generates metadata (location, cuisine tags)

## Usage

```bash
python multimodal_vector_index.py
# Reads structured_restaurant_data.json
# Builds embeddings and stores in chroma_db/
# Creates persistent searchable index
```

## Integration with App

The `recommend_by_vibe()` tool in `fastMCP_server/server.py`:
1. Tries **lexical matching** first (fast, <50ms)
2. If results < 5, queries **vector DB** (semantic, <200ms)
3. Combines both for diverse recommendations

## Vector DB Files

- `chroma.sqlite3` - Metadata index
- `.bin` files - Vector data (generated, not committed to Git)
- Auto-created on first run

## Notes

- Vector DB is optional (app works without it)
- Embeddings cached after first build
- Rebuild only needed when restaurant data changes
- Local model download happens on first run (~350MB)
