"""FastMCP server exposing low-latency restaurant tools.

Tools:
- get_restaurant_info: fuzzy name lookup in structured data.
- recommend_by_vibe: preference/vibe/taste matching with optional vector recall.
- get_review: fetches review text and metadata by restaurant name.
"""

# Libraries to import to create our MCP server and handle data loading
from fastmcp import FastMCP
from pathlib import Path
import json
import re
import hashlib
import sys

import numpy as np

try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Initializing our MCP server instance
mcp = FastMCP("Connoisseur-Server")

# Data paths
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


CULINARY_MAP_PATH = _first_existing(
    [
        DATA_DIR / "California-Culinary-Map.txt",
        PROJECT_ROOT / "Loading-Data" / "California-Culinary-Map.txt",
        PROJECT_ROOT / "California-Culinary-Map.txt",
    ]
)

RESTAURANT_DATA_PATH = _first_existing(
    [
        DATA_DIR / "structured_restaurant_data.json",
        DATA_DIR / "structured-restaurant-data.json",
        PROJECT_ROOT / "structured_restaurant_data.json",
    ]
)

REVIEW_DATA_PATH = _first_existing(
    [
        DATA_DIR / "augmented_user_review.json",
        DATA_DIR / "augmented-user-review.json",
        PROJECT_ROOT / "processing-data" / "augmented_user_review.json",
        PROJECT_ROOT / "augmented_user_review.json",
    ]
)

CHROMA_DB_PATH = _first_existing(
    [
        PROJECT_ROOT / "chroma_db",
        PROJECT_ROOT / "vectorDB-index" / "chroma_db",
    ]
)

_ARTICLE_DB = None
_TEXT_MODEL = None
_TEXT_MODEL_READY = False


def _debug_log(message: str) -> None:
    """Write server diagnostics to stderr so stdout stays JSON-RPC only."""
    print(message, file=sys.stderr, flush=True)


def _deterministic_query_vec(value: str, dim: int = 384) -> np.ndarray:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def _get_text_query_vector(query: str) -> np.ndarray:
    global _TEXT_MODEL, _TEXT_MODEL_READY

    if SentenceTransformer is None:
        return _deterministic_query_vec(query)

    if not _TEXT_MODEL_READY:
        try:
            # Keep request latency predictable: do not trigger network model downloads.
            _TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            _TEXT_MODEL_READY = True
        except Exception:
            _TEXT_MODEL = None
            _TEXT_MODEL_READY = True

    if _TEXT_MODEL is None:
        return _deterministic_query_vec(query)

    vec = _TEXT_MODEL.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    return vec


def _get_article_db():
    """Lazily initialize and cache the Chroma restaurant collection handle."""
    global _ARTICLE_DB
    if _ARTICLE_DB is not None:
        return _ARTICLE_DB

    if Chroma is None or CHROMA_DB_PATH is None:
        return None

    try:
        _ARTICLE_DB = Chroma(
            collection_name="restaurants",
            persist_directory=str(CHROMA_DB_PATH),
        )
        return _ARTICLE_DB
    except Exception:
        return None


def _extract_name_from_doc(doc: str) -> str:
    """Extract a restaurant name from a vector document payload."""
    m = re.search(r"Restaurant:\s*(.+)", doc)
    return m.group(1).strip() if m else "Unknown"


def _is_unknown(value) -> bool:
    """Return True when a field contains a placeholder/empty value."""
    text = str(value or "").strip().lower()
    return text in {"", "unknown", "n/a", "none", "null"}


def _normalize_name(value: str) -> str:
    """Normalize names for resilient dictionary-based joining."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


_CUISINE_HINTS = (
    "italian",
    "japanese",
    "kaiseki",
    "sushi",
    "ramen",
    "szechuan",
    "chinese",
    "korean",
    "mexican",
    "oaxacan",
    "indian",
    "persian",
    "mediterranean",
    "vietnamese",
    "french",
    "american",
    "seafood",
    "lebanese",
    "tapas",
    "bbq",
    "barbecue",
    "vegan",
    "vegetarian",
    "fusion",
)


def _extract_cuisine_from_description(description: str) -> str | None:
    if not description:
        return None

    patterns = [
        r"specializing in\s+\*\*(.*?)\*\*",
        r"offers\s+an\s+\*\*(.*?)\*\*\s+menu",
        r"offers\s+an\s+\*\*(.*?)\*\*\s+experience",
    ]
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(" ,.-")
            if candidate and not _is_unknown(candidate):
                return candidate
    return None


def _infer_cuisine(restaurant: dict) -> str:
    """Infer cuisine from explicit fields, vibe tags, then description patterns."""
    for key in ("cuisine", "food_style", "type"):
        candidate = str(restaurant.get(key, "")).strip()
        if candidate and not _is_unknown(candidate):
            return candidate

    vibes = [str(v).strip() for v in restaurant.get("vibes", []) if str(v).strip()]
    for vibe in vibes:
        lower_vibe = vibe.lower()
        if any(h in lower_vibe for h in _CUISINE_HINTS):
            return vibe

    from_description = _extract_cuisine_from_description(str(restaurant.get("description", "")))
    if from_description:
        return from_description

    return "Unknown"


def _parse_price_dollar_count(text: str) -> int | None:
    m = re.search(r"Price range:\s*(\$+)", text, re.IGNORECASE)
    if not m:
        return None
    return len(m.group(1))


def _parse_rating(text: str) -> float | None:
    m = re.search(r"(\d(?:\.\d)?)/5", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _fallback_restaurants_from_map() -> list[dict]:
    """Build minimal structured rows from the raw culinary map when JSON is missing."""
    if not CULINARY_MAP_PATH:
        return []

    raw = CULINARY_MAP_PATH.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

    rows = []
    for para in paragraphs:
        # Skip title lines
        if para.startswith("### "):
            continue

        bold_chunks = re.findall(r"\*\*(.*?)\*\*", para)
        if not bold_chunks:
            continue

        name = bold_chunks[0].strip()
        if not name:
            continue

        location_match = re.search(r"\bin\s+\*\*(.*?)\*\*", para)
        location = location_match.group(1).strip() if location_match else "Unknown"

        # Keep broad tags for vibe matching. They may include cuisine/type labels too.
        tags = [b.strip() for b in bold_chunks[1:] if b and "/5" not in b and "$" not in b]
        inferred = _infer_cuisine({"vibes": tags, "description": para})

        rows.append(
            {
                "name": name,
                "neighborhood": location,
                "location": location,
                "cuisine": inferred,
                "food_style": inferred,
                "rating": _parse_rating(para),
                "vibes": tags,
                "price_range": _parse_price_dollar_count(para),
                "description": para,
            }
        )

    return rows

# Helper functions
def load_restaurant_data() -> list[dict]:
    """Load restaurant records and backfill missing cuisine-like fields."""
    if RESTAURANT_DATA_PATH and RESTAURANT_DATA_PATH.exists():
        with open(RESTAURANT_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return []

            # Enrich records with inferred cuisine when source JSON has placeholder values.
            for row in data:
                inferred = _infer_cuisine(row)
                if _is_unknown(row.get("cuisine")):
                    row["cuisine"] = inferred
                if _is_unknown(row.get("food_style")):
                    row["food_style"] = inferred
            return data

    # Fallback keeps app usable when step outputs are not generated yet.
    return _fallback_restaurants_from_map()

def load_review_data() -> list[dict]:
    """Load review rows from known filenames if present."""
    if not REVIEW_DATA_PATH or not REVIEW_DATA_PATH.exists():
        return []
    with open(REVIEW_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else []

# MCP Resource - Exposing the Raw Culinary Map data
@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """The full raw California Culinary Map text from Module 1.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    if not CULINARY_MAP_PATH:
        return "Culinary map file not found. Expected California-Culinary-Map.txt in Loading-Data or fastMCP_server."
    return CULINARY_MAP_PATH.read_text(encoding="utf-8")

# TOOL 1 — Get Restaurant Info (Structured Search)
@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a restaurant by name and return its structured details
    including cuisine, rating, price range, and signature dish."""
    _debug_log(f"[*] Looking up restaurant: {restaurant_name}")
    restaurants = load_restaurant_data()
    _debug_log(f"[*] Loaded {len(restaurants)} restaurants from database")
    query = restaurant_name.lower().strip()

    # Finding restuarants that match the query in the structured JSON data
    matches = []
    for restaurant in restaurants:
        name = str(restaurant.get("name", "")).lower()
        if query in name or name in query:
            matches.append(restaurant)

    _debug_log(f"[*] Found {len(matches)} matching restaurants")

    # Return a not found message if no matches are found
    if not matches:
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No restaurant found matching '{restaurant_name}'.",
                "suggestion": "Try a partial name like 'Iron' or 'Sakura'.",
            },
            indent=2,
        )

    _debug_log(f"[*] Formatting results for {len(matches)} restaurants")
    return json.dumps(
        {"status": "found", "count": len(matches), "results": matches},
        indent=2,
    )
# TOOL 2 — Recommend by Vibe (Semantic Search)
@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """Return matches for vibe/taste/preference text with latency-aware ranking.

    Strategy:
    1. Run fast structured lexical matching over vibe tags/description.
    2. Use vector recall only when structured matches are sparse.
    3. Backfill vector metadata from structured rows for cleaner output.
    4. Filter results by location if specified in the vibe string.
    """
    _debug_log(f"[*] Searching for '{vibe}' vibe recommendations")
    restaurants = load_restaurant_data()
    _debug_log(f"[*] Loaded {len(restaurants)} restaurants, running fast lexical matching...")
    vibe_lower = vibe.lower().strip()

    # Extract location filter if present (e.g., "italian in pasadena" or "italian near pasadena")
    location_filter = ""
    location_match = re.search(r"\bin\s+([a-z][a-z\s\-()]{1,40})$", vibe_lower)
    if location_match:
        location_filter = location_match.group(1).strip()
        vibe_lower = vibe_lower[:location_match.start()].strip()
        _debug_log(f"[*] Location filter detected: {location_filter}")

    # Pass 1: Search structured vibe tags in JSON (fast lexical path)
    structured_matches = []
    for restaurant in restaurants:
        vibes_list = [v.lower() for v in restaurant.get("vibes", [])]
        description = restaurant.get("description", "").lower()

        if any(vibe_lower in v for v in vibes_list) or vibe_lower in description:
            # Apply location filter if specified
            if location_filter:
                neighborhood = (restaurant.get("neighborhood") or restaurant.get("location", "")).lower()
                if location_filter not in neighborhood:
                    continue

            structured_matches.append(
                {
                    "name": restaurant.get("name", "Unknown"),
                    "neighborhood": restaurant.get("neighborhood") or restaurant.get("location", "Unknown"),
                    "cuisine": _infer_cuisine(restaurant),
                    "rating": restaurant.get("rating", "N/A"),
                    "vibes": restaurant["vibes"],
                    "price_range": restaurant.get("price_range", "N/A"),
                }
            )

    _debug_log(f"[*] Structured matches found: {len(structured_matches)}")

    # Vector recall only when lexical pass is sparse.
    vector_matches = []
    article_db = _get_article_db() if len(structured_matches) < 5 else None
    if article_db is not None:
        _debug_log(f"[*] Structured results sparse, querying vector database for semantic matches...")
        try:
            qvec = _get_text_query_vector(vibe_lower)
            queried = article_db._collection.query(
                query_embeddings=[qvec.tolist()],
                n_results=8,
                include=["documents", "metadatas", "distances"],
            )

            docs = queried.get("documents", [[]])[0]
            metas = queried.get("metadatas", [[]])[0]
            dists = queried.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                # Apply location filter to vector results as well
                if location_filter:
                    vector_location = ((meta or {}).get("location", "")).lower()
                    if location_filter not in vector_location:
                        continue

                vector_matches.append(
                    {
                        "name": _extract_name_from_doc(doc or ""),
                        "neighborhood": (meta or {}).get("location", "Unknown"),
                        "cuisine": (meta or {}).get("cuisine", "Unknown"),
                        "score": round(1.0 - float(dist), 4),
                    }
                )
            _debug_log(f"[*] Vector search returned {len(vector_matches)} candidates")
        except Exception as e:
            _debug_log(f"[*] Vector search skipped (db unavailable or error): {e}")
            vector_matches = []
    else:
        _debug_log(f"[*] Sufficient structured matches ({len(structured_matches)}) found, skipping vector search")

    # Backfill sparse vector metadata from structured records by name.
    _debug_log("[*] Backfilling metadata from structured records...")
    restaurant_by_name = {
        _normalize_name(str(r.get("name", ""))): r
        for r in restaurants
        if str(r.get("name", "")).strip()
    }
    for match in vector_matches:
        source = restaurant_by_name.get(_normalize_name(match.get("name", "")))
        if not source:
            continue
        if _is_unknown(match.get("cuisine")):
            match["cuisine"] = _infer_cuisine(source)
        if _is_unknown(match.get("neighborhood")):
            match["neighborhood"] = source.get("neighborhood") or source.get("location", "Unknown")

    # Pass 2: Search the raw text for additional matches 
    raw_text = CULINARY_MAP_PATH.read_text(encoding="utf-8") if CULINARY_MAP_PATH else ""
    paragraphs = raw_text.split("\n\n")
    text_excerpts = []
    for para in paragraphs:
        if vibe_lower in para.lower() and para.strip():
            # Apply location filter to text excerpts
            if location_filter:
                if location_filter not in para.lower():
                    continue
            text_excerpts.append(para.strip()[:300])

    _debug_log(f"[*] Complete. Returning {len(structured_matches) + len(vector_matches)} matches.")
    return json.dumps(
        {
            "vibe_searched": vibe,
            "location_filtered": location_filter if location_filter else None,
            "vector_matches": vector_matches,
            "structured_matches": structured_matches,
            "raw_text_excerpts": text_excerpts[:5],
        },
        indent=2,
    )

# TOOL 3 — Get Review (Returns Review Data for Lab 2 Demonstration)
@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """Retrieve the full review for a restaurant."""
    _debug_log(f"[*] Fetching review for: {restaurant_name}")
    reviews = load_review_data()
    _debug_log(f"[*] Loaded {len(reviews)} reviews from database")
    query = restaurant_name.lower().strip()

    # Find the matching review
    matching_review = None
    for review in reviews:
        review_name = str(review.get("restaurant_name") or review.get("name") or "").lower()
        if query in review_name:
            matching_review = review
            break
    
    # Return a not found message if no review matches the query
    if not matching_review:
        _debug_log(f"[*] No review found matching: {restaurant_name}")
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No review found for '{restaurant_name}'.",
                "hint": "Review dataset is missing or this restaurant has no review entry yet.",
            },
            indent=2,
        )

    _debug_log("[*] Review found, formatting response")
    return json.dumps(
        {
            "status": "found",
            "restaurant": matching_review.get("restaurant_name") or matching_review.get("name", "Unknown"),
            "reviewer": matching_review.get("reviewer", "Unknown"),
            "rating": matching_review.get("rating", "N/A"),
            "review_text": matching_review.get("review_text") or matching_review.get("review", "N/A"),
            "image_description": matching_review.get("image_description", "N/A"),
            "visit_date": matching_review.get("visit_date", "N/A"),
        },
        indent=2,
    )

# Run the Server
if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8080,
    )
