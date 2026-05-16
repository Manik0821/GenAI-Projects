# Libraries to import to create our MCP server and handle data loading
from fastmcp import FastMCP
from pathlib import Path
import json
import re
import hashlib

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
DATA_DIR = Path(__file__).parent
PROJECT_ROOT = DATA_DIR.parent


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
        DATA_DIR / "structured-restaurant-data.json",
        DATA_DIR / "structured_restaurant_data.json",
        PROJECT_ROOT / "structured-restaurant-data.json",
        PROJECT_ROOT / "structured_restaurant_data.json",
        PROJECT_ROOT / "structured_restaurants.json",
        PROJECT_ROOT / "Loading-Data" / "structured_restaurants.json",
        PROJECT_ROOT / "Managing-data" / "structured_restaurant_data.json",
    ]
)

REVIEW_DATA_PATH = _first_existing(
    [
        DATA_DIR / "augmented-user-review.json",
        DATA_DIR / "augmented_user_review.json",
        PROJECT_ROOT / "augmented-user-review.json",
        PROJECT_ROOT / "augmented_user_review.json",
        PROJECT_ROOT / "Synthetic-User-Reviews.json",
        PROJECT_ROOT / "processing-data" / "Synthetic-User-Reviews.json",
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
            _TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            _TEXT_MODEL_READY = True
        except Exception:
            _TEXT_MODEL = None
            _TEXT_MODEL_READY = True

    if _TEXT_MODEL is None:
        return _deterministic_query_vec(query)

    vec = _TEXT_MODEL.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    return vec


def _get_article_db():
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
    m = re.search(r"Restaurant:\s*(.+)", doc)
    return m.group(1).strip() if m else "Unknown"


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

        rows.append(
            {
                "name": name,
                "neighborhood": location,
                "location": location,
                "cuisine": "Unknown",
                "food_style": "Unknown",
                "rating": _parse_rating(para),
                "vibes": tags,
                "price_range": _parse_price_dollar_count(para),
                "description": para,
            }
        )

    return rows

# Helper functions
def load_restaurant_data() -> list[dict]:
    """Load the structured restaurant data produced in Module 1."""
    if RESTAURANT_DATA_PATH and RESTAURANT_DATA_PATH.exists():
        with open(RESTAURANT_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []

    # Fallback keeps app usable when step outputs are not generated yet.
    return _fallback_restaurants_from_map()

def load_review_data() -> list[dict]:
    """Load the augmented user reviews produced in Module 1."""
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
    restaurants = load_restaurant_data()
    query = restaurant_name.lower().strip()

    # Finding restuarants that match the query in the structured JSON data
    matches = []
    for restaurant in restaurants:
        name = str(restaurant.get("name", "")).lower()
        if query in name or name in query:
            matches.append(restaurant)

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

    return json.dumps(
        {"status": "found", "count": len(matches), "results": matches},
        indent=2,
    )
# TOOL 2 — Recommend by Vibe (Semantic Search)
@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """Find restaurants that match a given vibe or atmosphere keyword.
    Searches both structured vibe tags and raw text descriptions.
    Examples of vibe keywords: "moody", "sun-drenched", "romantic"""
    restaurants = load_restaurant_data()
    vibe_lower = vibe.lower().strip()

    # Fast path: use Chroma vector search when index is available.
    vector_matches = []
    article_db = _get_article_db()
    if article_db is not None:
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
                vector_matches.append(
                    {
                        "name": _extract_name_from_doc(doc or ""),
                        "neighborhood": (meta or {}).get("location", "Unknown"),
                        "cuisine": (meta or {}).get("cuisine", "Unknown"),
                        "score": round(1.0 - float(dist), 4),
                    }
                )
        except Exception:
            vector_matches = []

    # Pass 1: Search structured vibe tags in JSON
    structured_matches = []
    for restaurant in restaurants:
        vibes_list = [v.lower() for v in restaurant.get("vibes", [])]
        description = restaurant.get("description", "").lower()

        if any(vibe_lower in v for v in vibes_list) or vibe_lower in description:
            structured_matches.append(
                {
                    "name": restaurant.get("name", "Unknown"),
                    "neighborhood": restaurant.get("neighborhood") or restaurant.get("location", "Unknown"),
                    "cuisine": restaurant.get("cuisine") or restaurant.get("food_style", "Unknown"),
                    "rating": restaurant.get("rating", "N/A"),
                    "vibes": restaurant["vibes"],
                    "price_range": restaurant.get("price_range", "N/A"),
                }
            )

    # Pass 2: Search the raw text for additional matches 
    raw_text = CULINARY_MAP_PATH.read_text(encoding="utf-8") if CULINARY_MAP_PATH else ""
    paragraphs = raw_text.split("\n\n")
    text_excerpts = []
    for para in paragraphs:
        if vibe_lower in para.lower() and para.strip():
            text_excerpts.append(para.strip()[:300])

    return json.dumps(
        {
            "vibe_searched": vibe,
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
    reviews = load_review_data()
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
        return json.dumps(
            {
                "status": "not_found",
                "message": f"No review found for '{restaurant_name}'.",
                "hint": "Review dataset is missing or this restaurant has no review entry yet.",
            },
            indent=2,
        )

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
    mcp.run()
