"""Build Chroma indexes for restaurant recommendation search."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DB_DIR = PROJECT_ROOT / "chroma_db"


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of these files exist:\n" + "\n".join(str(p) for p in paths)
    )


RESTAURANT_PATH = first_existing(
    [
        PROJECT_ROOT / "structured_restaurant_data.json",
        PROJECT_ROOT / "Managing-data" / "structured_restaurant_data.json",
        PROJECT_ROOT / "Loading-Data" / "structured_restaurants.json",
    ]
)

REVIEW_PATH = first_existing(
    [
        PROJECT_ROOT / "processing-data" / "augmented_user_review.json",
        PROJECT_ROOT / "augmented_user_review.json",
        PROJECT_ROOT / "processing-data" / "Synthetic-User-Reviews.json",
    ]
)


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def normalize_name(value: str) -> str:
    return " ".join(str(value or "").lower().strip().split())


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


print("Loading data...")
restaurants = load_json(RESTAURANT_PATH)
reviews = load_json(REVIEW_PATH)

print(f"Restaurants loaded: {len(restaurants)}")
print(f"Reviews loaded: {len(reviews)}")

reviews_by_name = {}

for review in reviews:
    name = (
        review.get("restaurant_name")
        or review.get("name")
        or review.get("restaurant")
        or ""
    )

    if not name:
        continue

    reviews_by_name.setdefault(normalize_name(name), []).append(review)


documents: list[Document] = []

for index, restaurant in enumerate(restaurants):
    name = restaurant.get("name", "")
    if not name:
        continue

    restaurant_reviews = reviews_by_name.get(normalize_name(name), [])

    review_texts = []
    image_captions = []

    for review in restaurant_reviews:
        review_texts.append(
            review.get("review_text")
            or review.get("review")
            or review.get("text")
            or ""
        )

        captions = review.get("image_captions") or review.get("image_description") or ""

        if isinstance(captions, list):
            image_captions.extend(captions)
        else:
            image_captions.append(captions)

    cuisine = (
        restaurant.get("cuisine")
        or restaurant.get("food_style")
        or restaurant.get("type")
        or ""
    )

    location = (
        restaurant.get("neighborhood")
        or restaurant.get("location")
        or ""
    )

    content = f"""
Restaurant: {name}
Cuisine: {cuisine}
Type: {restaurant.get("type", "")}
Location: {location}
Rating: {restaurant.get("rating", "")}
Price Range: {restaurant.get("price_range", "")}
Signature Dish: {restaurant.get("signature_dish", "")}
Vibes: {safe_text(restaurant.get("vibes", []))}
Description: {restaurant.get("description", "")}
Reviews: {safe_text(review_texts)}
Image Captions: {safe_text(image_captions)}
""".strip()

    documents.append(
        Document(
            page_content=content,
            metadata={
                "doc_id": f"rest_{index}",
                "source": "restaurant",
                "name": name,
                "cuisine": cuisine,
                "location": location,
            },
        )
    )

print(f"Documents created: {len(documents)}")

if DB_DIR.exists():
    print(f"Removing old Chroma DB: {DB_DIR}")
    shutil.rmtree(DB_DIR)

DB_DIR.mkdir(parents=True, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc.page_content for doc in documents]
embeddings = model.encode(texts, normalize_embeddings=True).astype(np.float32)

print("Creating Chroma collection...")

db = Chroma(
    collection_name="restaurants",
    persist_directory=str(DB_DIR),
    embedding_function=None,
)

db._collection.upsert(
    ids=[doc.metadata["doc_id"] for doc in documents],
    embeddings=embeddings.tolist(),
    documents=[doc.page_content for doc in documents],
    metadatas=[doc.metadata for doc in documents],
)

count = db._collection.count()

print(f"Chroma restaurant collection count: {count}")
print(f"Chroma DB path: {DB_DIR}")

if count == 0:
    raise RuntimeError("Chroma DB build failed. Collection is empty.")

print("DONE: Chroma index built successfully.")