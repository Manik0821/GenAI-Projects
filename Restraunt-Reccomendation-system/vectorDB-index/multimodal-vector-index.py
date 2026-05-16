# ================================
# Imports
# ================================
import glob
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import requests

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


print("✅ Environment ready")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the expected files were found:\n" + "\n".join(str(p) for p in paths)
    )


def _seed_from_text(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _seed_from_file(path: str) -> int:
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _deterministic_vec(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


# ================================
# Download dataset
# ================================
ZIP_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
ZIP_PATH = PROJECT_ROOT / "synthetic-recipe-images.zip"
IMG_DIR = PROJECT_ROOT / "recipe_images"

if not ZIP_PATH.exists():
    print("⬇️ Downloading dataset...")
    r = requests.get(ZIP_URL, timeout=60)
    r.raise_for_status()
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)

if not IMG_DIR.exists():
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(IMG_DIR)

image_paths = sorted(glob.glob(str(IMG_DIR / "**" / "*.png"), recursive=True))
print(f"✅ Images found: {len(image_paths)}")

# ================================
# Load data
# ================================
restaurant_path = _first_existing(
    [
        PROJECT_ROOT / "structured_restaurant_data.json",
        PROJECT_ROOT / "structured-restaurant-data.json",
        SCRIPT_DIR / "structured_restaurant_data.json",
    ]
)

recipe_path = _first_existing(
    [
        PROJECT_ROOT / "augmented_food_recipe.json",
        PROJECT_ROOT / "Recipes.json",
        PROJECT_ROOT / "recipes.json",
    ]
)

with open(restaurant_path, "r", encoding="utf-8") as f:
    restaurants = json.load(f)

with open(recipe_path, "r", encoding="utf-8") as f:
    recipes = json.load(f)

print(f"✅ Restaurants: {len(restaurants)}")
print(f"✅ Recipes: {len(recipes)}")

# ================================
# Text embeddings
# ================================
TEXT_EMBED_DIM = 384

try:
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    USE_TEXT_MODEL = True
except Exception as e:
    print(f"⚠️ Could not load text embedding model; using deterministic fallback. Reason: {e}")
    text_model = None
    USE_TEXT_MODEL = False


def embed_texts(texts: list[str]) -> np.ndarray:
    if USE_TEXT_MODEL:
        return text_model.encode(texts, normalize_embeddings=True).astype(np.float32)

    vectors = [_deterministic_vec(_seed_from_text(t), TEXT_EMBED_DIM) for t in texts]
    return np.array(vectors, dtype=np.float32)


print("✅ Text model ready")

# ================================
# Image embeddings
# ================================
device = "cpu"
IMAGE_EMBED_DIM = 512

try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    USE_CLIP_MODEL = True
except Exception as e:
    print(f"⚠️ Could not load CLIP model; using deterministic fallback. Reason: {e}")
    clip_model = None
    clip_processor = None
    USE_CLIP_MODEL = False


@torch.no_grad()
def embed_images(paths: list[str]) -> np.ndarray:
    if not USE_CLIP_MODEL:
        vectors = [_deterministic_vec(_seed_from_file(p), IMAGE_EMBED_DIM) for p in paths]
        return np.array(vectors, dtype=np.float32)

    vectors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        feat = clip_model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        vectors.append(feat.cpu().numpy()[0])
    return np.array(vectors, dtype=np.float32)


print("✅ Image model ready")

# ================================
# Build documents
# ================================
article_docs = []
for i, r in enumerate(restaurants):
    name = r.get("name")
    if not name:
        continue

    content = f"""
    Restaurant: {name}
    Cuisine: {r.get('food_style', '')}
    Location: {r.get('location', '')}
    """

    article_docs.append(
        Document(
            page_content=content.strip(),
            metadata={
                "doc_id": f"rest_{i}",
                "source": "restaurant",
                "cuisine": r.get("food_style"),
                "location": r.get("location"),
            },
        )
    )

print(f"✅ Article docs: {len(article_docs)}")

image_docs = []
for i, (path, rec) in enumerate(zip(image_paths, recipes)):
    image_docs.append(
        Document(
            page_content=rec.get("name", f"image_{i}"),
            metadata={
                "doc_id": f"img_{i}",
                "image_path": path,
                "source": "image",
                "cuisine": rec.get("cuisine"),
            },
        )
    )

print(f"✅ Image docs: {len(image_docs)}")

# ================================
# Vector DB
# ================================
DB_DIR = str(PROJECT_ROOT / "chroma_db")

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

article_db = Chroma(collection_name="restaurants", persist_directory=DB_DIR)

article_vectors = embed_texts([d.page_content for d in article_docs])

article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=article_vectors.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)

print("✅ Article DB ready")

image_db = Chroma(
    collection_name="images",
    persist_directory=DB_DIR,
    embedding_function=None,
)

image_vectors = embed_images([d.metadata["image_path"] for d in image_docs])

image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in image_docs],
    embeddings=image_vectors.tolist(),
    documents=[d.page_content for d in image_docs],
    metadatas=[d.metadata for d in image_docs],
)

print("✅ Image DB ready")
print("🎉 DONE: Multimodal index built")

