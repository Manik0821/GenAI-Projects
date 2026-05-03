# ================================
# Imports
# ================================
import glob
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

# ================================
# Download dataset
# ================================
ZIP_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
ZIP_PATH = "synthetic-recipe-images.zip"
IMG_DIR = "recipe_images"

if not os.path.exists(ZIP_PATH):
    print("⬇️ Downloading dataset...")
    r = requests.get(ZIP_URL)
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)

if not os.path.exists(IMG_DIR):
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(IMG_DIR)

image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))
print(f"✅ Images found: {len(image_paths)}")

# ================================
# Load data
# ================================
with open("structured_restaurant_data.json", "r") as f:
    restaurants = json.load(f)

with open("augmented_food_recipe.json", "r") as f:
    recipes = json.load(f)

print(f"✅ Restaurants: {len(restaurants)}")
print(f"✅ Recipes: {len(recipes)}")

# ================================
# TEXT EMBEDDINGS
# ================================
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return text_model.encode(
        texts,
        normalize_embeddings=True
    ).astype(np.float32)

print("✅ Text model ready")

# ================================
# IMAGE EMBEDDINGS
# ================================
device = "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def embed_images(paths):
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
# BUILD DOCUMENTS
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
                "location": r.get("location")
            }
        )
    )

print(f"✅ Article docs: {len(article_docs)}")

# ---- Image docs ----
image_docs = []

for i, (path, rec) in enumerate(zip(image_paths, recipes)):
    image_docs.append(
        Document(
            page_content=rec.get("name", f"image_{i}"),
            metadata={
                "doc_id": f"img_{i}",
                "image_path": path,
                "source": "image",
                "cuisine": rec.get("cuisine")
            }
        )
    )

print(f"✅ Image docs: {len(image_docs)}")

# ================================
# VECTOR DB
# ================================
DB_DIR = str(Path("chroma_db"))

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

# ---- Article DB ----
article_db = Chroma(
    collection_name="restaurants",
    persist_directory=DB_DIR
)

article_vectors = embed_texts([d.page_content for d in article_docs])

article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=article_vectors.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)

print("✅ Article DB ready")

# ---- Image DB ----
image_db = Chroma(
    collection_name="images",
    persist_directory=DB_DIR,
    embedding_function=None  # we provide embeddings manually
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

