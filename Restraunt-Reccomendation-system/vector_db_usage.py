"""Demonstrates retrieval queries over prebuilt text and image vector indexes."""

# ================================
# Import dependencies
# ================================

# Standard library
import os
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")

from langchain_chroma import Chroma
import os
import gc

DB_DIR = "./chroma_db"

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run index construction first."
    )

article_db = None
image_db = None

try:
    # ✅ MUST match build step names
    article_db = Chroma(
        collection_name="restaurants",
        persist_directory=DB_DIR,
    )

    image_db = Chroma(
        collection_name="images",
        persist_directory=DB_DIR,
    )

    n_articles = article_db._collection.count()
    n_images = image_db._collection.count()

    if n_articles <= 0 or n_images <= 0:
        raise RuntimeError(
            f"Collections empty ❌ | restaurants={n_articles}, images={n_images}"
        )

    print(f"✅ Article vectors: {n_articles}")
    print(f"✅ Image vectors:   {n_images}")

finally:
    # 🔥 ALWAYS release (important in notebooks)
    # release_chroma(article_db)
    # release_chroma(image_db)

    # article_db = None
    # image_db = None
    gc.collect()

# ================================
# Initialize embedding models
# ================================

# ---- Text embedding model (384-d) ----
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)

print("✅ Text embedder ready")


# ---- Image embedding model (512-d) ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

print("✅ Image embedder ready")

# ================================
# Retrieval utilities
# ================================

# Chroma returns lists-of-lists; unwrap the first query.
def _unwrap(res: dict):  
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists

def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        dist = float(dists[i]) if i < len(dists) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        # compact metadata view
        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | distance={dist:.4f}")
        print(f"{snippet}")

# ================================
# Article retrieval
# ================================

# Similarity retrieval over restaurant articles with optional metadata filtering.
def retrieve_articles(query: str, k: int = 5, where: dict | None = None):

    q_vec = embed_texts([query])[0]  # 384-d, cosine-ready

    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

print("✅ Article retrieval ready")

# ================================
# Image retrieval
# ================================

# Similarity retrieval over food images using an image query.
def retrieve_images_by_image(query_image_path: str, k: int = 5, where: dict | None = None):

    q_vec = embed_images([query_image_path])[0]  # 512-d, cosine-ready

    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

print("✅ Image retrieval ready")

# ================================
# Demo 1 — Article similarity search (no filter)
# ================================

q = "cozy restaurant with noodles and warm atmosphere"

ids, docs, metas, dists = retrieve_articles(q, k=5, where=None)
print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")

print("✅ Demo 1 complete")

# ================================
# Demo 2 — Article similarity search + metadata filter
# ================================

q = "handmade pasta and romantic dinner"

# ---- metadata constraint (must exist in your dataset) ----
where_filter = {"location": "Pasadena"}  # adjust if needed

ids, docs, metas, dists = retrieve_articles(q, k=5, where=where_filter)

if len(ids) == 0:
    print("⚠️ No results found with current filter.")
else:
    print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search + metadata filter")
    
print("✅ Demo 2 complete")

# ================================
# Demo 3 — Image similarity search (image→image)
# ================================

from PIL import Image
from IPython.display import display

meta_all = image_db._collection.get(include=["metadatas"])["metadatas"]

QUERY_INDEX = 0  # change this

if QUERY_INDEX >= len(meta_all):
    raise ValueError("QUERY_INDEX out of range.")

query_img = meta_all[QUERY_INDEX]["image_path"]

print(f"Query image: {query_img}")
img = Image.open(query_img)
img.thumbnail((300, 300))
display(img)

# -------------------------------
# 1. Optional metadata filter
# -------------------------------
# Example: filter by cuisine (set to None to disable)
FILTER = None
# FILTER = {"cuisine": "italian"}

# -------------------------------
# 2. Compute embedding for query image
# -------------------------------
query_vec = embed_images([query_img]).tolist()

# -------------------------------
# 3. Query top-5 similar images
# -------------------------------
results = image_db._collection.query(
    query_embeddings=query_vec,
    n_results=5,
    where=FILTER  # applies only if not None
)

# -------------------------------
# 4. Display results
# -------------------------------
print("\n🔎 Demo 3 — Image similarity search (image→image)\n")

for i in range(len(results["ids"][0])):
    meta = results["metadatas"][0][i]
    path = meta["image_path"]

    print(f"Result {i+1}")
    print(f"Recipe:  {results['documents'][0][i]}")
    print(f"Cuisine: {meta.get('cuisine')}")
    print(f"Path:    {path}")

    try:
        img = Image.open(path)
        img.thumbnail((250, 250))
        img.show()
    except Exception as e:
        print(f"⚠️ Could not load image: {e}")

    print("-" * 40)

print("✅ Demo 3 complete")
print("🎉 Similarity Retrieval with Metadata Filtering COMPLETE")