"""
compute_embeddings.py
Reads images from data/images (matching data/products.csv),
computes CLIP image embeddings, writes:
 - embeddings.npy        (shape: N x D)
 - index.faiss           (FAISS index file)
 - products_metadata.json (list of metadata dicts in same order)

This script uses Hugging Face's transformers CLIPModel + CLIPProcessor
and faiss-cpu for indexing.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from PIL import Image

# Config
CSV_PATH = "data/products.csv"
IMAGES_DIR = "data/images"
OUTPUT_DIR = "backend/embeddings_output"
BATCH_SIZE = 16   # change based on your CPU/RAM/GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # reliable default

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_products(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure rows are deterministic — we'll use DataFrame index as order
    products = []
    for _, row in df.iterrows():
        products.append({
            "id": str(row["id"]),
            "name": str(row.get("name", "")),
            "category": str(row.get("category", "")),
            "price": float(row.get("price", 0.0)) if not pd.isna(row.get("price")) else None,
            "image_url": str(row.get("image_url", "")),
            "local_path": str(row.get("local_path", "")),
            "description": str(row.get("description", "")),
        })
    return products

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image {path}: {e}")
        return None

def main():
    print("Device:", DEVICE)
    products = load_products(CSV_PATH)
    print(f"Loaded {len(products)} products from {CSV_PATH}")

    # Filter only those with existing local images
    items = []
    for p in products:
        local_path = p["local_path"]
        # if path is relative, join with repo root
        img_path = Path(local_path)
        if not img_path.is_file():
            # also try IMAGES_DIR / filename if local_path was a filename
            fallback = Path(IMAGES_DIR) / img_path.name
            if fallback.is_file():
                p["local_path"] = str(fallback)
                items.append(p)
            else:
                print(f"Warning: image file missing for id={p['id']} path={local_path} — skipping")
        else:
            items.append(p)

    n_items = len(items)
    if n_items == 0:
        raise SystemExit("No images found in data/images. Please run download_images.py first or upload images.")

    # Load model + processor
    print("Loading CLIP model:", CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    all_embeds = []
    metadata = []

    model.eval()
    with torch.no_grad():
        for i in range(0, n_items, BATCH_SIZE):
            batch = items[i:i+BATCH_SIZE]
            images = []
            for p in batch:
                img = load_image(p["local_path"])
                if img is None:
                    # push a black image to keep batch size consistent
                    img = Image.new("RGB", (224,224), (0,0,0))
                images.append(img)

            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            # get image embeddings
            outputs = model.get_image_features(**inputs)
            # normalize
            embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            embeds = embeds.cpu().numpy()
            all_embeds.append(embeds)
            metadata.extend(batch)

            print(f"Processed batch {i//BATCH_SIZE + 1} / { (n_items + BATCH_SIZE -1)//BATCH_SIZE }")

    embeddings = np.vstack(all_embeds).astype("float32")
    print("Embeddings shape:", embeddings.shape)

    # Save embeddings (numpy)
    embeddings_path = Path(OUTPUT_DIR) / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print("Saved embeddings to:", embeddings_path)

    # Build FAISS index (inner product on normalized vectors == cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # exact search, good for small-medium datasets
    index.add(embeddings)
    index_path = Path(OUTPUT_DIR) / "index.faiss"
    faiss.write_index(index, str(index_path))
    print("Saved FAISS index to:", index_path)

    # Save metadata in same order as embeddings
    metadata_path = Path(OUTPUT_DIR) / "products_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("Saved metadata to:", metadata_path)

    # Save a small manifest
    manifest = {
        "n_items": n_items,
        "embeddings_path": str(embeddings_path),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "dim": dim
    }
    with open(Path(OUTPUT_DIR) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("All done. Outputs in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
