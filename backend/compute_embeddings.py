"""
compute_embeddings.py
Reads images from data/images (matching data/products.csv),
computes CLIP image embeddings, writes:
 - embeddings.npy
 - index.faiss
 - products_metadata.json
Uses Hugging Face CLIPModel + CLIPProcessor and faiss-cpu.
"""

import os, json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from PIL import Image

# Paths and config
CSV_PATH = "data/products.csv"
IMAGES_DIR = Path("data/images")
OUTPUT_DIR = Path("backend/embeddings_output")
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_products(csv_path):
    df = pd.read_csv(csv_path)
    products = []
    for _, row in df.iterrows():
        # construct local path automatically
        img_path = IMAGES_DIR / f"{row['id']}.jpg"
        products.append({
            "id": str(row["id"]),
            "name": str(row.get("name", "")),
            "category": str(row.get("category", "")),
            "price": float(row.get("price", 0.0)) if not pd.isna(row.get("price")) else None,
            "image_url": str(row.get("image_url", "")),
            "local_path": str(img_path),
            "description": str(row.get("description", "")),
        })
    return products

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Failed to open image {path}: {e}")
        return None

def main():
    print("Device:", DEVICE)
    products = load_products(CSV_PATH)
    print(f"Loaded {len(products)} products from {CSV_PATH}")

    # Keep only those whose images exist
    items = [p for p in products if Path(p["local_path"]).is_file()]
    if not items:
        raise SystemExit("❌ No images found in data/images. Run download_images.py first.")

    print(f"Found {len(items)} valid images. Loading CLIP model...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    all_embeds, metadata = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i:i+BATCH_SIZE]
            images = []
            for p in batch:
                img = load_image(p["local_path"])
                if img is None:
                    img = Image.new("RGB", (224,224), (0,0,0))
                images.append(img)

            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            outputs = model.get_image_features(**inputs)
            embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu().numpy())
            metadata.extend(batch)
            print(f"Processed batch {i//BATCH_SIZE + 1} / {(len(items)+BATCH_SIZE-1)//BATCH_SIZE}")

    embeddings = np.vstack(all_embeds).astype("float32")
    print("Embeddings shape:", embeddings.shape)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    print("✅ Saved embeddings.npy")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    print("✅ Saved index.faiss")

    with open(OUTPUT_DIR / "products_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("✅ Saved products_metadata.json")

    manifest = {
        "n_items": len(items),
        "dim": dim,
        "device": DEVICE
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"🎉 Done. {len(items)} embeddings saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
