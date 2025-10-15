# backend/main.py
import io
import os
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import requests

# --- START: Corrected Config for Deployment ---
# In Render, the root directory is 'backend', so we look for the subfolder directly.
EMBEDDINGS_DIR = Path("embeddings_output") 
INDEX_PATH = EMBEDDINGS_DIR / "index.faiss"
METADATA_PATH = EMBEDDINGS_DIR / "products_metadata.json"
# --- END: Corrected Config ---

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_DEFAULT = 5

app = FastAPI(title="Visual Product Matcher API (FastAPI)")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and metadata on startup
print("Starting server — device:", DEVICE)
if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    raise SystemExit(
        f"Index or metadata not found. Make sure embedding step ran. Missing: {INDEX_PATH.exists()=}, {METADATA_PATH.exists()=}"
    )

print("Loading FAISS index:", INDEX_PATH)
index = faiss.read_index(str(INDEX_PATH))

print("Loading metadata:", METADATA_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load CLIP
print("Loading CLIP model:", CLIP_MODEL_NAME)
model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
model.eval()

def _image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def _compute_embedding_from_image(img: Image.Image) -> np.ndarray:
    """
    Returns a normalized numpy vector (float32) for the image.
    """
    inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    vec = feat.cpu().numpy().astype("float32")[0]
    return vec

def _search_index(vec: np.ndarray, top_k: int = TOP_K_DEFAULT):
    """
    Searches FAISS index (expects normalized vectors if using IndexFlatIP).
    Returns (indices, scores) numpy arrays.
    """
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    D, I = index.search(vec, top_k)
    return I[0].tolist(), D[0].tolist()

@app.post("/search")
async def search_image(file: Optional[UploadFile] = File(None), image_url: Optional[str] = Form(None), top_k: int = Form(TOP_K_DEFAULT)):
    if file is None and not image_url:
        return JSONResponse({"error": "Provide either file upload or image_url"}, status_code=400)

    try:
        if file:
            img_bytes = await file.read()
        else:
            resp = requests.get(image_url, timeout=10)
            resp.raise_for_status()
            img_bytes = resp.content
        img = _image_from_bytes(img_bytes)
    except Exception as e:
        return JSONResponse({"error": f"Failed to load image: {e}"}, status_code=400)

    try:
        vec = _compute_embedding_from_image(img)
    except Exception as e:
        return JSONResponse({"error": f"Failed to compute embedding: {e}"}, status_code=500)

    try:
        ids, scores = _search_index(vec, top_k=top_k)
    except Exception as e:
        return JSONResponse({"error": f"Search failed: {e}"}, status_code=500)

    results = []
    for idx, score in zip(ids, scores):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return {"query_device": DEVICE, "top_k": len(results), "results": results}

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "n_items": len(metadata)}

# This block is ignored by Uvicorn on Render but is useful for local testing
if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
