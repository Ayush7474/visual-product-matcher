# backend/app_streamlit.py

import streamlit as st
from PIL import Image
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import json
import requests
import io

# --- START: All-in-One Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, 'embeddings_output')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

METADATA_PATH = os.path.join(EMBEDDINGS_DIR, 'products_metadata.json')
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, 'index.faiss')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cpu"
# --- END: Configuration ---


# --- START: Backend Logic (Merged from main.py) ---

@st.cache_resource
def load_models_and_data():
    """Load all necessary models and data into memory once."""
    st.write("Loading models and data... This may take a moment on first run.")
    
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Could not find index or metadata files in 'backend/embeddings_output'.")

    index = faiss.read_index(FAISS_INDEX_PATH)
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    
    return index, metadata, model, processor

def _image_from_bytes(data: bytes) -> Image.Image:
    """Creates a PIL Image from bytes."""
    return Image.open(io.BytesIO(data)).convert("RGB")

def _compute_embedding(img: Image.Image, model, processor) -> np.ndarray:
    """Computes a normalized embedding for the image."""
    inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")

def _search_index(vec: np.ndarray, index, top_k: int):
    """Searches the FAISS index."""
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    distances, indices = index.search(vec, top_k)
    return indices[0].tolist(), distances[0].tolist()

def get_image_path(product_id):
    """Constructs the local path for a product image."""
    str_product_id = str(product_id).zfill(10)
    folder = str_product_id[:3]
    return os.path.join(IMAGES_DIR, folder, f"{str_product_id}.jpg")
    
# --- END: Backend Logic ---


# --- START: Streamlit UI ---

st.set_page_config(page_title="Visual Product Matcher", layout="wide")
st.title("🖼️ Visual Product Matcher")
st.write("Upload a product image or paste a URL to find visually similar items.")

try:
    index, metadata, model, processor = load_models_and_data()
except Exception as e:
    st.error(f"Fatal Error: Could not load necessary model files. Please check the logs.")
    st.error(e)
    st.stop()

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
with col2:
    image_url = st.text_input("Or enter an image URL")

top_k = st.slider("Number of similar results", 3, 10, 5)

if st.button("🔍 Search"):
    if not uploaded_file and not image_url:
        st.warning("Please upload an image or enter a URL.")
        st.stop()

    try:
        if uploaded_file:
            img_bytes = uploaded_file.getvalue()
        else:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img_bytes = response.content
        
        query_image = _image_from_bytes(img_bytes)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        st.stop()

    st.subheader("Query Image:")
    st.image(query_image, use_container_width=True) # <-- FIX IS HERE

    query_vector = _compute_embedding(query_image, model, processor)
    indices, scores = _search_index(query_vector, index, top_k=top_k)
    
    st.subheader("Similar Items Found:")
    results_cols = st.columns(top_k)
    
    for i, (idx, score) in enumerate(zip(indices, scores)):
        with results_cols[i]:
            item_metadata = metadata[idx]
            item_id = item_metadata['product_id']
            item_image_path = get_image_path(item_id)
            
            if os.path.exists(item_image_path):
                st.image(item_image_path, use_container_width=True) # <-- FIX IS HERE
                st.caption(f"ID: {item_id}\n\nScore: {score:.2f}")
            else:
                st.warning(f"Img not found for {item_id}")
