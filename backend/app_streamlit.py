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

# Build paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, 'embeddings_output')

# Define file paths for the model and metadata
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, 'products_metadata.json')
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, 'index.faiss')

# Model and device configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cpu" # Force CPU; Streamlit Cloud's free tier has no GPU

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
    distances, indices = index.search(vec, top_k)
    return indices[0].tolist(), distances[0].tolist()

# --- END: Backend Logic ---


# --- START: Streamlit UI ---

st.set_page_config(page_title="Visual Product Matcher", layout="wide")
st.title("🖼️ Visual Product Matcher")
st.write("Upload a product image or paste an image URL to find visually similar items.")

# Load models and data at the start
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

    # Get image from either source
    try:
        if uploaded_file:
            img_bytes = uploaded_file.getvalue()
        else:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img_bytes = response.content
        
        query_image = _image_from_bytes(img_bytes)
        st.subheader("Query Image")
        st.image(query_image, width=200)

    except Exception as e:
        st.error(f"Could not load image. Error: {e}")
        st.stop()

    # Perform search using the local functions
    try:
        query_vector = _compute_embedding(query_image, model, processor)
        indices, scores = _search_index(query_vector, index, top_k)

        st.success(f"Found {len(indices)} similar items!")
        
        # Display results
        cols = st.columns(len(indices))
        for i, (idx, score) in enumerate(zip(indices, scores)):
            with cols[i]:
                item = metadata[idx]
                # Display image from URL if available, otherwise assume a local path needs constructing
                # Based on your main.py, metadata contains the URL
                if "image_url" in item and item["image_url"]:
                     st.image(item["image_url"], use_column_width=True)
                # Fallback for local path if needed, though your schema uses URLs
                elif "local_path" in item and item["local_path"]:
                     st.image(item["local_path"], use_column_width=True)
                
                # Default captioning
                name = item.get('name', 'N/A')
                category = item.get('category', 'N/A')
                st.caption(f"{name} ({category})\n\n🔹 Score: {score:.2f}")

    except Exception as e:
        st.error(f"An error occurred during the search process. Error: {e}")

# --- END: Streamlit UI ---
