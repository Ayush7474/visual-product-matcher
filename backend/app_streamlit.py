import streamlit as st
import requests
import os
from PIL import Image
import io

# Backend URL (your local FastAPI server or Render deployment)
BACKEND_URL = "http://127.0.0.1:8000/search"  # change later if deployed

st.set_page_config(page_title="Visual Product Matcher", layout="wide")

st.title("🖼️ Visual Product Matcher")
st.write("Upload a product image or paste an image URL to find visually similar items.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
with col2:
    image_url = st.text_input("Or enter an image URL")

top_k = st.slider("Number of similar results", 3, 10, 5)

if st.button("🔍 Search"):
    if uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        data = {"top_k": top_k}
        response = requests.post(BACKEND_URL, files=files, data=data)
    elif image_url:
        data = {"image_url": image_url, "top_k": top_k}
        response = requests.post(BACKEND_URL, data=data)
    else:
        st.warning("Please upload an image or enter a URL.")
        st.stop()

    if response.status_code == 200:
        results = response.json()
        st.success(f"Found {len(results['results'])} similar items!")

        cols = st.columns(len(results["results"]))
        for i, item in enumerate(results["results"]):
            with cols[i]:
                st.image(item["image_url"] or item["local_path"], use_container_width=True)
                st.caption(f"{item['name']} ({item['category']})\n\n🔹 Score: {item['score']:.2f}")
    else:
        st.error(f"Error: {response.status_code}")
