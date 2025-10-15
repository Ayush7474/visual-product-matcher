import os
import csv
import requests
from tqdm import tqdm

# Paths
CSV_PATH = "data/products.csv"
IMAGE_DIR = "data/images"

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Read product metadata
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    products = list(reader)

print(f"Found {len(products)} products in {CSV_PATH}")

# Download each image
for p in tqdm(products, desc="Downloading images"):
    image_url = p["image_url"]
    image_path = os.path.join(IMAGE_DIR, f"{p['id']}.jpg")

    # Skip if already downloaded
    if os.path.exists(image_path):
        continue

    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(image_path, "wb") as img_file:
                img_file.write(response.content)
        else:
            print(f"⚠️ Skipped ID {p['id']} (status {response.status_code})")
    except Exception as e:
        print(f"❌ Error downloading ID {p['id']}: {e}")

print(f"\n✅ Download complete! Images saved to: {IMAGE_DIR}")
