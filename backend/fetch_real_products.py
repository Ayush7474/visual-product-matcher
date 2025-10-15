"""
Fetches ~100 real product images using Unsplash + Pexels APIs.
Saves them to data/images/ and creates data/products.csv with metadata.
"""

import os
import csv
import requests
from pathlib import Path
from tqdm import tqdm
import random

# ====== CONFIG ======
UNSPLASH_KEY = "vaM5RJmE4qOdbHHwfD-HrHEtnI62asRYQzhz5mlc0m4"   # 🔑 Replace this
PEXELS_KEY = "CAsLF9RfmTjhySHwr_L4R_m69rp4zuhR5c9UxRi2Y1g"          # optional
CATEGORIES = ["shoes", "watches", "bags", "phones", "laptops",
              "headphones", "sunglasses", "jackets", "bottle", "wallet"]

SAVE_DIR = Path("data/images")
CSV_PATH = Path("data/products.csv")
N_PER_CATEGORY = 10  # total ~100
# =====================

SAVE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_unsplash(query, per_page=10):
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={per_page}&orientation=squarish"
    headers = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        data = r.json().get("results", [])
        return [img["urls"]["small"] for img in data]
    return []

def fetch_pexels(query, per_page=10):
    if not PEXELS_KEY:
        return []
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}"
    headers = {"Authorization": PEXELS_KEY}
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 200:
        data = r.json().get("photos", [])
        return [img["src"]["medium"] for img in data]
    return []

def download_image(url, path):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False

def main():
    rows = []
    img_id = 1

    print(f"Fetching ~{len(CATEGORIES)*N_PER_CATEGORY} product images...")

    for cat in CATEGORIES:
        urls = fetch_unsplash(cat, N_PER_CATEGORY//2)
        urls += fetch_pexels(cat, N_PER_CATEGORY//2)
        random.shuffle(urls)
        urls = urls[:N_PER_CATEGORY]

        for url in tqdm(urls, desc=f"{cat:10s}"):
            filename = f"{img_id}.jpg"
            img_path = SAVE_DIR / filename
            if download_image(url, img_path):
                rows.append({
                    "id": img_id,
                    "name": f"{cat.capitalize()} {img_id}",
                    "category": cat,
                    "price": round(random.uniform(500, 5000), 2),
                    "image_url": url,
                    "local_path": str(img_path),
                    "description": f"High quality {cat} product"
                })
                img_id += 1

    # Write metadata CSV
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Download complete! {len(rows)} images saved to {SAVE_DIR}")
    print(f"✅ Metadata written to {CSV_PATH}")

if __name__ == "__main__":
    main()
