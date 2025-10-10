import pandas as pd
import requests
import os

os.makedirs("data/images", exist_ok=True)
df = pd.read_csv("data/products.csv")

for _, row in df.iterrows():
    image_url = row["image_url"]
    local_path = row["local_path"]

    try:
        print(f"Downloading: {image_url}")
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            # Save image bytes to the specified path
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Saved to {local_path}")
        else:
            print(f"⚠️ Failed: {image_url} (status {response.status_code})")

    except Exception as e:
        print(f"❌ Error downloading {image_url}: {e}")
