import csv
import random

output_csv = "data/products.csv"

categories = {
    "Shoes": ["Running Shoes", "Sneakers", "Boots", "Sandals"],
    "Bags": ["Backpack", "Handbag", "Laptop Bag", "Duffel Bag"],
    "Watches": ["Analog Watch", "Digital Watch", "Smartwatch"],
    "Glasses": ["Sunglasses", "Reading Glasses", "Aviator Glasses"],
    "Headphones": ["Wireless Headphones", "Earbuds", "Gaming Headset"],
    "Clothes": ["T-shirt", "Hoodie", "Jacket", "Jeans"]
}

# Use Picsum (stable and random images)
base_url = "https://picsum.photos/seed"

num_products = 100
rows = []

for i in range(1, num_products + 1):
    category = random.choice(list(categories.keys()))
    name = random.choice(categories[category])
    price = random.randint(500, 5000)
    image_url = f"{base_url}/{i}/640/480"  # deterministic per ID

    rows.append({
        "id": i,
        "name": f"{name} {i}",
        "category": category,
        "price": price,
        "image_url": image_url
    })

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "name", "category", "price", "image_url"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Generated {num_products} fake products in {output_csv}")
