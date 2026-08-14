import os
import pandas as pd
import requests

CSV_FILE = "byrappa_tejas_31july.csv"
IMAGE_DIR = "data/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

session = requests.Session()

success = 0
failed = 0
failed_rows = []
for index, row in df.iterrows():
    sku = row["SKU"]
    url = row["image_url"]

    filename = f"{index:04d}_{sku}.webp"
    filepath = os.path.join(IMAGE_DIR, filename)

    # Don't download again if it already exists
    if os.path.exists(filepath):
        success += 1
        continue

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        success += 1
        print(f"[{success}/{len(df)}] Downloaded {filename}")

    except Exception as e:
        failed += 1
        failed_rows.append({
        "index": index,
        "sku": sku,
        "url": url,
        "error": str(e)
    })        
        print(f"[FAILED] Row {index} | {url}")
        print(f"        {e}")

print("\n--- DOWNLOAD COMPLETE ---")
print(f"Successful: {success}")
print(f"Failed:     {failed}")
print("\n--- FAILED DOWNLOADS ---")

for item in failed_rows:
    print(
        f"Row {item['index']} | "
        f"SKU {item['sku']} | "
        f"{item['error']}"
    )
    print(item["url"])