import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "patrickjohncyh/fashion-clip"

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "sarees_fashion"

CSV_FILE = "byrappa_tejas_31july.csv"

TOP_K = 5

device = "cpu"


# ==========================================
# Load FashionCLIP
# ==========================================

print("Loading FashionCLIP...")

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("FashionCLIP loaded.")


# ==========================================
# Load catalogue
# ==========================================

print("Loading catalogue...")

catalogue = pd.read_csv(
    CSV_FILE
)

catalogue["SKU"] = (
    catalogue["SKU"]
    .astype(str)
    .str.strip()
)

catalogue_by_sku = {}

for _, row in catalogue.iterrows():

    sku = str(row["SKU"]).strip()

    if sku not in catalogue_by_sku:
        catalogue_by_sku[sku] = row.to_dict()

print(
    f"Catalogue loaded: {len(catalogue)} products"
)


# ==========================================
# Load Qdrant
# ==========================================

print("Loading FashionCLIP Qdrant...")

client = QdrantClient(
    path=QDRANT_PATH
)

print("Qdrant loaded.")


# ==========================================
# Generate FashionCLIP embedding
# ==========================================

def get_image_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        image_features = model.visual_projection(
            vision_outputs.pooler_output
        )

    # Normalize
    image_features = (
        image_features
        / image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    return image_features.cpu().numpy()[0]


# ==========================================
# Extract SKU from filename
# ==========================================

def get_sku_from_filename(filename):

    if not filename:
        return None

    # Example:
    # 0229_QS225313.webp
    #
    # Split:
    # 0229
    # QS225313

    name = os.path.splitext(
        filename
    )[0]

    parts = name.split("_")

    if len(parts) < 2:
        return None

    return "_".join(parts[1:])


# ==========================================
# Search
# ==========================================

def search_image(
    image,
    top_k=TOP_K,
    exclude_filename=None
):

    embedding = get_image_embedding(
        image
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding.tolist(),
        limit=top_k + 5
    ).points

    output = []

    for result in results:

        payload = result.payload or {}

        filename = payload.get(
            "filename"
        )

        # --------------------------------------
        # Fallback: Qdrant may store filename
        # as the point payload
        # --------------------------------------

        if not filename:
            continue

        # Exclude query image
        if (
            exclude_filename
            and filename == exclude_filename
        ):
            continue

        # --------------------------------------
        # Get SKU from filename
        # --------------------------------------

        sku = get_sku_from_filename(
            filename
        )

        # --------------------------------------
        # Lookup catalogue metadata
        # --------------------------------------

        product = catalogue_by_sku.get(
            sku,
            {}
        )

        output.append(
            {
                "filename": filename,

                "score": float(
                    result.score
                ),

                "name": product.get(
                    "Name"
                ),

                "sku": product.get(
                    "SKU",
                    sku
                ),

                "stock": product.get(
                    "Stock"
                ),

                "retail_price": product.get(
                    "Retail Price"
                ),

                "discounted_price": product.get(
                    "Discounted Price"
                ),

                "website_link": product.get(
                    "Website Link"
                ),

                "image_url": product.get(
                    "image_url"
                )
            }
        )

        if len(output) == top_k:
            break

    return output