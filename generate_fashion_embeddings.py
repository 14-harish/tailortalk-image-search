import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "patrickjohncyh/fashion-clip"
IMAGE_DIR = "data/images"

EMBEDDINGS_FILE = "data/fashion_embeddings.npy"
FILES_FILE = "data/fashion_valid_files.npy"

print("Loading FashionCLIP...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model = CLIPModel.from_pretrained(MODEL_NAME)
model = model.to("cpu")
model.eval()

print("FashionCLIP loaded.")

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".webp")
])

print(f"Images found: {len(image_files)}")

embeddings = []
valid_files = []

print("\nGenerating FashionCLIP embeddings...\n")

for i, filename in enumerate(image_files, start=1):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    try:
        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():

            vision_outputs = model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            pooled_output = vision_outputs.pooler_output

            image_features = model.visual_projection(
                pooled_output
            )

        embedding = (
            image_features
            .cpu()
            .numpy()[0]
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            print(
                f"Skipping zero embedding: {filename}"
            )
            continue

        embedding = embedding / norm

        embeddings.append(
            embedding.astype("float32")
        )

        valid_files.append(filename)

        print(
            f"[{i}/{len(image_files)}] "
            f"{filename}",
            flush=True
        )

    except Exception as e:

        print(
            f"[FAILED] {filename}: {e}",
            flush=True
        )


embeddings = np.array(
    embeddings,
    dtype="float32"
)

valid_files = np.array(
    valid_files
)


print("\n================================")
print("FASHIONCLIP EMBEDDING COMPLETE")
print("================================")

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Valid images:",
    len(valid_files)
)

np.save(
    EMBEDDINGS_FILE,
    embeddings
)

np.save(
    FILES_FILE,
    valid_files
)

print(
    f"Saved: {EMBEDDINGS_FILE}"
)

print(
    f"Saved: {FILES_FILE}"
)