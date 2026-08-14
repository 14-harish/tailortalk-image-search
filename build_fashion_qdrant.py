import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

EMBEDDINGS_FILE = "data/fashion_embeddings.npy"
FILES_FILE = "data/fashion_valid_files.npy"

COLLECTION_NAME = "sarees_fashion"

print("Loading FashionCLIP embeddings...")

embeddings = np.load(
    EMBEDDINGS_FILE
).astype("float32")

valid_files = np.load(
    FILES_FILE,
    allow_pickle=True
)

print("Embeddings:", embeddings.shape)
print("Files:", len(valid_files))

print("Loading Qdrant...")

client = QdrantClient(
    path="data/qdrant"
)

if client.collection_exists(COLLECTION_NAME):

    print(
        f"Deleting existing '{COLLECTION_NAME}'..."
    )

    client.delete_collection(
        COLLECTION_NAME
    )

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=embeddings.shape[1],
        distance=Distance.COSINE
    )
)

print(
    f"Created collection: {COLLECTION_NAME}"
)

points = []

for i, (embedding, filename) in enumerate(
    zip(embeddings, valid_files)
):

    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "filename": str(filename)
            }
        )
    )

print(
    f"Uploading {len(points)} vectors..."
)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

info = client.get_collection(
    COLLECTION_NAME
)

print(
    "Vectors in collection:",
    info.points_count
)

print("\nFASHIONCLIP QDRANT SETUP COMPLETE")
