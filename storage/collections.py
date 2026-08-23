from qdrant_client.models import Distance, VectorParams

from storage.qdrant_client import get_qdrant_client
from storage.config import REQUIREMENTS_COLLECTION, VECTOR_SIZE


def create_collections():
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    if REQUIREMENTS_COLLECTION not in existing:
        client.create_collection(
            collection_name=REQUIREMENTS_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {REQUIREMENTS_COLLECTION}")
    else:
        print(f"Collection already exists: {REQUIREMENTS_COLLECTION}")


        