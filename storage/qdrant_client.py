from qdrant_client import QdrantClient
from storage.config import QDRANT_HOST, QDRANT_PORT

_client = None


def get_qdrant_client():
    """
    Returns a singleton Qdrant client.
    """

    global _client

    if _client is None:
        _client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )

    return _client