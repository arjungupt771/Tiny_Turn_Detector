from sentence_transformers import SentenceTransformer
from storage.config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model

def embed(text: str) -> list[float]:
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch-encodes so a 250-requirement policy doesn't make 250
    separate model calls."""
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return vectors.tolist()