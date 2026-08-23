from pathlib import Path
# from embeddings import get_embedding_model

# Local Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Collection Name
# POLICY_COLLECTION = "gbvh_policies"
REQUIREMENTS_COLLECTION = "gbvh_requirements"

UPLOADED_FILES = "uploaded_policies"

POLICY_INFO = "policy_info"

POLICY_ALIGNED = "policy_aligned_data"

# Embedding model
EMBEDDING_MODEL = "/home/itm/models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

# Vector Size
VECTOR_SIZE = 384
# VECTOR_SIZE = get_embedding_model().get_sentence_embedding_dimension()


KEEP_LOCAL_PDF_COPY = True

# Qdrant payloads aren't built for large blobs. Only base64-embed PDFs
# under this size; larger ones just get metadata about the file.
MAX_PDF_PAYLOAD_BYTES = 8 * 1024 * 1024  # 8 MB