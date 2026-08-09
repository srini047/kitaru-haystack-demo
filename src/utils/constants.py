# Embedder
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
SPARSE_EMBEDDING_MODEL = "prithivida/Splade_PP_en_v2"
EMBEDDING_DIM = 1024

# Qdrant
QDRANT_INDEX = "kitaru_documents"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Nice GUI
HEALTH_CHECK_INTERVAL = 5.0  # seconds
