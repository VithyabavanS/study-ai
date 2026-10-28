"""Central settings for StudyAI. Change values here, not in the code."""

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 3
RETRIEVAL_MODE = "vector"  # "vector", "bm25" or "hybrid"
CANDIDATES = 20            # results fetched from each retriever before fusion
RRF_K = 60                 # standard RRF constant

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 10     # chunks passed to the reranker

# LLM
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0

# Embeddings
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"