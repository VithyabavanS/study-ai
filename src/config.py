"""Central settings for StudyAI. Change values here, not in the code."""

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 3
RETRIEVAL_MODE = "hybrid_rerank"  # "vector", "bm25", "hybrid" or "hybrid_rerank"
CANDIDATES = 20            # results fetched from each retriever before fusion
RRF_K = 60                 # standard RRF constant

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_CANDIDATES = 10     # chunks passed to the reranker

# LLM
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TEMPERATURE = 0

# Embeddings
EMBEDDING_MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",  # English only
        "query_prefix": "",
        "passage_prefix": "",
    },
    "e5-small": {
        "name": "intfloat/multilingual-e5-small",  # ~100 languages incl. Tamil
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}
EMBEDDING_MODEL = "e5-small"
EMBEDDING_DEVICE = "cpu"