"""Central settings for StudyAI. Change values here, not in the code."""

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 3

# LLM
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0

# Embeddings
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
HF_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"