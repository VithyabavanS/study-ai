"""Builds the retriever selected in config: vector, bm25, hybrid or hybrid_rerank."""

from config import RETRIEVAL_MODE
from hybrid import BM25Retriever, HybridRetriever
from reranker import RerankingRetriever
from vector_store import VectorStore

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def build_retriever(chunks, mode=RETRIEVAL_MODE, reranker_model=None):
    """Return a retriever with a similarity_search(question, k) method."""
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode: {mode!r}. Choose from {MODES}.")
    if mode == "bm25":
        return BM25Retriever(chunks)

    store = VectorStore(use_openai=False)
    store.create_vectorstore(chunks)
    if mode == "vector":
        return store

    hybrid = HybridRetriever(store, chunks)
    if mode == "hybrid":
        return hybrid
    return RerankingRetriever(hybrid, model=reranker_model)