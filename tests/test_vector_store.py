"""Tests for VectorStore using free local HuggingFace embeddings."""

from langchain_core.documents import Document

from vector_store import VectorStore


def test_vector_store_stores_and_retrieves_relevant_chunks():
    vs = VectorStore(use_openai=False)
    chunks = [
        Document(page_content="Python is a programming language"),
        Document(page_content="Machine learning uses algorithms to learn from data"),
        Document(page_content="Neural networks are inspired by the human brain"),
    ]

    num_stored = vs.create_vectorstore(chunks)
    results = vs.similarity_search("Tell me about AI", k=2)

    assert num_stored == 3
    assert len(results) == 2
    assert "Python" not in results[0].page_content