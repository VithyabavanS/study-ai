import pytest
from langchain_core.documents import Document

from hybrid import BM25Retriever
from retrieval import build_retriever


def test_bm25_mode_builds_bm25_retriever():
    chunks = [Document(page_content="Adam optimizer")]
    assert isinstance(build_retriever(chunks, mode="bm25"), BM25Retriever)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_retriever([], mode="magic")