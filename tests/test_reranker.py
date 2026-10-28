from langchain_core.documents import Document

from reranker import RerankingRetriever


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, question, k=3):
        return self.docs[:k]


class FakeCrossEncoder:
    """Scores a chunk by how many question words it contains."""

    def predict(self, pairs):
        return [
            sum(word in chunk.lower() for word in question.lower().split())
            for question, chunk in pairs
        ]


def test_reranker_moves_best_chunk_to_top():
    docs = [Document(page_content="Training data"), Document(page_content="the adam optimizer was used")]
    retriever = RerankingRetriever(FakeRetriever(docs), model=FakeCrossEncoder())

    results = retriever.similarity_search("which optimizer was used", k=1)

    assert results[0].page_content == "the adam optimizer was used"


def test_reranker_handles_no_candidates():
    retriever = RerankingRetriever(FakeRetriever([]), model=FakeCrossEncoder())
    assert retriever.similarity_search("anything") == []