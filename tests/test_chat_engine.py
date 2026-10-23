from langchain_core.documents import Document

from chat_engine import NOT_FOUND_MESSAGE, ChatEngine


class FakeVectorStore:
    """Stands in for the real vector store so tests run fast and offline."""

    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, question, k=3):
        return self.chunks[:k]


def test_free_mode_answers_from_chunks(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    chunks = [Document(page_content="CBAM attention improves the ResNet50 model accuracy.")]
    engine = ChatEngine(FakeVectorStore(chunks))

    result = engine.get_answer("What does CBAM attention improve?")

    assert engine.mode == "free"
    assert "CBAM" in result["answer"]
    assert result["sources"] == chunks


def test_no_chunks_returns_not_found(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    engine = ChatEngine(FakeVectorStore([]))

    result = engine.get_answer("Anything?")

    assert result["answer"] == NOT_FOUND_MESSAGE
    assert result["sources"] == []