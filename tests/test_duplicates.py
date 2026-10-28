from langchain_core.documents import Document

from chat_engine import ChatEngine
from vector_store import VectorStore

SENTENCE = "CBAM attention improves the ResNet50 model accuracy on MRI scans."


class FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, question, k=3):
        return self.chunks[:k]


def test_free_mode_does_not_repeat_sentences(monkeypatch):
    """Cause A: the same sentence inside two overlapping chunks."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    chunks = [
        Document(page_content=f"Intro text about the study. {SENTENCE}"),
        Document(page_content=f"{SENTENCE} Further results follow here."),
    ]
    engine = ChatEngine(FakeVectorStore(chunks))

    answer = engine.get_answer("What does CBAM attention improve?")["answer"]

    assert answer.count(SENTENCE.rstrip(".")) == 1


def test_processing_twice_does_not_duplicate_chunks():
    """Cause B: indexing the same document twice in one app session."""
    chunks = [
        Document(page_content="Python is a programming language"),
        Document(page_content="Machine learning uses algorithms to learn from data"),
        Document(page_content="Neural networks are inspired by the human brain"),
    ]
    VectorStore(use_openai=False).create_vectorstore(chunks)
    second = VectorStore(use_openai=False)
    second.create_vectorstore(chunks)

    results = second.similarity_search("Tell me about AI", k=3)
    texts = [doc.page_content for doc in results]

    assert len(texts) == len(set(texts))