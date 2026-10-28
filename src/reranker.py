"""Cross-encoder reranking on top of any retriever."""

from config import EMBEDDING_DEVICE, RERANK_CANDIDATES, RERANKER_MODEL, TOP_K


class RerankingRetriever:
    """Fetches candidates from a base retriever, then reorders them with a cross-encoder."""

    def __init__(self, base_retriever, model=None, candidates=RERANK_CANDIDATES):
        self.base = base_retriever
        self.candidates = candidates
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(RERANKER_MODEL, device=EMBEDDING_DEVICE)
        self.model = model

    def similarity_search(self, question, k=TOP_K):
        docs = self.base.similarity_search(question, k=self.candidates)
        if not docs:
            return []
        scores = self.model.predict([(question, doc.page_content) for doc in docs])
        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in ranked[:k]]