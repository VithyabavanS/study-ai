"""Sentence-transformer embeddings with optional query/passage prefixes (e.g. for e5)."""

from langchain_core.embeddings import Embeddings


class PrefixedEmbeddings(Embeddings):
    """Embeds documents and queries, adding the prefixes some models were trained with."""

    def __init__(self, model_name, device="cpu", query_prefix="", passage_prefix="", model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device)
        self.model = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts):
        inputs = [self.passage_prefix + text for text in texts]
        return self.model.encode(inputs, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()