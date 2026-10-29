"""Vector store: embeds chunks and searches them with ChromaDB."""

import uuid

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

from config import EMBEDDING_DEVICE, EMBEDDING_MODEL, EMBEDDING_MODELS, TOP_K
from embeddings import PrefixedEmbeddings
from exceptions import VectorStoreError


def load_embeddings(key=EMBEDDING_MODEL):
    """Create the embedding model configured under EMBEDDING_MODELS[key]."""
    spec = EMBEDDING_MODELS[key]
    return PrefixedEmbeddings(
        spec["name"],
        device=EMBEDDING_DEVICE,
        query_prefix=spec["query_prefix"],
        passage_prefix=spec["passage_prefix"],
    )


class VectorStore:
    """Dense (embedding) search over document chunks."""

    def __init__(self, embeddings=None):
        self.embeddings = embeddings or load_embeddings()
        self.vectorstore = None

    def create_vectorstore(self, chunks):
        """Embed the chunks into a fresh ChromaDB collection; returns the chunk count."""
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=f"studyai-{uuid.uuid4().hex}",
                client_settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e:
            raise VectorStoreError(f"Error creating vector store: {e}") from e
        return len(chunks)

    def similarity_search(self, question, k=TOP_K):
        if self.vectorstore is None:
            raise VectorStoreError("Create the vector store before searching.")
        try:
            return self.vectorstore.similarity_search(question, k=k)
        except Exception as e:
            raise VectorStoreError(f"Error during similarity search: {e}") from e