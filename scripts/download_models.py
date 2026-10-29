"""Download the embedding and reranker models ahead of time (used in the Docker build)."""

from sentence_transformers import CrossEncoder, SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_MODELS, RERANKER_MODEL


def main():
    SentenceTransformer(EMBEDDING_MODELS[EMBEDDING_MODEL]["name"], device="cpu")
    CrossEncoder(RERANKER_MODEL, device="cpu")
    print("Models downloaded.")


if __name__ == "__main__":
    main()