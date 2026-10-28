"""Keyword (BM25) and hybrid (BM25 + vector) retrieval."""

import string
import unicodedata

from rank_bm25 import BM25Okapi

from config import CANDIDATES, RRF_K, TOP_K


def tokenize(text):
    """Lowercase, split on whitespace and strip punctuation.

    Whitespace splitting (instead of a \\w regex) keeps Tamil words intact,
    because Tamil vowel signs are not 'word characters' in Python regex.
    """
    text = unicodedata.normalize("NFC", text)
    tokens = (token.strip(string.punctuation) for token in text.lower().split())
    
    return [token for token in tokens if token]


class BM25Retriever:
    """Keyword search over chunks using BM25."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])

    def similarity_search(self, question, k=TOP_K):
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]


def reciprocal_rank_fusion(result_lists, rrf_k=RRF_K):
    """Merge several ranked lists; chunks ranked high in many lists win."""
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered]


class HybridRetriever:
    """Runs vector search and BM25, then fuses the rankings with RRF."""

    def __init__(self, vector_store, chunks, candidates=CANDIDATES):
        self.vector_store = vector_store
        self.bm25 = BM25Retriever(chunks)
        self.candidates = candidates

    def similarity_search(self, question, k=TOP_K):
        vector_results = self.vector_store.similarity_search(question, k=self.candidates)
        bm25_results = self.bm25.similarity_search(question, k=self.candidates)
        return reciprocal_rank_fusion([vector_results, bm25_results])[:k]