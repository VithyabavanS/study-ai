"""Retrieval evaluation: does the retriever find the evidence passage?

Usage:
    python eval/eval_retrieval.py --mode vector|bm25|hybrid
"""

import argparse
import json
import time
from pathlib import Path

from check_questions import DOCS_DIR, QUESTIONS_FILE, normalize

from config import CHUNK_OVERLAP, CHUNK_SIZE, RETRIEVAL_MODE
from document_processor import DocumentProcessor
from hybrid import BM25Retriever, HybridRetriever
from reranker import RerankingRetriever
from vector_store import VectorStore

K = 5

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


class LocalFile:
    """Makes a file on disk look like a Streamlit upload."""

    def __init__(self, path):
        self.name = path.name
        self._data = path.read_bytes()
        self.size = len(self._data)

    def read(self):
        return self._data


def load_questions():
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def find_rank(results, evidence):
    """Return the 1-based rank of the first chunk containing the evidence, or None."""
    target = normalize(evidence)
    for rank, doc in enumerate(results, start=1):
        if target in normalize(doc.page_content):
            return rank
    return None


def build_store(doc_name, mode=RETRIEVAL_MODE):
    """Build the retriever for one document: vector, bm25 or hybrid."""
    processor = DocumentProcessor(
        LocalFile(DOCS_DIR / doc_name), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = processor.process()
    if mode == "bm25":
        return BM25Retriever(chunks)

    store = VectorStore(use_openai=False)
    store.create_vectorstore(chunks)
    
    if mode == "hybrid":
        return HybridRetriever(store, chunks)
    if mode == "hybrid_rerank":
        return RerankingRetriever(HybridRetriever(store, chunks))
    return store


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--mode", choices=MODES, default=RETRIEVAL_MODE)
    mode = parser.parse_args().mode

    questions = [q for q in load_questions() if q["evidence"]]
    stores = {}
    rows = []
    latencies = []

    for q in questions:
        if q["doc"] not in stores:
            stores[q["doc"]] = build_store(q["doc"], mode)

        start = time.perf_counter()
        results = stores[q["doc"]].similarity_search(q["question"], k=K)
        latencies.append(time.perf_counter() - start)

        rank = find_rank(results, q["evidence"])
        rows.append({"id": q["id"], "rank": rank})
        print(f"{q['id']}: rank {rank if rank else '-'}")

    n = len(rows)
    summary = {
        "mode": mode,
        "questions": n,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "hit@1": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 1) / n, 3),
        "hit@3": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 3) / n, 3),
        "hit@5": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 5) / n, 3),
        "mrr@5": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
        "avg_latency_ms": round(1000 * sum(latencies) / n),
    }
    print(json.dumps(summary, indent=2))

    results_file = Path(f"eval/results/retrieval_{mode}.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(
        json.dumps({"summary": summary, "per_question": rows}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()