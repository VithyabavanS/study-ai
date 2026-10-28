"""Retrieval evaluation: does the retriever find the evidence passage?"""

import json
from pathlib import Path

from check_questions import DOCS_DIR, QUESTIONS_FILE, normalize
from config import CHUNK_OVERLAP, CHUNK_SIZE
from document_processor import DocumentProcessor
from vector_store import VectorStore

K = 5
RESULTS_FILE = Path("eval/results/retrieval.json")


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


def build_store(doc_name):
    processor = DocumentProcessor(
        LocalFile(DOCS_DIR / doc_name), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    store = VectorStore(use_openai=False)
    store.create_vectorstore(processor.process())
    return store


def main():
    questions = [q for q in load_questions() if q["evidence"]]
    stores = {}
    rows = []

    for q in questions:
        if q["doc"] not in stores:
            stores[q["doc"]] = build_store(q["doc"])
        results = stores[q["doc"]].similarity_search(q["question"], k=K)
        rank = find_rank(results, q["evidence"])
        rows.append({"id": q["id"], "rank": rank})
        print(f"{q['id']}: rank {rank if rank else '-'}")

    n = len(rows)
    summary = {
        "questions": n,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "hit@1": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 1) / n, 3),
        "hit@3": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 3) / n, 3),
        "hit@5": round(sum(1 for r in rows if r["rank"] and r["rank"] <= 5) / n, 3),
        "mrr@5": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3),
    }
    print(json.dumps(summary, indent=2))

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps({"summary": summary, "per_question": rows}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()