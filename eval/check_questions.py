"""Check that every evidence phrase appears in the extracted PDF text."""

import json
import re
from pathlib import Path

import pymupdf

DOCS_DIR = Path("data/eval_docs")
QUESTIONS_FILE = Path("eval/questions.jsonl")


def normalize(text):
    """Lowercase and collapse all whitespace, so line breaks don't matter."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_text(doc_name):
    with pymupdf.open(DOCS_DIR / doc_name) as pdf:
        return normalize(" ".join(page.get_text() for page in pdf))


def main():
    texts = {}
    problems = 0
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        if q["evidence"] is None:
            print(f"{q['id']}: unanswerable (skipped)")
            continue
        if q["doc"] not in texts:
            texts[q["doc"]] = load_text(q["doc"])
        if normalize(q["evidence"]) in texts[q["doc"]]:
            print(f"{q['id']}: OK")
        else:
            print(f"{q['id']}: NOT FOUND -> {q['evidence']}")
            problems += 1
    print(f"\n{problems} problem(s)")


if __name__ == "__main__":
    main()