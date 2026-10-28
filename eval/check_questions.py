"""Check that every evidence phrase appears in the extracted PDF text.

Usage: python eval/check_questions.py [questions_file]
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf

DOCS_DIR = Path("data/eval_docs")
DEFAULT_QUESTIONS = Path("eval/questions.jsonl")


def normalize(text):
    """NFC-normalize, lowercase and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def load_text(doc_name):
    with pymupdf.open(DOCS_DIR / doc_name) as pdf:
        return normalize(" ".join(page.get_text() for page in pdf))


def main():
    questions_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_QUESTIONS
    texts = {}
    problems = 0
    for line in questions_file.read_text(encoding="utf-8").splitlines():
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