"""Compare Tamil PDF text extraction between pypdf and PyMuPDF.

Usage: python eval/check_extraction.py data/eval_docs/photosynthesis_ta.pdf ஒளித்தொகுப்பு
"""

import sys

import pymupdf
from pypdf import PdfReader

CONSONANTS = {chr(c) for c in range(0x0B95, 0x0BBA)}
DEPENDENT_SIGNS = {chr(c) for c in range(0x0BBE, 0x0BCE)}  # vowel signs and pulli


def invalid_sign_rate(text):
    """Share of vowel signs/pulli not directly after a consonant (correct text: 0)."""
    positions = [i for i, ch in enumerate(text) if ch in DEPENDENT_SIGNS]
    bad = sum(1 for i in positions if i == 0 or text[i - 1] not in CONSONANTS)
    return bad / len(positions) if positions else 0.0


def extract(path):
    with pymupdf.open(path) as pdf:
        pymupdf_text = "\n".join(page.get_text() for page in pdf)
    pypdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return {"pypdf": pypdf_text, "pymupdf": pymupdf_text}


def main():
    path, words = sys.argv[1], sys.argv[2:]
    for name, text in extract(path).items():
        counts = {word: text.count(word) for word in words}
        print(f"{name:8} invalid vowel-sign positions: {invalid_sign_rate(text):.1%}  {counts}")


if __name__ == "__main__":
    main()