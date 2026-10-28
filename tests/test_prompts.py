from langchain_core.documents import Document

from prompts import build_prompt

CHUNKS = [
    Document(page_content="Alz-InsightNet uses CBAM attention."),
    Document(page_content="Grad-CAM shows important image regions."),
]


def test_prompt_contains_question():
    prompt = build_prompt("What is CBAM?", CHUNKS)
    assert "What is CBAM?" in prompt


def test_prompt_numbers_each_chunk():
    prompt = build_prompt("What is CBAM?", CHUNKS)
    assert "[1] Alz-InsightNet uses CBAM attention." in prompt
    assert "[2] Grad-CAM shows important image regions." in prompt


def test_prompt_restricts_answer_to_context():
    prompt = build_prompt("What is CBAM?", CHUNKS)
    assert "only" in prompt.lower()