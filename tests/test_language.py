from langchain_core.documents import Document

from language import detect_language
from prompts import NOT_FOUND, build_prompt


def test_english_question():
    assert detect_language("Why are sinusoidal encodings used?") == "English"


def test_tamil_question():
    assert detect_language("ஒளிச்சேர்க்கை என்றால் என்ன?") == "Tamil"


def test_code_mixed_question_mostly_tamil():
    assert detect_language("Transformer-இல் attention எப்படி வேலை செய்கிறது?") == "Tamil"


def test_prompt_names_the_answer_language():
    chunks = [Document(page_content="Some context")]
    assert "Write your answer in Tamil" in build_prompt("இது என்ன?", chunks)
    assert "Write your answer in English" in build_prompt("What is this?", chunks)


def test_not_found_sentence_matches_question_language():
    chunks = [Document(page_content="Some context")]
    assert NOT_FOUND["Tamil"] in build_prompt("இது என்ன?", chunks)
    assert NOT_FOUND["English"] in build_prompt("What is this?", chunks)
    assert NOT_FOUND["English"] not in build_prompt("இது என்ன?", chunks)