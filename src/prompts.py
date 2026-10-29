"""Prompt construction for the RAG pipeline."""

from language import detect_language

NOT_FOUND = {
    "English": "I couldn't find this in the document.",
    "Tamil": "இந்தத் தகவல் ஆவணத்தில் இல்லை.",
}

INSTRUCTIONS = """Answer the question using ONLY the context below.
If the answer is not in the context, reply with exactly: "{not_found}" and add no citations.
Otherwise, cite the sources you used, like [1] or [2].
Write your answer in {language}, even if the context is in a different language."""


def build_prompt(question, chunks):
    """Build the full prompt from a question and retrieved chunks."""
    numbered = []
    for number, chunk in enumerate(chunks, start=1):
        numbered.append(f"[{number}] {chunk.page_content}")

    context = "\n\n".join(numbered)
    language = detect_language(question)
    instructions = INSTRUCTIONS.format(language=language, not_found=NOT_FOUND[language])

    return f"""{instructions}

Context:
{context}

Question: {question}
Answer:"""