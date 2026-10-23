"""Prompt construction for the RAG pipeline."""

INSTRUCTIONS = """Answer the question using ONLY the context below.
If the answer is not in the context, say: "I couldn't find this in the document."
Cite the sources you used, like [1] or [2].
Answer in the same language as the question."""


def build_prompt(question, chunks):
    """Build the full prompt from a question and retrieved chunks."""
    numbered = []
    for number, chunk in enumerate(chunks, start=1):
        numbered.append(f"[{number}] {chunk.page_content}")

    context = "\n\n".join(numbered)

    return f"""{INSTRUCTIONS}

Context:
{context}

Question: {question}
Answer:"""