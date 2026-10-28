"""Question answering: retrieve chunks, build a prompt, generate an answer."""

import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from config import LLM_MODEL, LLM_TEMPERATURE, TOP_K
from exceptions import LLMError
from prompts import build_prompt

NOT_FOUND_MESSAGE = "I couldn't find relevant information in the document."
STOP_WORDS = {"what", "is", "the", "a", "an", "how", "when", "where", "why", "who", "about"}


class ChatEngine:
    """Answers questions about a document using retrieval + an LLM.

    Uses Gemini when GEMINI_API_KEY is set; otherwise falls back to a free
    keyword-based extractive mode.
    """

    def __init__(self, vectorstore, api_key=None):
        self.vectorstore = vectorstore
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None

    @property
    def mode(self):
        return "gemini" if self.client else "free"

    def get_answer(self, question, k=TOP_K):
        """Retrieve relevant chunks once, then answer from them."""
        chunks = self.vectorstore.similarity_search(question, k=k)
        if not chunks:
            return {"answer": NOT_FOUND_MESSAGE, "sources": []}

        if self.client:
            answer = self._generate(question, chunks)
        else:
            answer = self._extractive_answer(question, chunks)

        return {"answer": answer, "sources": chunks}

    def _generate(self, question, chunks):
        """Send the RAG prompt to Gemini and return its answer."""
        prompt = build_prompt(question, chunks)
        try:
            response = self.client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=LLM_TEMPERATURE),
            )
        except genai_errors.APIError as e:
            raise LLMError(f"Gemini request failed: {e}") from e

        if not response.text:
            raise LLMError("Gemini returned an empty response.")
        return response.text

    def _extractive_answer(self, question, chunks):
        """Free mode: return the sentences that best match the question's keywords."""
        sentences = []
        for chunk in chunks:
            sentences.extend(self._key_sentences(chunk.page_content, question, max_sentences=2))

        sentences = list(dict.fromkeys(sentences))  # remove duplicates, keep order

        if not sentences:
            return NOT_FOUND_MESSAGE

        lines = [f"{i}. {s}" for i, s in enumerate(sentences[:4], start=1)]
        return (
            "**Based on your document:**\n\n"
            + "\n\n".join(lines)
            + "\n\n---\n💡 *Free mode (keyword extraction). Add GEMINI_API_KEY to .env for AI answers.*"
        )

    @staticmethod
    def _key_sentences(text, question, max_sentences=3):
        """Score sentences by how many question keywords they contain."""
        raw = text.replace("?", ".").replace("!", ".").split(".")
        sentences = [s.strip() for s in raw if len(s.strip()) > 20]
        keywords = set(question.lower().split()) - STOP_WORDS

        scored = []
        for sentence in sentences:
            score = sum(1 for keyword in keywords if keyword in sentence.lower())
            if score > 0:
                scored.append((score, sentence))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [sentence for _, sentence in scored[:max_sentences]]