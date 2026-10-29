class StudyAIError(Exception):
    """Base error for all StudyAI failures."""


class DocumentProcessingError(StudyAIError):
    """Raised when a PDF cannot be loaded or split."""


class VectorStoreError(StudyAIError):
    """Raised when embedding, storing, or searching fails."""


class LLMError(StudyAIError):
    """Raised when the language model request fails."""

class SpeechError(StudyAIError):
    """Raised when the answer can't be read aloud."""