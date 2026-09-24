class StudyAIError(Exception):
    """Base error for all StudyAI failures."""


class DocumentProcessingError(StudyAIError):
    """Raised when a PDF cannot be loaded or split."""


class VectorStoreError(StudyAIError):
    """Raised when embedding, storing, or searching fails."""