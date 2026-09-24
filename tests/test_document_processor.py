"""Tests for DocumentProcessor."""

import pytest

from document_processor import DocumentProcessor
from exceptions import DocumentProcessingError


class MockFile:
    """Fake uploaded file with invalid PDF content."""

    name = "test.pdf"
    size = 1024

    def read(self):
        return b"Mock PDF content"


def test_processor_stores_settings():
    mock_file = MockFile()
    processor = DocumentProcessor(mock_file, chunk_size=500, chunk_overlap=50)

    assert processor.file == mock_file
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 50


def test_invalid_pdf_raises_document_processing_error():
    processor = DocumentProcessor(MockFile(), chunk_size=500, chunk_overlap=50)

    with pytest.raises(DocumentProcessingError):
        # processor.process_document()
        processor.process()