"""
Document Processor Module

Loads PDF documents and splits their text into chunks for RAG processing.
"""

import contextlib
import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,  # PyMuPDF extracts Tamil correctly; pypdf reorders vowel signs
)

from config import CHUNK_OVERLAP, CHUNK_SIZE
from exceptions import DocumentProcessingError


class DocumentProcessor:
    """
    Handles PDF document loading and text processing.

    Attributes:
        file: The uploaded PDF file object
        chunk_size: Size of text chunks (in characters)
        chunk_overlap: Overlap between chunks (to maintain context)
    """

    def __init__(self, file, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """
        Initialize the document processor.

        Args:
            file: Uploaded file object (Streamlit upload or anything with read())
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.file = file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temp_file_path = None  # path of the temporary PDF copy

    def load_pdf(self):
        """
        Load the PDF file and extract its text.

        Returns:
            List of document objects, one per page
        """
        # PyMuPDFLoader needs a file path, so save the upload to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(self.file.read())
            self.temp_file_path = tmp_file.name

        # Load the PDF with PyMuPDF (correct Tamil text extraction)
        loader = PyMuPDFLoader(self.temp_file_path)
        return loader.load()

    def split_into_chunks(self, documents):
        """
        Split documents into smaller, overlapping chunks.

        Args:
            documents: List of document objects from load_pdf()

        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,  # measure length in characters
        )
        return text_splitter.split_documents(documents)

    def process(self):
        """Load the PDF and split it into chunks.

        Returns:
            List of text chunks ready for embedding.
        """
        try:
            documents = self.load_pdf()
            return self.split_into_chunks(documents)
        except Exception as e:
            raise DocumentProcessingError(f"Error processing document: {e}") from e
        finally:
            self._remove_temp_file()

    def _remove_temp_file(self):
        """Delete the temporary PDF copy.

        On Windows, PyMuPDF can keep the file locked after a failed parse, so a
        failed delete is ignored rather than allowed to hide the real error.
        """
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            with contextlib.suppress(OSError):
                os.unlink(self.temp_file_path)