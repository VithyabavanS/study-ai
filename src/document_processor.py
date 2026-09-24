"""
Document Processor Module

This module handles loading and processing PDF documents.
It extracts text and splits it into chunks for RAG processing.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # Updated

from exceptions import DocumentProcessingError

# Hint: We need PyPDFLoader and RecursiveCharacterTextSplitter from langchain


# ============================================================================
# DOCUMENT PROCESSOR CLASS
# ============================================================================
class DocumentProcessor:
    """
    Handles PDF document loading and text processing.
    
    Attributes:
        file: The uploaded PDF file object
        chunk_size: Size of text chunks (in characters)
        chunk_overlap: Overlap between chunks (to maintain context)
    """
    
    def __init__(self, file, chunk_size=1000, chunk_overlap=200):
        """
        Initialize the document processor.
        
        Args:
            file: Uploaded file object from Streamlit
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        # TODO: Store the parameters as instance variables
        self.file = file  # Store the uploaded file
        self.chunk_size = chunk_size  # How big each chunk should be
        self.chunk_overlap = chunk_overlap  # How much chunks should overlap
        self.temp_file_path = None  # Will store temporary file path

    
    
    def load_pdf(self):
        """
        Load PDF file and extract text.
        
        Returns:
            List of document objects containing page content
        """
         # Create a temporary file to save the uploaded PDF
        # (PyPDFLoader needs a file path, not a file object)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write the uploaded file's content to temp file
            tmp_file.write(self.file.read())
            self.temp_file_path = tmp_file.name  # Store path for later cleanup
        
        # Load the PDF using LangChain's PyPDFLoader
        loader = PyPDFLoader(self.temp_file_path)
        documents = loader.load()  # Extract text from all pages
        
        return documents
    
    
    def split_into_chunks(self, documents):
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of document objects from load_pdf()
            
        Returns:
            List of text chunks
        """
        # Create text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # Maximum chunk size
            chunk_overlap=self.chunk_overlap,  # Overlap between chunks
            length_function=len,  # How to measure length (character count)
        )
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    
    
    def process(self):
        """
        Main processing pipeline: load PDF and split into chunks.
        
        Returns:
            List of processed text chunks ready for embedding
        """
        try:
            # Step 1: Load the PDF
            documents = self.load_pdf()
            
            # Step 2: Split into chunks
            chunks = self.split_into_chunks(documents)
            
            # Step 3: Clean up temporary file
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.unlink(self.temp_file_path)  # Delete temp file
            
            return chunks
    
        except Exception as e:
            # If anything goes wrong, clean up and raise error
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.unlink(self.temp_file_path)
            # raise Exception(f"Error processing document: {e!s}")
            raise DocumentProcessingError(f"Error processing document: {e}") from e