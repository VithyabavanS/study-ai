"""
Vector Store Module

Handles embedding generation and vector similarity search using ChromaDB.
This is the core of the RAG (Retrieval-Augmented Generation) system.
"""

# ============================================================================
# IMPORTS
# ============================================================================
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Updated!
from langchain_community.vectorstores import Chroma
import os


# ============================================================================
# VECTOR STORE CLASS
# ============================================================================
class VectorStore:
    """
    Manages vector embeddings and similarity search.
    
    Supports both OpenAI (paid) and HuggingFace (free) embeddings.
    
    Attributes:
        api_key: OpenAI API key (optional, for paid embeddings)
        use_openai: Whether to use OpenAI or HuggingFace embeddings
        embeddings: Embeddings model instance
        vectorstore: ChromaDB vectorstore instance
    """
    
    def __init__(self, api_key=None, use_openai=False):
        """
        Initialize the vector store.
        
        Args:
            api_key: OpenAI API key (only needed if use_openai=True)
            use_openai: If True, use OpenAI embeddings. If False, use HuggingFace (free)
        """
        self.api_key = api_key
        self.use_openai = use_openai
        
        if use_openai and api_key:
            # Use OpenAI embeddings (paid, high quality)
            print("Using OpenAI embeddings...")
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model="text-embedding-3-small"
            )
        else:
            # Use HuggingFace embeddings (free, good quality, runs locally)
            print("Using HuggingFace embeddings (free, local)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  # Fast, good quality
                model_kwargs={'device': 'cpu'},  # Use CPU (no GPU needed)
                encode_kwargs={'normalize_embeddings': True}  # Better similarity
            )
        
        # Will store the ChromaDB vectorstore (created when we add documents)
        self.vectorstore = None
    
    
    def create_vectorstore(self, chunks):
        """
        Create a vector store from document chunks.
        
        Converts each chunk to embeddings and stores in ChromaDB.
        
        Args:
            chunks: List of document chunks from DocumentProcessor
            
        Returns:
            Number of chunks stored
        """
        try:
            print(f"Creating embeddings for {len(chunks)} chunks...")
            
            # Create ChromaDB vectorstore from document chunks
            # This does 2 things:
            # 1. Converts each chunk to embeddings using self.embeddings
            # 2. Stores embeddings + text in ChromaDB for fast search
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                # persist_directory="./chroma_db"  # Uncomment to save to disk
            )
            
            print(f"✅ Successfully created vector store with {len(chunks)} chunks")
            return len(chunks)
        
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    
    def similarity_search(self, question, k=3):
        """
        Search for chunks most similar to the question.
        
        Args:
            question: User's question (string)
            k: Number of most relevant chunks to return
            
        Returns:
            List of k most relevant document chunks
        """
        # Check if vectorstore exists
        if not self.vectorstore:
            raise ValueError("Vector store not created yet. Call create_vectorstore first.")
        
        try:
            # Perform similarity search
            # This converts question to vector, then finds k nearest vectors
            results = self.vectorstore.similarity_search(
                query=question,
                k=k  # Return top k most similar chunks
            )
            
            return results
        
        except Exception as e:
            raise Exception(f"Error during similarity search: {str(e)}")
    
    
    def get_vectorstore(self):
        """
        Get the vectorstore instance.
        
        Returns:
            ChromaDB vectorstore instance or None
        """
        return self.vectorstore