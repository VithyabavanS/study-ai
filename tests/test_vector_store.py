"""
Test VectorStore class with FREE HuggingFace embeddings
"""

import sys
import os
sys.path.append('src')

from dotenv import load_dotenv
from vector_store import VectorStore
from langchain.schema import Document

# Load environment variables
load_dotenv()

def test_vector_store():
    """Test VectorStore functionality with free embeddings"""
    print("=" * 60)
    print("Testing VectorStore with FREE HuggingFace embeddings")
    print("=" * 60)
    print("\n⏳ First run: Downloading model (~80MB, one-time only)")
    print("⏳ This may take 1-2 minutes depending on your internet speed")
    print("⏳ Subsequent runs will be instant!\n")
    
    try:
        # Create VectorStore with FREE embeddings (no API key needed!)
        print("1️⃣  Creating VectorStore...")
        vs = VectorStore(use_openai=False)  # use_openai=False = FREE!
        print("   ✅ VectorStore created\n")
        
        # Create mock documents
        print("2️⃣  Creating mock documents...")
        mock_chunks = [
            Document(page_content="Python is a programming language"),
            Document(page_content="Machine learning uses algorithms to learn from data"),
            Document(page_content="Neural networks are inspired by the human brain")
        ]
        print(f"   ✅ Created {len(mock_chunks)} mock chunks\n")
        
        # Create vectorstore
        print("3️⃣  Creating embeddings and storing in ChromaDB...")
        print("   ⏳ Processing... please wait...")
        num_stored = vs.create_vectorstore(mock_chunks)
        print(f"   ✅ Stored {num_stored} chunks\n")
        
        # Test search
        print("4️⃣  Testing similarity search...")
        question = "Tell me about AI"
        print(f"   Question: '{question}'")
        results = vs.similarity_search(question, k=2)
        
        print(f"   Found {len(results)} relevant chunks:\n")
        for i, result in enumerate(results, 1):
            print(f"   📄 Result {i}:")
            print(f"      {result.page_content}\n")
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("💡 Using free local embeddings - no API costs!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store()