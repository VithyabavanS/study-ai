"""
Quick test for DocumentProcessor
Run this to verify your class works!
"""

import sys
sys.path.append('src')  # Add src to path so we can import

from document_processor import DocumentProcessor

def test_class_creation():
    """Test if we can create a DocumentProcessor object"""
    print("Testing class creation...")
    
    # Create a mock file object
    class MockFile:
        def __init__(self):
            self.name = "test.pdf"
            self.size = 1024
        
        def read(self):
            return b"Mock PDF content"
    
    mock_file = MockFile()
    
    # Create processor
    processor = DocumentProcessor(mock_file, chunk_size=500, chunk_overlap=50)
    
    # Check attributes
    assert processor.file == mock_file
    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 50
    
    print("✅ Class creation works!")
    print(f"   - File: {processor.file.name}")
    print(f"   - Chunk size: {processor.chunk_size}")
    print(f"   - Overlap: {processor.chunk_overlap}")

if __name__ == "__main__":
    test_class_creation()