# Project Summary: StudyAI - Intelligent PDF Analyser

This document provides a comprehensive overview of the StudyAI project, its architecture, technologies, and implementation details.

## 1. High-Level Goal

**StudyAI** is a web application designed to be an intelligent study and research assistant. Its core function is to allow users to "chat" with their PDF documents. A user uploads a PDF, and the application enables them to ask questions in plain English. The system then finds the most relevant information within that document and generates a concise, accurate answer, complete with source references from the document itself.

## 2. Core Concept: Retrieval-Augmented Generation (RAG)

The project is built on a **Retrieval-Augmented Generation (RAG)** architecture. This solves the problem of Large Language Models (LLMs) not having knowledge of private, user-specific documents.

The RAG process involves two main phases:

1.  **Indexing (Processing the Document)**: The system reads a user's document and creates a searchable "knowledge base" from it. This is a one-time process per document.
2.  **Retrieval & Generation (Answering a Question)**: When a user asks a question, the system first **retrieves** the most relevant snippets of information from the knowledge base. It then **augments** an LLM's prompt with this retrieved context and asks it to **generate** an answer based *only* on that provided information. This ensures answers are factual and grounded in the document.

## 3. Step-by-Step Workflow

### Phase 1: Indexing the Document (Triggered by "Process Document")

**Step 1: Document Loading (`document_processor.py`)**
-   The uploaded PDF file is loaded using `PyPDFLoader` from the LangChain library, which extracts text from all pages.

**Step 2: Text Splitting (Chunking) (`document_processor.py`)**
-   The extracted text is split into smaller, overlapping chunks (e.g., 1000 characters with a 200-character overlap) using `RecursiveCharacterTextSplitter`.
-   The overlap is crucial to maintain semantic context between chunks.

**Step 3: Embedding Generation (`vector_store.py`)**
-   Each text chunk is converted into a numerical vector representation called an **embedding**. These embeddings capture the semantic meaning of the text.
-   The application supports two modes for this step:
    -   **Free Mode**: Uses `HuggingFaceEmbeddings` with the `"all-MiniLM-L6-v2"` model. This runs locally on the CPU and is free.
    -   **AI Mode (Paid)**: Uses `OpenAIEmbeddings` with the `"text-embedding-3-small"` model via an API call to OpenAI. This offers higher quality but requires an API key.

**Step 4: Storing in a Vector Database (`vector_store.py`)**
-   The text chunks and their corresponding embeddings are stored in a `ChromaDB` in-memory vector database.
-   ChromaDB is optimized for extremely fast similarity searches on these embedding vectors.

### Phase 2: Generating an Answer (Triggered by "Get Answer")

**Step 5: Querying and Retrieval (`chat_engine.py`)**
-   The user's question is converted into an embedding using the same model from Step 3.
-   The system queries ChromaDB to find the top `k` (e.g., 3) text chunks whose embeddings are most semantically similar to the question's embedding. This is the "retrieval" step.

**Step 6: Answer Generation (`chat_engine.py`)**
-   The retrieved chunks (context) and the user's original question are combined into a prompt.
-   The generation process differs based on the mode:
    -   **AI Mode (Generative)**:
        -   The prompt, context, and question are sent to an OpenAI LLM (`gpt-3.5-turbo`) using the `RetrievalQA` chain from LangChain.
        -   The LLM **synthesizes** the information to generate a new, fluent, and natural-sounding answer.
    -   **Free Mode (Extractive)**:
        -   No LLM is used. A custom Python function (`_extract_key_sentences`) runs locally.
        -   It **extracts** the most relevant sentences from the retrieved context based on keyword overlap with the question. The answer is a formatted list of these sentences.

**Step 7: Displaying the Result (`app.py`)**
-   The final answer and the source chunks are displayed in the Streamlit UI.

## 4. Technology Stack & Purpose

-   **Streamlit**: The frontend web framework used to build the entire interactive user interface in Python.
-   **LangChain**: The primary AI orchestration library.
    -   `PyPDFLoader`: For loading PDFs.
    -   `RecursiveCharacterTextSplitter`: For splitting text into chunks.
    -   `OpenAIEmbeddings` & `HuggingFaceEmbeddings`: Standardized interfaces for embedding models.
    -   `Chroma`: Wrapper for the ChromaDB vector database.
    -   `RetrievalQA`: A pre-built chain for the retrieve-and-generate process.
    -   `ChatOpenAI`: Wrapper for the OpenAI `gpt-3.5-turbo` model.
    -   `PromptTemplate`: For creating custom instructions for the LLM.
-   **ChromaDB**: The in-memory vector database for storing embeddings and performing similarity searches.
-   **HuggingFace Sentence Transformers**: Provides the free, local embedding model (`all-MiniLM-L6-v2`).
-   **OpenAI API**: Provides the paid, high-performance models:
    -   `text-embedding-3-small`: For embeddings.
    -   `gpt-3.5-turbo`: For answer generation.
-   **PyPDF**: The underlying library used by `PyPDFLoader` for PDF text extraction.
-   **python-dotenv**: A utility for loading secrets (like an `OPENAI_API_KEY`) from a `.env` file.

## 5. Code Structure (`src/` directory)

The code is modularized for clarity and maintainability:

-   `app.py`: The main application entry point. Contains all Streamlit UI code and orchestrates calls to the other modules.
-   `document_processor.py`: Contains the `DocumentProcessor` class, responsible for loading and chunking the PDF.
-   `vector_store.py`: Contains the `VectorStore` class, responsible for creating embeddings and managing the ChromaDB database.
-   `chat_engine.py`: Contains the `ChatEngine` class, responsible for retrieving context and generating the final answer for both AI and Free modes.
