# Smart Study Buddy 📚

An AI-powered study assistant that lets you chat with your documents using RAG (Retrieval-Augmented Generation).

## Features
- Upload PDF documents
- Ask questions about your notes
- Get accurate answers with context

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your OpenAI API key
6. Run: `streamlit run src/app.py`

## Tech Stack
- Streamlit (UI)
- LangChain (RAG framework)
- ChromaDB (Vector database)
- OpenAI (LLM & Embeddings)