# 📚 StudyAI – Intelligent PDF Analyser

StudyAI is an intelligent web application that allows users to chat with their PDF documents. Upload a PDF, ask questions in plain English, and receive accurate, source-grounded answers directly from the document.

The project is built using a Retrieval-Augmented Generation (RAG) architecture, ensuring answers are factual, reliable, and strictly based on the uploaded document.

## 🚀 Key Features

- 📄 Upload and process PDF documents
- 💬 Ask natural language questions about the document
- 🔍 Semantic search over document content
- 🤖 Two modes:
  - **AI Mode** – Generative answers using OpenAI LLMs
  - **Free Mode** – Fully local, extractive answers (no API key required)
- 📌 Source references included with answers
- ⚡ Fast in-memory vector search using ChromaDB
- 🖥️ Simple and intuitive Streamlit interface

## 🧠 Core Architecture: Retrieval-Augmented Generation (RAG)

StudyAI uses a RAG pipeline to overcome the limitation of LLMs not knowing private documents.

### Phase 1: Document Indexing

- Load PDF and extract text
- Split text into overlapping semantic chunks
- Convert chunks into embeddings
- Store embeddings in a vector database

### Phase 2: Question Answering

- Embed the user's question
- Retrieve the most relevant chunks from the vector store
- Generate or extract an answer using retrieved context

This ensures answers are grounded in the document, not hallucinated.

## 🧩 Workflow Overview

### 📥 Document Processing

- PDF loaded using PyPDFLoader
- Text split using RecursiveCharacterTextSplitter
- Embeddings generated using:
  - **Free Mode:** all-MiniLM-L6-v2 (HuggingFace)
  - **AI Mode:** text-embedding-3-small (OpenAI)
- Stored in an in-memory ChromaDB vector store

### ❓ Question Answering

- Question embedded using the same model
- Top-K relevant chunks retrieved
- Answer generated:
  - **AI Mode:** OpenAI gpt-3.5-turbo via LangChain RetrievalQA
  - **Free Mode:** Keyword-based sentence extraction

## 🛠️ Technology Stack

| Category           | Tools                         |
| ------------------ | ----------------------------- |
| Frontend           | Streamlit                     |
| AI Orchestration   | LangChain                     |
| Vector Database    | ChromaDB                      |
| Embeddings (Free)  | HuggingFace all-MiniLM-L6-v2  |
| Embeddings (Paid)  | OpenAI text-embedding-3-small |
| LLM                | OpenAI gpt-3.5-turbo          |
| PDF Processing     | PyPDF                         |
| Environment Config | python-dotenv                 |

## 📁 Project Structure

```
src/
│
├── app.py                 # Streamlit UI and app orchestration
├── document_processor.py  # PDF loading and text chunking
├── vector_store.py        # Embedding generation & ChromaDB handling
├── chat_engine.py         # Retrieval and answer generation logic
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VithyabavanS/study-ai.git
cd studyai
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Configure OpenAI API

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

💡 If no API key is provided, StudyAI automatically runs in Free Mode.

### ▶️ Run the Application

```bash
streamlit run src/app.py
```

Open your browser and start chatting with your PDFs 🚀

## 🆓 Free Mode vs 🤖 AI Mode

| Feature           | Free Mode  | AI Mode             |
| ----------------- | ---------- | ------------------- |
| Cost              | 100% Free  | Requires OpenAI API |
| Runs Locally      | ✅         | ❌                  |
| Answer Type       | Extractive | Generative          |
| Answer Quality    | Good       | Excellent           |
| Internet Required | ❌         | ✅                  |

## 🎯 Use Cases

- Academic study & exam preparation
- Research paper analysis
- Technical documentation Q&A
- Legal and policy document review
- Quick understanding of long PDFs

## 🔮 Future Improvements

- Persistent vector storage
- Multi-document support
- Highlight answers directly on PDF pages
- Chat history per document
- User authentication

## 📜 License

This project is open-source and available under the MIT License.

## 🙌 Acknowledgements

- [LangChain](https://langchain.com)
- [OpenAI](https://openai.com)
- [HuggingFace](https://huggingface.co)
- [ChromaDB](https://www.trychroma.com)
- [Streamlit](https://streamlit.io)
