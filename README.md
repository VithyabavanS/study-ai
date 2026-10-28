# 📚 StudyAI – Intelligent PDF Analyser

StudyAI is a web application for asking questions about your PDF documents. Upload a PDF, ask a question in plain language, and get an answer grounded in the document, with citations to the passages it used.

It is built as a Retrieval-Augmented Generation (RAG) pipeline written from scratch (retrieve → build prompt → generate), and it is **evaluated**: retrieval and answer quality are measured on a hand-written test set (see [Evaluation](#-evaluation)).

## 🚀 Key Features

- 📄 Upload and process PDF documents
- 💬 Ask natural-language questions about the document
- 🔍 Semantic search over document chunks
- 🤖 Two modes:
  - **AI Mode**: generative answers with citations, using Google Gemini (free tier)
  - **Free Mode**: local, extractive answers (no API key required)
- 🛡️ Grounded answers: the model answers only from the retrieved text and says so when the document doesn't contain the answer
- 📌 Source passages shown with every answer
- 🧪 Tested (pytest) and evaluated (retrieval metrics + LLM-as-judge)

## 🧠 Architecture: Retrieval-Augmented Generation (RAG)

### Phase 1: Document indexing

1. Load the PDF and extract its text
2. Split the text into overlapping chunks
3. Convert each chunk into an embedding
4. Store the embeddings in a vector database (ChromaDB)

### Phase 2: Question answering

```
question
   │
   ▼
1. RETRIEVE      BM25 + vector search, fused with RRF, then reranked by a cross-encoder
   │
   ▼
2. BUILD PROMPT  instructions + numbered chunks [1], [2], ... + question
   │
   ▼
3. GENERATE      Gemini answers only from the context and cites sources
   │
   ▼
answer + sources
```

The prompt instructs the model to answer **only** from the retrieved context, to cite sources as [1], [2], to answer in the language of the question, and to say "I couldn't find this in the document" when the answer isn't there.

## 🛠️ Technology Stack

| Category | Tools |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini (`gemini-3.5-flash-lite`, free tier) via `google-genai` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector database | ChromaDB (in-memory) |
| PDF loading & chunking | PyPDF, LangChain text splitters |
| Testing & linting | pytest, Ruff |
| Configuration | python-dotenv, central `config.py` |
| Keyword search | BM25 (`rank-bm25`) |
| Reranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` (sentence-transformers) |

## 📁 Project Structure

```
src/
├── app.py                 # Streamlit UI
├── config.py              # All settings in one place (chunking, top-k, models)
├── document_processor.py  # PDF loading and chunking
├── vector_store.py        # Embeddings and ChromaDB
├── prompts.py             # RAG prompt construction
├── chat_engine.py         # Retrieval + answer generation (Gemini / free mode)
└── exceptions.py          # Custom error types
tests/                     # Unit tests (run offline, no API key needed)
eval/                      # Evaluation set, scripts and results
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/VithyabavanS/study-ai.git
cd study-ai
```

### 2️⃣ Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ (Optional) Add a Gemini API key for AI Mode

Get a free key from [Google AI Studio](https://aistudio.google.com), then copy `.env.example` to `.env` and fill it in:

```
GEMINI_API_KEY=your_api_key_here
ANONYMIZED_TELEMETRY=False
```

💡 Without a key, StudyAI runs in Free Mode.

⚠️ On Gemini's free tier, Google may use your inputs to improve its models. Don't upload confidential documents in AI Mode.

### ▶️ Run the application

```bash
streamlit run src/app.py
```

## 🆓 Free Mode vs 🤖 AI Mode

| Feature | Free Mode | AI Mode |
|---|---|---|
| Cost | Free | Free tier (rate-limited) |
| API key needed | ❌ | ✅ Gemini |
| Answer type | Extractive (key sentences) | Generative, with citations |
| Runs locally | ✅ (after the embedding model's one-time download) | Retrieval local, generation via API |

## 📊 Evaluation

StudyAI is evaluated on a hand-written test set of 25 questions about
*Attention Is All You Need* (20 answerable, 5 not answerable from the paper).
Each answerable question has an exact evidence phrase from the document.
See [`eval/`](eval/).

Setup for all runs: chunk size 1000 · overlap 200 · top-k 3 · `gemini-3.5-flash-lite`.

### Retrieval (20 answerable questions)

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Latency / query |
|---|---|---|---|---|---|
| Vector search (baseline) | 0.40 | 0.75 | 0.85 | 0.578 | – |
| BM25 | 0.75 | 0.90 | 0.95 | 0.829 | – |
| Hybrid: BM25 + vector, RRF | 0.55 | 0.90 | 1.00 | 0.731 | 14 ms |
| **Hybrid + cross-encoder rerank (default)** | **0.80** | **0.95** | 0.95 | **0.867** | 314 ms |

### Answers (25 questions, LLM-as-judge)

| Retriever | Accuracy | Faithfulness |
|---|---|---|
| Vector search (baseline) | 0.84 | 1.00 |
| **Hybrid + rerank** | **0.96** | 0.96 |

Manual review of the hybrid + rerank run: the judge's one failure (q24) was a
grading error. The system correctly said the paper gives the batch size only in
tokens, not sentences. All 25 answers are correct in content and supported by the
retrieved text. One answer (q14) was given in French although the question was
in English; this is tracked as an open issue.

### What the numbers show
- **Keyword matching matters.** BM25 found exact-term answers (q06 "Adam",
  q11 "ReLU") that vector search missed.
- **Fusion finds, reranking orders.** Equal-weight RRF lowered Hit@1 compared with
  BM25 alone, but placed the answer in the top 5 for every question. The
  cross-encoder then reordered those candidates (Hit@1 0.55 → 0.80).
- **Better retrieval means better answers.** All three baseline failures caused by
  retrieval misses (q05, q06, q11) are now answered correctly.
- **The cost:** reranking adds about 0.3 s per question on CPU, compared with
  roughly 1–2 s for the LLM call itself.

### Reproduce
```bash
python eval/check_questions.py
PYTHONPATH=src:eval python eval/eval_retrieval.py --mode vector   # or bm25, hybrid, hybrid_rerank
PYTHONPATH=src:eval python eval/eval_answers.py                    # needs GEMINI_API_KEY
```

### Limitations
- Small, single-document test set: each question is 4–5% of a score.
- Questions were written while reading the paper and often reuse its wording,
  which favours keyword search (BM25). Paraphrased questions may score differently.
- Evidence-phrase matching can undercount retrieval when a fact appears in
  several places (q02, q13).
- The judge is an LLM from the same model family and makes mistakes (q24 above);
  failures were checked by hand.
- The reranker (`ms-marco-MiniLM-L-6-v2`) is English-only.
- Results vary slightly between runs, even at temperature 0.

## 🧪 Development

```bash
pip install -r requirements-dev.txt
ruff check src tests      # lint
pytest                    # unit tests
```

## 🔮 Roadmap

- ONNX/INT8 optimization of the reranker
- Tamil + English (bilingual) support with multilingual embeddings
- FastAPI backend, Docker, CI/CD and cloud deployment
- Persistent vector storage and multi-document support

## 📜 License

This project is open-source and available under the MIT License.

## 🙌 Acknowledgements

- [Google Gemini](https://ai.google.dev)
- [HuggingFace](https://huggingface.co)
- [ChromaDB](https://www.trychroma.com)
- [LangChain](https://langchain.com)
- [Streamlit](https://streamlit.io)