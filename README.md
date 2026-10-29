# 📚 StudyAI – Intelligent PDF Analyser

![CI](https://github.com/VithyabavanS/study-ai/actions/workflows/ci.yml/badge.svg)

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

## ♿ Accessibility

Built so students with different needs can use it comfortably:
- **Reading settings:** text size (normal, large, extra large), night mode,
  high contrast, and wider letter, word and line spacing (WCAG 1.4.12 values,
  helpful for many readers with dyslexia)
- **Listen to answer:** answers are read aloud in their own language, using a
  Sri Lankan Tamil voice (`ta-LK-SaranyaNeural`) for Tamil
- **Screen readers:** labelled inputs, Tamil text marked with `lang="ta"` so it is
  pronounced correctly, and an announcement when a new answer appears
- **Keyboard:** press Enter to ask; every control shows a clear focus outline
- Glass panels turn solid when the system's "reduce transparency" setting is on

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
| Embeddings | HuggingFace `multilingual-e5-small (MIT)` (local) |
| Vector database | ChromaDB (in-memory) |
| PDF loading & chunking | PyPDF, LangChain text splitters |
| Testing & linting | pytest, Ruff |
| Configuration | python-dotenv, central `config.py` |
| Keyword search | BM25 (`rank-bm25`) |
| Reranking | Cross-encoder `mmarco-mMiniLMv2-L12-H384-v1 (Apache 2.0)` (sentence-transformers) |
| PDF extraction | PyMuPDF (AGPL) |

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

Two hand-written test sets (see [`eval/`](eval/)):
- **English:** 25 questions on *Attention Is All You Need* (20 answerable, 5 not answerable from the paper)
- **Tamil:** 15 questions on the Tamil Wikipedia article *ஒளித்தொகுப்பு* (photosynthesis):
  11 in Tamil and 4 in English about the Tamil text (cross-lingual); 12 answerable, 3 not

### Retrieval: Hit@3 / MRR@5

| Setup | English | Tamil |
|---|---|---|
| Vector search, MiniLM (original) | 0.75 / 0.578 | 0.17 / 0.069 |
| Hybrid + English reranker, MiniLM embeddings | 0.95 / 0.867 | 0.17 / 0.207 |
| Hybrid, multilingual-e5-small | 0.85 / 0.775 | 0.75 / 0.597 |
| **Hybrid, e5-small + multilingual reranker (default)** | 0.90 / 0.804 | **0.83 / 0.850** |

The default setup takes about 0.6–0.7 s per question for retrieval on CPU.

### Answers (default setup)

| Set | Accuracy (judge) | Accuracy (after manual review) | Faithfulness | Language match |
|---|---|---|---|---|
| English (25) | 0.92 | 0.96 | 1.00 | 1.00 |
| Tamil set (15) | 0.93 | 0.93 | 1.00 | 1.00 |

"Language match" is checked in code: Tamil questions must get Tamil answers and
English questions English answers. For comparison, the original vector-search
system scored 0.84 accuracy on the English set.

### Findings
- **PDF extraction broke Tamil.** pypdf reordered Tamil vowel signs (12–54% invalid
  positions, depending on version); the article's main term was found 0 times instead
  of 42. Switching to PyMuPDF fixed this, and a test guards against regressions.
- **English-only models fail on Tamil.** The setup that was best for English scored
  Tamil Hit@3 of only 0.17.
- **Multilingual embeddings helped both languages:** e5-small improved English vector
  search too (Hit@3 0.75 → 0.90).
- **A multilingual reranker closed the gap for Tamil** (Hit@3 0.75 → 0.83, MRR
  0.60 → 0.85), at the cost of one English question compared with the English-only setup.
- **Cross-lingual questions mostly work:** two of three English questions about the
  Tamil article were retrieved at rank 1.
- **Dialect matters.** The same question using the Sri Lankan term இலத்திரன் was
  retrieved at rank 1, but with the Indian term எலக்ட்ரான் at rank 5 (one pair only).
- **The LLM judge needs checking.** It first marked two failures as correct, then,
  after a stricter prompt, marked one correct answer as wrong. All failures were
  reviewed by hand.

### Reproduce
```bash
python eval/check_questions.py [eval/questions_ta.jsonl]
PYTHONPATH=src:eval python eval/eval_retrieval.py --mode hybrid_rerank [--questions eval/questions_ta.jsonl]
PYTHONPATH=src:eval python eval/eval_answers.py [--questions eval/questions_ta.jsonl]   # needs GEMINI_API_KEY
```

### Limitations
- Small test sets (25 English, 15 Tamil), each from a single document.
- English questions often reuse the paper's wording, which favours keyword search.
- The reranker's model card does not list Tamil, and it was trained on a translation
  of MS MARCO, whose data has research-use terms.
- The dialect finding rests on a single question pair.
- The judge is an LLM from the same model family as the answer model.
- Results vary slightly between runs, even at temperature 0.
- Answers are checked for language by script share; a mostly-Tamil answer with an
  English fragment, or a stray character from another script, can still pass (both seen once
  before the "not found" sentence was made language-specific).
- Read-aloud uses `edge-tts`, an unofficial client for Microsoft Edge's online
  voices: it needs internet access and could stop working if the service changes.

## 🧪 Development

```bash
pip install -r requirements-dev.txt
ruff check src tests      # lint
pytest                    # unit tests
```

## 🔮 Roadmap

- FastAPI backend, Docker, CI/CD and cloud deployment
- Language-based routing: use the English reranker for English questions and the multilingual one for Tamil
- ONNX/INT8 optimization of the reranker to reduce CPU latency
- More dialect question pairs (Sri Lankan vs Indian Tamil terms)
- Persistent vector storage and multi-document support

## 📜 License

This project is open-source and available under the MIT License.

## 🙌 Acknowledgements

- [Google Gemini](https://ai.google.dev)
- [HuggingFace](https://huggingface.co)
- [ChromaDB](https://www.trychroma.com)
- [LangChain](https://langchain.com)
- [Streamlit](https://streamlit.io)