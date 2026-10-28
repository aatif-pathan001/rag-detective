# 🔍 RAG Detective Backend

An enterprise-ready, modular, hybrid Retrieval-Augmented Generation (RAG) backend engineered with clean architecture, strict context grounding, dual-index search (Dense Semantic + Sparse BM25), Weighted Reciprocal Rank Fusion (Weighted RRF), and Gemini-powered query rewriting.

---

## 🏛️ Architecture Overview

```
                                      [User Query]
                                           │
                                           ▼
                             ┌───────────────────────────┐
                             │    Query Rewriter (LLM)   │
                             │  - Intent Resolution      │
                             │  - Ambiguity Check        │
                             │  - Dense & Sparse Queries │
                             └─────────────┬─────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │    Dense Retriever    │                     │  Sparse BM25 Retriever│
        │   (ChromaDB + Cosine  │                     │  (Inverted Index +    │
        │     BGE Embeddings)   │                     │   Robertson-Spärck)   │
        └───────────┬───────────┘                     └───────────┬───────────┘
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                             ┌───────────────────────────┐
                             │   Weighted Rank Fusion    │
                             │  - Reciprocal Rank Fusion │
                             │  - α * Sparse + (1-α)*Dense│
                             └─────────────┬─────────────┘
                                           │ Top-K Evidence Chunks
                                           ▼
                             ┌───────────────────────────┐
                             │   Detective Synthesizer   │
                             │  - Grounded reasoning     │
                             │  - Strict context bounding│
                             │  - Citation preservation  │
                             └─────────────┬─────────────┘
                                           │
                                           ▼
                             [Investigation Report & Evidence]
```

---

## 📁 Clean Folder Structure

```
rag_detective/
├── __init__.py               # Public API exports
├── config.py                 # Type-safe configuration with env var overrides
├── models.py                 # Pydantic v2 schemas (Chunks, Requests, Results)
├── exceptions.py             # Custom domain exception hierarchy
├── engine.py                 # RAGDetectiveEngine (Unified backend orchestrator)
├── cli.py                    # Terminal CLI for ingestion, search, & queries
├── ingestion/                # Document loaders & chunk splitters
│   ├── __init__.py
│   ├── base.py               # BaseLoader & BaseSplitter interfaces
│   ├── markdown_splitter.py  # Markdown header splitter (# to ######)
│   ├── pdf_loader.py         # PDF-to-Markdown parser (pdfmuse with fallbacks)
│   └── pipeline.py           # Unified document ingestion pipeline
├── indexing/                 # Embeddings & Vector Store
│   ├── __init__.py
│   ├── embeddings.py         # HuggingFace BGE embeddings manager
│   └── vector_store.py       # Chroma vector database wrapper (cosine space)
├── retrievers/               # Search engines
│   ├── __init__.py
│   ├── base.py               # BaseRetriever interface
│   ├── dense.py              # Semantic vector retriever
│   ├── sparse.py             # Okapi BM25 sparse retriever with inverted index
│   └── hybrid.py             # Hybrid coordinator for dense + sparse
├── fusion/                   # Rank Fusion Algorithms
│   ├── __init__.py
│   └── rrf.py                # Weighted Reciprocal Rank Fusion (Weighted RRF)
├── query/                    # Query Processing & Rewriting
│   ├── __init__.py
│   ├── prompts.py            # System prompts & XML formatting templates
│   └── rewriter.py           # Gemini-based query rewriter & disambiguator
├── generation/               # Detective Answer Synthesis
│   ├── __init__.py
│   ├── prompts.py            # Context-bounded prompt generator
│   └── synthesizer.py        # Gemini-based factual answer synthesizer
└── api/                      # REST API Layer
    ├── __init__.py
    ├── app.py                # FastAPI application factory
    └── routes.py             # REST endpoints (/investigate, /search, /ingest)
```

---

## 🚀 Quickstart

### 1. Python Library Usage

```python
from rag_detective import RAGDetectiveEngine, InvestigationRequest

# Initialize engine
engine = RAGDetectiveEngine()

# Ingest document (PDF, Markdown, or TXT)
engine.ingest_file("paper.pdf")

# Run an investigation
request = InvestigationRequest(
    user_query="Who introduces Generation-Augmented Retrieval (GAR)?",
    top_k=3,
    alpha=0.7,
)
result = engine.investigate(request)

print("Answer:", result.answer)
for evidence in result.evidence:
    print(f"Rank {evidence.rank} [Score: {evidence.score}] ID: {evidence.chunk_id}")
    print(evidence.content[:150])
```

### 2. Command Line Interface (CLI)

```bash
# Show system statistics
python -m rag_detective.cli stats

# Ingest a document
python -m rag_detective.cli ingest paper.pdf

# Search across indexed chunks (modes: hybrid, dense, sparse)
python -m rag_detective.cli search "Generation-Augmented Retrieval" --mode hybrid

# Run full investigation
python -m rag_detective.cli query "Who introduces Generation-Augmented Retrieval (GAR)?"

# Start FastAPI backend server
python -m rag_detective.cli serve --port 8000
```

### 3. REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/investigate` | Full RAG pipeline (rewriting, hybrid retrieval, fusion, synthesis) |
| `POST` | `/api/v1/search` | Direct search (`hybrid`, `dense`, or `sparse`) |
| `POST` | `/api/v1/ingest/file` | Multipart file upload (PDF, MD, TXT) |
| `POST` | `/api/v1/ingest/text` | Direct text/markdown ingestion |
| `GET`  | `/api/v1/stats` | Indexed chunks and system parameters |
| `GET`  | `/api/v1/health` | Health status check |

---

## ⚙️ Configuration Reference

All settings can be customized through environment variables or by passing `RAGDetectiveConfig`:

| Environment Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(Set in config)* | Google GenAI API key for Rewriter & Synthesizer |
| `GEMINI_MODEL_NAME` | `gemini-3.1-flash-lite` | Gemini model name |
| `EMBEDDING_MODEL_NAME`| `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHROMA_COLLECTION_NAME`| `collection0` | Chroma collection name |
| `CHROMA_PERSIST_DIR` | `.chroma_db` | Storage path for Chroma DB |
| `CHROMA_DISTANCE_METRIC`| `cosine` | Vector distance metric |
| `BM25_K1` | `1.2` | Term frequency saturation constant |
| `BM25_B` | `0.75` | Document length normalization |
| `RRF_ALPHA` | `0.7` | Weight for sparse retrieval in Weighted RRF |
| `RRF_K` | `60` | Rank constant smoothing factor |

---

## 🧪 Testing

Run unit tests directly:

```bash
python -m unittest tests/test_rag_detective.py
```
