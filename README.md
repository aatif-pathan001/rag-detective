# RAG Detective

Interactive Hybrid RAG pipeline visualizer and investigation engine powered by **FastAPI**, **BAAI/bge-small-en-v1.5**, **ChromaDB**, **BM25**, **Weighted Reciprocal Rank Fusion (RRF)**, and **Google Gemini**.

## Features
- **Stage-by-Stage Pipeline Trace**: Real-time streaming visualization of document intake, Markdown structure recovery, header-aware chunking, 384-d dense embedding, 3-D PCA vector store projection, and BM25 inverted indexing.
- **Hybrid Investigation**: Dual query rewriting (dense semantic + sparse keyword), Weighted RRF rank fusion, and grounded answer synthesis with inline chunk citations.
- **Session Isolation & Privacy**: Every visitor gets an isolated in-memory vector store that is automatically cleared on every new case.
