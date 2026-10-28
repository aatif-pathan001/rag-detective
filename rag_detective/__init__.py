"""
RAG Detective: Production-grade Modular Hybrid RAG Backend.
Features:
- Dual semantic & keyword search (Chroma Cosine VectorDB + Custom BM25)
- Weighted Reciprocal Rank Fusion (Weighted RRF)
- Gemini-powered query rewriting & sub-query decomposition
- Context-bounded detective answer synthesis
- REST API and CLI interfaces
"""

from rag_detective.config import RAGDetectiveConfig, default_config
from rag_detective.engine import RAGDetectiveEngine
from rag_detective.fusion.rrf import WeightedRRF, weighted_rrf_scores
from rag_detective.generation.synthesizer import DetectiveSynthesizer
from rag_detective.indexing.embeddings import EmbeddingManager
from rag_detective.indexing.vector_store import ChromaVectorStore
from rag_detective.ingestion.markdown_splitter import MarkdownDocumentSplitter
from rag_detective.ingestion.pdf_loader import DocumentLoader
from rag_detective.ingestion.pipeline import IngestionPipeline
from rag_detective.models import (
    DocumentChunk,
    IngestionResponse,
    InvestigationRequest,
    InvestigationResult,
    QueryRewriteResult,
    SearchResult,
    SubQuery,
)
from rag_detective.query.rewriter import QueryRewriter
from rag_detective.retrievers.dense import DenseRetriever
from rag_detective.retrievers.hybrid import HybridRetriever
from rag_detective.retrievers.sparse import BM25Retriever

__version__ = "1.0.0"

__all__ = [
    # Core Engine & Config
    "RAGDetectiveEngine",
    "RAGDetectiveConfig",
    "default_config",
    # Ingestion & Splitting
    "IngestionPipeline",
    "MarkdownDocumentSplitter",
    "DocumentLoader",
    # Indexing & Storage
    "EmbeddingManager",
    "ChromaVectorStore",
    # Retrievers
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    # Fusion
    "WeightedRRF",
    "weighted_rrf_scores",
    # Query & Generation
    "QueryRewriter",
    "DetectiveSynthesizer",
    # Models
    "DocumentChunk",
    "SearchResult",
    "SubQuery",
    "QueryRewriteResult",
    "InvestigationRequest",
    "InvestigationResult",
    "IngestionResponse",
]
