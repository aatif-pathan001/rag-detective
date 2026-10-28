"""
Ingestion module for RAG Detective.
"""

from rag_detective.ingestion.base import BaseChunkSplitter, BaseDocumentLoader
from rag_detective.ingestion.markdown_splitter import MarkdownDocumentSplitter
from rag_detective.ingestion.pdf_loader import DocumentLoader
from rag_detective.ingestion.pipeline import IngestionPipeline

__all__ = [
    "BaseDocumentLoader",
    "BaseChunkSplitter",
    "MarkdownDocumentSplitter",
    "DocumentLoader",
    "IngestionPipeline",
]
