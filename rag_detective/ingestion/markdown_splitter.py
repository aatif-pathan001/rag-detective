"""
Markdown Header-based Document Splitter.
Splits markdown documents along heading hierarchies and assigns standardized IDs.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from langchain_text_splitters import MarkdownHeaderTextSplitter

from rag_detective.config import default_config
from rag_detective.exceptions import IngestionError
from rag_detective.ingestion.base import BaseChunkSplitter
from rag_detective.models import DocumentChunk

logger = logging.getLogger("uvicorn.error")


class MarkdownDocumentSplitter(BaseChunkSplitter):
    """
    Splits markdown content based on structural header boundaries (# through ######).
    Matches the chunking strategy prototyped in rag-det.ipynb.
    """

    def __init__(
        self,
        headers_to_split_on: Optional[Sequence[Tuple[str, str]]] = None,
        chunk_id_prefix: str = "Chunk_",
        start_index: int = 1,
    ) -> None:
        """
        Initialize the markdown header splitter.

        Args:
            headers_to_split_on: Tuples of (header_marker, header_name).
                                 Defaults to levels 1 through 6.
            chunk_id_prefix: Prefix string for chunk identifiers (default 'Chunk_').
            start_index: Initial integer index for chunk naming (default 1).
        """

        self.headers_to_split_on = list(
            headers_to_split_on or default_config.headers_to_split_on
        )
        self.chunk_id_prefix = chunk_id_prefix
        self.start_index = start_index
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )

    def split(self, text: str, source_name: str = "") -> List[DocumentChunk]:
        """
        Splits markdown text into DocumentChunk instances with assigned unique IDs.

        Args:
            text: Raw markdown text string.
            source_name: Optional file or source identifier to attach to chunk metadata.

        Returns:
            List of DocumentChunk objects with assigned chunk IDs.
        """
        
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text passed to MarkdownDocumentSplitter.")
            return []

        try:
            lc_docs = self._splitter.split_text(text)
        except Exception as e:
            logger.error("Failed to split markdown text: %s", e)
            raise IngestionError(f"Markdown splitting failed: {e}") from e

        chunks: List[DocumentChunk] = []
        for idx, doc in enumerate(lc_docs, start=self.start_index):
            chunk_id = f"{self.chunk_id_prefix}{idx}"
            metadata = dict(doc.metadata) if doc.metadata else {}
            if source_name:
                metadata["source"] = source_name
            metadata["chunk_index"] = idx

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

        logger.info(
            "Split source '%s' into %d chunks using MarkdownHeaderTextSplitter.",
            source_name or "in-memory text",
            len(chunks),
        )
        return chunks
