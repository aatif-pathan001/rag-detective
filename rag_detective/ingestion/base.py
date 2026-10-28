"""
Base interfaces for document loaders and chunk splitters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from rag_detective.models import DocumentChunk


class BaseDocumentLoader(ABC):
    """Abstract interface for reading and converting documents to markdown."""

    @abstractmethod
    def load(self, source: Union[str, Path, bytes]) -> str:
        """
        Loads document from file path or bytes and returns markdown text.
        """
        pass


class BaseChunkSplitter(ABC):
    """Abstract interface for splitting markdown or text into structured chunks."""

    @abstractmethod
    def split(self, text: str, source_name: str = "") -> List[DocumentChunk]:
        """
        Splits text into a list of DocumentChunk instances with assigned IDs and metadata.
        """
        pass
