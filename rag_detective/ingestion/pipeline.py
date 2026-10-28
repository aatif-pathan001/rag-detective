"""
Unified Ingestion Pipeline for RAG Detective.
Handles reading documents (PDF, Markdown, text), splitting into header-aware chunks,
and returning validated DocumentChunk models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from rag_detective.exceptions import IngestionError
from rag_detective.ingestion.base import BaseChunkSplitter, BaseDocumentLoader
from rag_detective.ingestion.markdown_splitter import MarkdownDocumentSplitter
from rag_detective.ingestion.pdf_loader import DocumentLoader
from rag_detective.models import DocumentChunk

logger = logging.getLogger("uvicorn.error")


class IngestionPipeline:
    """
    Coordinates document extraction and chunking for the RAG Detective system.
    """

    def __init__(
        self,
        loader: Optional[BaseDocumentLoader] = None,
        splitter: Optional[BaseChunkSplitter] = None,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.splitter = splitter or MarkdownDocumentSplitter()

    def process_file(
        self,
        file_path: Union[str, Path],
        start_chunk_idx: int = 1,
    ) -> List[DocumentChunk]:
        """
        Parses a file from disk into DocumentChunk instances.

        Supports:
        - PDF (.pdf) via pdfmuse / fallback
        - Word (.docx) via python-docx
        - Markdown (.md)
        - Text (.txt)

        Args:
            file_path: Path to target file.
            start_chunk_idx: Initial index for chunk naming.

        Returns:
            List of standardized DocumentChunk instances.
        """
        path = Path(file_path).resolve()
        markdown_text = self.loader.load(path)

        # Apply splitting
        if hasattr(self.splitter, "start_index"):
            self.splitter.start_index = start_chunk_idx

        chunks = self.splitter.split(markdown_text, source_name=path.name)
        return chunks

    # def extract_markdown(self, file_path: Union[str, Path]) -> str:
    #     """
    #     Extracts the textual content of a file as Markdown (without chunking).

    #     Supports:
    #     - PDF (.pdf) via pdfmuse / pypdf fallback
    #     - Word (.docx) via python-docx (heading styles -> #, ##, ###)
    #     - Markdown (.md, .markdown) and text (.txt)

    #     Args:
    #         file_path: Path to target file.

    #     Returns:
    #         Markdown representation of the document.
    #     """
    #     # if not file_path.is_file():
    #     #     raise IngestionError(f"Target file does not exist: {file_path}")

    #     suffix = file_path.suffix.lower()
    #     logger.info("Processing file '%s' (format: %s)", file_path.name, suffix)

    #     if suffix == ".pdf":
    #         return self.pdf_loader.load(file_path)
    #     if suffix == ".docx":
    #         return self._docx_to_markdown(file_path)
    #     if suffix in (".md", ".markdown", ".txt"):
    #         try:
    #             return file_path.read_text(encoding="utf-8")
    #         except Exception as e:
    #             raise IngestionError(f"Failed to read text file '{file_path}': {e}") from e

    #     # Attempt to parse as text if possible
    #     try:
    #         return path.read_text(encoding="utf-8")
    #     except Exception as e:
    #         raise IngestionError(
    #             f"Unsupported file format '{suffix}' for '{path.name}': {e}"
            # ) from e

    # @staticmethod
    # def _docx_to_markdown(path: Path) -> str:
    #     """Converts a .docx file into Markdown, mapping Word heading styles to '#' markers."""
    #     try:
    #         import docx  # python-docx
    #     except ImportError as e:
    #         raise IngestionError(
    #             "python-docx is required for .docx files. Install via `pip install python-docx`"
    #         ) from e

    #     try:
    #         document = docx.Document(str(path))
    #     except Exception as e:
    #         raise IngestionError(f"Failed to open Word document '{path.name}': {e}") from e

    #     lines: List[str] = []
    #     for para in document.paragraphs:
    #         text = para.text.strip()
    #         if not text:
    #             continue
    #         style = (para.style.name if para.style is not None else "") or ""
    #         if style.lower() == "title":
    #             lines.append(f"# {text}")
    #         elif style.lower().startswith("heading"):
    #             digits = "".join(ch for ch in style if ch.isdigit())
    #             level = min(6, max(1, int(digits))) if digits else 1
    #             lines.append(f"{'#' * level} {text}")
    #         elif style.lower().startswith("list"):
    #             lines.append(f"- {text}")
    #         else:
    #             lines.append(text)
    #         lines.append("")

        # for table in document.tables:
        #     for row in table.rows:
        #         cells = [cell.text.strip() for cell in row.cells]
        #         lines.append("| " + " | ".join(cells) + " |")
        #     lines.append("")

        # return "\n".join(lines)

    def process_markdown(
        self,
        markdown_text: str,
        source_name: str = "direct_input.md",
        start_chunk_idx: int = 1,
    ) -> List[DocumentChunk]:
        """
        Processes raw markdown text into DocumentChunk instances.
        """
        if hasattr(self.splitter, "start_index"):
            self.splitter.start_index = start_chunk_idx

        return self.splitter.split(markdown_text, source_name=source_name)
