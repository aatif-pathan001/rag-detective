"""
PDF Document Loader for converting PDF documents into structured Markdown text.
Uses pdfmuse (from rag-det.ipynb prototype) with resilient fallback support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, List

from rag_detective.exceptions import DocumentParseError
from rag_detective.ingestion.base import BaseDocumentLoader

logger = logging.getLogger("uvicorn.error")



class DocumentLoader(BaseDocumentLoader):
    """
    Parses documents into markdown text.
    """

    def __init__(self, source_name: str = "auto") -> None:
        """
        Initialize documnet loader. Support (pdf, docx, md, txt)
        """
        self.source_name = None

    def load(self, source: Union[str, Path]) -> str:
        """
        Loads and converts a source document into Markdown text.

        Args:
            source: File path (str or Path).

        Returns:
            Markdown representation of content.

        Raises:
            DocumentParseError: If file reading or parsing fails.
        """

        if not source.is_file():
            raise DocumentParseError(f"Target file does not exist: {source}")

        suffix = source.suffix.lower()
        pdf_bytes = self._read_source_bytes(source)
        self.source_name = str(source)

        if suffix == ".pdf" or self.source_name in ("auto", "pdfmuse"):
            return self.load_pdf(pdf_bytes)
        if suffix == '.docx':
            return self.load_docx(source)
        if suffix in (".md", ".markdown", ".txt"):
            try:
                return source.read_text(encoding="utf-8")
            except Exception as e:
                raise DocumentParseError(f"Failed to read text file '{source}': {e}") from e

        try:
            return source.read_text(encoding="utf-8")
        except Exception as e:
            raise DocumentParseError(
                f"Unsupported file format '{suffix}' for '{source.name}': {e}"
            ) from e
        
    def load_pdf(self, pdf_bytes: bytes) -> str:
        import io

        try:
            import pdfmuse
            logger.info("Parsing PDF '%s' using pdfmuse...", self.source_name)
            markdown_text = pdfmuse.to_markdown(pdf_bytes)
            if markdown_text and markdown_text.strip():
                return markdown_text
        except ImportError:
            logger.debug("pdfmuse not installed, falling back to pypdf.")
        except Exception as e:
            logger.warning("pdfmuse failed for '%s': %s. Falling back to pypdf...", self.source_name, e)

        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(p.strip() for p in pages if p.strip())
            if text:
                return text
        except Exception as e:
            raise DocumentParseError(f"Failed to parse PDF '{self.source_name}': {e}") from e

        raise DocumentParseError(f"No extractable text found in PDF '{self.source_name}'.")

    def load_docx(self, file_path: Union[(str, Path)]):
        import zipfile

        try:
            import docx
        except ImportError as e:
            logger.error("docx not installed.")
            raise DocumentParseError(
                "python-docx is required for .docx files. Install via `pip install python-docx`"
                ) from e

        # Guard against DOCX ZIP-bomb / XML memory amplification attacks (max 50 MB uncompressed)
        max_uncompressed_bytes = 50 * 1024 * 1024
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                total_uncompressed = sum(info.file_size for info in zf.infolist())
                if total_uncompressed > max_uncompressed_bytes:
                    raise DocumentParseError(
                        "DOCX archive uncompressed content exceeds the 50 MB safety limit."
                    )
        except DocumentParseError:
            raise
        except Exception as e:
            raise DocumentParseError(f"Invalid or corrupted DOCX archive: {e}") from e         
                  
        try:
            document = docx.Document(file_path)
        except Exception as e:
            logger.error("docx failed for '%s': %s...", self.source_name, e)
            raise DocumentParseError(f"Failed to open Word document '{file_path.name}': {e}") from e
                  
                          
        logger.info("Parsing DOCX '%s' using python-docx...", self.source_name)
        lines: List[str] = []
        for para in document.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            if style.lower() == "title":
                lines.append(f"# {text}")
            elif style.lower().startswith("heading"):
                digits = "".join(ch for ch in style if ch.isdigit())
                level = min(6, max(1, int(digits))) if digits else 1
                lines.append(f"{'#' * level} {text}")
            elif style.lower().startswith("list"):
                lines.append(f"- {text}")
            else:
                lines.append(text)
            lines.append("")
                  
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")      
        return "\n".join(lines)
        
                  

    def _read_source_bytes(self, source: Union[str, Path, bytes]) -> bytes:
        if isinstance(source, bytes):
            return source
        path = Path(source).resolve()
        if not path.is_file():
            raise DocumentParseError(f"PDF file does not exist: {path}")
        try:
            return path.read_bytes()
        except Exception as e:
            raise DocumentParseError(f"Failed to read file {path}: {e}") from e







# class PDFDocumentLoader(BaseDocumentLoader):
#     """
#     Parses PDF documents into markdown text.
#     Primary engine: pdfmuse
#     """

#     def __init__(self, engine: str = "auto") -> None:
#         """
#         Initialize PDF loader.

#         Args:
#             engine: Engine selection ('auto', 'pdfmuse').
#         """
#         self.engine = engine

    # def load(self, source: Union[str, Path, bytes]) -> str:
    #     """
    #     Loads and converts a PDF source into Markdown text.

    #     Args:
    #         source: File path (str or Path) or raw PDF bytes.

    #     Returns:
    #         Markdown representation of the PDF content.

    #     Raises:
    #         DocumentParseError: If file reading or parsing fails.
    #     """

        # pdf_bytes = self._read_source_bytes(source)
        # source_name = str(source) if isinstance(source, (str, Path)) else "<in-memory bytes>"

        # if self.engine in ("auto", "pdfmuse"):
        #     try:
        #         import pdfmuse
        #         logger.info("Parsing PDF '%s' using pdfmuse...", source_name)
        #         markdown_text = pdfmuse.to_markdown(pdf_bytes)
        #         if markdown_text and markdown_text.strip():
        #             return markdown_text
        #     except ImportError:
        #         logger.debug("pdfmuse not installed, falling back to alternative parsers.")
        #     except Exception as e:
        #         logger.warning("pdfmuse failed for '%s': %s. Falling back to pypdf...", source_name, e)

    # def _read_source_bytes(self, source: Union[str, Path, bytes]) -> bytes:
    #     if isinstance(source, bytes):
    #         return source
    #     path = Path(source).resolve()
    #     if not path.is_file():
    #         raise DocumentParseError(f"PDF file does not exist: {path}")
    #     try:
    #         return path.read_bytes()
    #     except Exception as e:
    #         raise DocumentParseError(f"Failed to read file {path}: {e}") from e



# class DocxDocumentLoader(BaseDocumentLoader):
#     """
#     Parses docx documents into markdown text.
#     Primary engine: docx
#     """

#     def __init__(self) -> None:
#         """
#         Initialize docx loader.
#         """
#         self.engine = "docx"

#     def load(self, source: Union[str, Path]) -> str:
#         """
#         Loads and converts a docx source into Markdown text.

#         Args:
#             source: File path (str or Path).

#         Returns:
#             Markdown representation of the docx content.

#         Raises:
#             DocumentParseError: If file reading or parsing fails.
#         """

#         source_name = str(source)

        # try:
        #     import docx
        # except ImportError as e:
        #     logger.error("docx not installed.")
        #     raise DocumentParseError(
        #         "python-docx is required for .docx files. Install via `pip install python-docx`"
        #     ) from e


        # try:
        #     document = docx.Document(source))
        # except Exception as e:
        #     ogger.error("docx failed for '%s': %s...", source_name, e)
        #     raise DocumentParseError(f"Failed to open Word document '{path.name}': {e}") from e

        
        # logger.info("Parsing DOCX '%s' using python-docx...", source_name)
        # lines: List[str] = []
        # for para in document.paragraphs:
        #     text = para.text.strip()
        #     if not text:
        #         continue
        #     style = (para.style.name if para.style is not None else "") or ""
        #     if style.lower() == "title":
        #         lines.append(f"# {text}")
        #     elif style.lower().startswith("heading"):
        #         digits = "".join(ch for ch in style if ch.isdigit())
        #         level = min(6, max(1, int(digits))) if digits else 1
        #         lines.append(f"{'#' * level} {text}")
        #     elif style.lower().startswith("list"):
        #         lines.append(f"- {text}")
        #     else:
        #         lines.append(text)
        #     lines.append("")

        # for table in document.tables:
        #     for row in table.rows:
        #         cells = [cell.text.strip() for cell in row.cells]
        #         lines.append("| " + " | ".join(cells) + " |")
        #     lines.append("")

        # return "\n".join(lines)



