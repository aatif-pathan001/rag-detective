"""
Sparse Keyword Retriever implementing BM25 (Okapi BM25).
Directly modularizes and enhances the BM25Retriever algorithm from rag-det.ipynb.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rag_detective.config import default_config
from rag_detective.models import DocumentChunk, SearchResult
from rag_detective.retrievers.base import BaseRetriever

logger = logging.getLogger("uvicorn.error")


class BM25Retriever(BaseRetriever):
    """
    Term-based sparse retriever implementing BM25 scoring with inverted index.
    Preserves exact formulaic parity with rag-det.ipynb prototype:
    - Inverted index tracking term frequencies and token positions.
    - Robertson-Spärck Jones IDF with smoothing: ln(((N - n_qi + 0.5) / (n_qi + 0.5)) + 1).
    - BM25 score calculation with configurable k1 and b.
    - Native invoke() and retrieve() interfaces.
    """

    def __init__(
        self,
        chunks: Optional[Sequence[Union[DocumentChunk, Any]]] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> None:
        """
        Initialize the BM25Retriever.

        Args:
            chunks: Initial list of DocumentChunk or LangChain Document objects.
            k1: BM25 term frequency saturation parameter (default: 1.2).
            b: Document length normalization parameter (default: 0.75).
        """
        self.k1 = k1 if k1 is not None else default_config.bm25_k1
        self.b = b if b is not None else default_config.bm25_b

        self.doc_list: List[Any] = []
        self.inv_idx: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.N: int = 0
        self.avgdl: float = 0.0

        if chunks:
            self.index_documents(chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Extracts cleaned lowercase word tokens from text."""
        # Split on whitespace while stripping wrapping quotes
        raw_tokens = text.split()
        tokens = []
        for t in raw_tokens:
            cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', t.lower())
            if cleaned:
                tokens.append(cleaned)
            elif t.strip():
                tokens.append(t.lower().strip())
        return tokens

    def index_documents(self, documents: Sequence[Union[DocumentChunk, Any]]) -> None:
        """
        Builds the inverted index and corpus statistics over documents.

        """
        self.doc_list = list(documents) # doc_list --> list of all the Documnet Chunks = [Doc1, Doc2, Doc3]
        self.N = len(self.doc_list) # Total number of chunks
        self.inv_idx = {} # Table : {Word -> Doc} referece

        if self.N == 0: 
            self.avgdl = 0.0
            return

        total_words = 0
        for doc in self.doc_list:
            content = getattr(doc, "page_content", "") or ""
            total_words += len(self._tokenize(content))

        self.avgdl = round(total_words / self.N, 2) if self.N > 0 else 0.0
        self._create_inv_idx(self.doc_list)
        logger.info("BM25Retriever indexed %d documents (avgdl: %.2f)", self.N, self.avgdl)

    def _create_inv_idx(self, documents: Sequence[Any]) -> None:
        """Constructs inverted index with frequencies and positions."""
        for doc_id, doc in enumerate(documents, start=1):
            content = getattr(doc, "page_content", "") or ""
            word_list = self._tokenize(content)
            for i, word in enumerate(word_list):
                if word not in self.inv_idx:
                    self.inv_idx[word] = {}

                if doc_id in self.inv_idx[word]:
                    self.inv_idx[word][doc_id]["freq"] += 1
                    self.inv_idx[word][doc_id]["pos"].append(i)
                else:
                    self.inv_idx[word][doc_id] = {"freq": 1, "pos": [i]}

    def _idf(self, qi: str) -> float:
        """
        Computes the Robertson-Spärck Jones IDF with smoothing for term qi.
        Formula from notebook:
            x = (self.N - n_qi + 0.5) / (n_qi + 0.5)
            idf = math.log((x + 1), math.e)
        """
        if qi not in self.inv_idx:
            return 0.0
        n_qi = len(self.inv_idx[qi].keys())
        x = (self.N - n_qi + 0.5) / (n_qi + 0.5)
        return math.log(x + 1.0, math.e)

    def _score_qi(self, qi: str, doc_id: int) -> float:
        """
        Calculates the BM25 contribution of token qi to document doc_id.
        Formula from notebook:
            del_d = doc word length
            x = (f_qi_D * (k1 + 1)) / (f_qi_D + k1 * (1 - b + b * (del_d / self.avgdl)))
            score = idf * x
        """
        if qi not in self.inv_idx:
            return 0.0

        idf = self._idf(qi)
        f_qi_D = self.inv_idx[qi][doc_id]["freq"] if doc_id in self.inv_idx[qi] else 0
        if f_qi_D == 0:
            return 0.0

        del_d = len(self._tokenize(getattr(self.doc_list[doc_id - 1], "page_content", "")))
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        denominator = f_qi_D + self.k1 * (1.0 - self.b + self.b * (del_d / avgdl))
        if denominator == 0:
            return 0.0

        x = (f_qi_D * (self.k1 + 1.0)) / denominator
        return idf * x

    def invoke(self, query: str, top_k: int = 3) -> List[Tuple[float, Any]]:
        """
        Original notebook interface: returns sorted list of (BM25_score, document) tuples.

        Args:
            query: Keyword query string.
            top_k: Number of highest ranking documents to return.

        Returns:
            List of (score, doc) pairs matching rag-det.ipynb.
        """
        if not self.doc_list or not query or not query.strip():
            return []

        token_query = self._tokenize(query)
        bm25_ranking: List[Tuple[float, Any]] = []

        for doc_id, document in enumerate(self.doc_list, start=1):
            score = 0.0
            for qi in token_query:
                score += self._score_qi(qi, doc_id)
            bm25_ranking.append((score, document))

        bm25_ranking.sort(key=lambda item: item[0], reverse=True)
        return bm25_ranking[:top_k]

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Standardized BaseRetriever interface returning SearchResult objects.
        """
        ranked_pairs = self.invoke(query=query, top_k=top_k)
        results: List[SearchResult] = []

        for rank, (score, doc) in enumerate(ranked_pairs, start=1):
            chunk_id = getattr(doc, "id", None) or f"Chunk_{rank}"
            content = getattr(doc, "page_content", "")
            metadata = getattr(doc, "metadata", {}) or {}

            results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    content=content,
                    metadata=dict(metadata),
                    score=round(score, 4),
                    rank=rank,
                    engine="sparse",
                )
            )

        logger.info("BM25Retriever retrieved %d chunks for query: '%s'", len(results), query[:50])
        return results
