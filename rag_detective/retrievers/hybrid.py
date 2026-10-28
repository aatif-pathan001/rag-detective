"""
Hybrid Retriever combining Dense Vector and Sparse BM25 Search via Weighted RRF.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rag_detective.config import default_config
from rag_detective.fusion.rrf import WeightedRRF
from rag_detective.models import DocumentChunk, SearchResult
from rag_detective.retrievers.base import BaseRetriever
from rag_detective.retrievers.dense import DenseRetriever
from rag_detective.retrievers.sparse import BM25Retriever

logger = logging.getLogger("uvicorn.error")


class HybridRetriever(BaseRetriever):
    """
    Coordinates dense semantic and sparse lexical search, merging candidates
    using Weighted Reciprocal Rank Fusion (Weighted RRF).
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        fusion_engine: Optional[WeightedRRF] = None,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        final_k: Optional[int] = None,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.fusion = fusion_engine or WeightedRRF()
        self.dense_k = dense_k or default_config.retrieval_dense_k
        self.sparse_k = sparse_k or default_config.retrieval_sparse_k
        self.final_k = final_k or default_config.final_top_k
        self._chunk_map: Dict[str, DocumentChunk] = {}

    def set_indexed_chunks(self, chunks: List[DocumentChunk]) -> None: ## <-------- Needs Review
        """Stores chunk references to ensure complete metadata resolution during fusion.""" 
        self._chunk_map = {chunk.id: chunk for chunk in chunks}
        self.sparse_retriever.index_documents(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Executes hybrid retrieval using the same query string for both dense and sparse.
        """
        return self.retrieve_hybrid(dense_query=query, sparse_query=query, top_k=top_k)

    def retrieve_hybrid(
        self,
        dense_query: str,
        sparse_query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Executes dual hybrid retrieval using specialized dense and sparse queries.

        Args:
            dense_query: Natural language query optimized for semantic search.
            sparse_query: Keyword-centric query string optimized for BM25.
            top_k: Number of final fused results to return.
            alpha: Optional override for sparse fusion weight.

        Returns:
            List of SearchResult objects sorted by Weighted RRF score.
        """
        target_k = top_k or self.final_k

        # 1. Execute Dense Retrieval
        dense_results = self.dense_retriever.retrieve(
            dense_query, top_k=self.dense_k
        )

        # 2. Execute Sparse Retrieval
        sparse_results = self.sparse_retriever.retrieve(
            sparse_query, top_k=self.sparse_k
        )

        # 3. Apply Reciprocal Rank Fusion
        fusion_engine = self.fusion
        if alpha is not None and alpha != self.fusion.alpha:
            fusion_engine = WeightedRRF(alpha=alpha, k=self.fusion.k)

        fused_results = fusion_engine.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            chunk_map=self._chunk_map,
            top_k=target_k,
        )

        logger.info(
            "HybridRetriever produced %d fused results (dense candidates: %d, sparse candidates: %d)",
            len(fused_results),
            len(dense_results),
            len(sparse_results),
        )
        return fused_results
