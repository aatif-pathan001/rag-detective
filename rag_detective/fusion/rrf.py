"""
Reciprocal Rank Fusion (RRF) algorithms for hybrid search.
Implements the exact Weighted RRF algorithm from rag-det.ipynb.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

from rag_detective.config import default_config
from rag_detective.models import DocumentChunk, SearchResult

logger = logging.getLogger("uvicorn.error")


def weighted_rrf_scores(
    combined_ids: Sequence[str],
    sparse_ids: Sequence[str],
    dense_ids: Sequence[str],
    alpha: float = 0.7,
    k: int = 60,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Computes Weighted Reciprocal Rank Fusion (Weighted RRF) scores for document IDs.
    Directly corresponds to `weighted_RRF(comb_list, sparse_list, dense_list, top_k, alpha, k)`
    from rag-det.ipynb.

    Formula:
        score(doc_id) = alpha * (1 / (k + rank_sparse)) + (1 - alpha) * (1 / (k + rank_dense))

    Args:
        combined_ids: Deduplicated list of document IDs from both retrieval methods.
        sparse_ids: Ordered list of document IDs from sparse (BM25) retriever (rank 1 to N).
        dense_ids: Ordered list of document IDs from dense (vector) retriever (rank 1 to M).
        alpha: Weight given to sparse retrieval (0.0 to 1.0). Default: 0.7.
        k: Smoothing constant to penalize lower ranks without zeroing. Default: 60.
        top_k: Number of highest-ranked items to return. Default: 3.

    Returns:
        List of (doc_id, rrf_score) tuples, sorted in descending order of score.
    """
    sparse_ranks = {doc_id: idx + 1 for idx, doc_id in enumerate(sparse_ids)}
    dense_ranks = {doc_id: idx + 1 for idx, doc_id in enumerate(dense_ids)}

    scores: Dict[str, float] = {}
    for doc_id in combined_ids:
        rank_sparse = sparse_ranks.get(doc_id, float("inf"))
        rank_dense = dense_ranks.get(doc_id, float("inf"))

        sparse_component = (1.0 / (k + rank_sparse)) if rank_sparse != float("inf") else 0.0
        dense_component = (1.0 / (k + rank_dense)) if rank_dense != float("inf") else 0.0

        score = (alpha * sparse_component) + ((1.0 - alpha) * dense_component)
        scores[doc_id] = score

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs[:top_k]


class WeightedRRF:
    """
    Object-oriented Reciprocal Rank Fusion operator for RAG Detective.
    Merges dense and sparse search results into a single unified hybrid ranking.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        k: Optional[int] = None,
    ) -> None:
        """
        Initialize WeightedRRF.

        Args:
            alpha: Weight for sparse retriever (0.0 to 1.0). Default from config: 0.7.
            k: Rank constant smoothing factor. Default from config: 60.
        """
        self.alpha = alpha if alpha is not None else default_config.rrf_alpha
        self.k = k if k is not None else default_config.rrf_k

    def fuse(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        chunk_map: Optional[Dict[str, DocumentChunk]] = None,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Fuses dense and sparse search results into top_k hybrid SearchResult objects.

        Args:
            dense_results: Ordered SearchResult list from dense vector retriever.
            sparse_results: Ordered SearchResult list from sparse BM25 retriever.
            chunk_map: Optional mapping from chunk_id -> DocumentChunk to resolve contents.
            top_k: Number of fused results to return.

        Returns:
            List of fused SearchResult objects ranked by Weighted RRF score.
        """
        dense_ids = [r.chunk_id for r in dense_results]
        sparse_ids = [r.chunk_id for r in sparse_results]

        # Preserve first-seen metadata and content from results
        content_map: Dict[str, Tuple[str, dict]] = {}
        for r in dense_results + sparse_results:
            if r.chunk_id not in content_map:
                content_map[r.chunk_id] = (r.content, r.metadata)

        # Build union of unique document IDs
        combined_ids = list(dict.fromkeys(sparse_ids + dense_ids))

        ranked_scores = weighted_rrf_scores(
            combined_ids=combined_ids,
            sparse_ids=sparse_ids,
            dense_ids=dense_ids,
            alpha=self.alpha,
            k=self.k,
            top_k=top_k,
        )

        fused_results: List[SearchResult] = []
        for rank, (doc_id, score) in enumerate(ranked_scores, start=1):
            if chunk_map and doc_id in chunk_map:
                chunk = chunk_map[doc_id]
                content = chunk.page_content
                metadata = dict(chunk.metadata)
            elif doc_id in content_map:
                content, metadata = content_map[doc_id]
            else:
                content, metadata = "", {}

            fused_results.append(
                SearchResult(
                    chunk_id=doc_id,
                    content=content,
                    metadata=metadata,
                    score=round(score, 6),
                    rank=rank,
                    engine="hybrid_rrf",
                )
            )

        logger.debug(
            "Fused %d dense and %d sparse candidates into %d top RRF results.",
            len(dense_results),
            len(sparse_results),
            len(fused_results),
        )
        return fused_results
