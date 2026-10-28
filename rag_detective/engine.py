"""
Main RAG Detective Engine.
Orchestrates ingestion, dual-index representation, query rewriting, hybrid retrieval,
weighted rank fusion, and grounded evidence synthesis.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rag_detective.config import RAGDetectiveConfig, default_config
from rag_detective.fusion.rrf import WeightedRRF
from rag_detective.generation.synthesizer import DetectiveSynthesizer
from rag_detective.indexing.embeddings import EmbeddingManager
from rag_detective.indexing.vector_store import ChromaVectorStore
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

logger = logging.getLogger("uvicorn.error")


class RAGDetectiveEngine:
    """
    Unified backend engine for the RAG Detective application.
    """

    def __init__(
        self,
        config: Optional[RAGDetectiveConfig] = None,
        ingestion_pipeline: Optional[IngestionPipeline] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        synthesizer: Optional[DetectiveSynthesizer] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> None:
        """
        Initialize the RAG Detective Engine.
        """
        self.config = config or default_config

        # 1. Ingestion
        self.ingestion = ingestion_pipeline or IngestionPipeline()

        # 2. Embedding & Vector Store (Dense)
        self.embedding_manager = embedding_manager or EmbeddingManager(
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
            normalize_embeddings=self.config.embedding_normalize,
        )
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=self.config.chroma_collection_name,
            embedding_manager=self.embedding_manager,
            persist_directory=self.config.chroma_persist_dir,
            distance_metric=self.config.chroma_distance_metric,
        )

        # 3. Sparse Retriever (BM25)
        self.sparse_retriever = sparse_retriever or BM25Retriever(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )

        # 4. Dense Retriever
        self.dense_retriever = DenseRetriever(
            vector_store=self.vector_store,
            default_k=self.config.retrieval_dense_k,
        )

        # 5. Hybrid Retriever & Fusion
        self.fusion_engine = WeightedRRF(
            alpha=self.config.rrf_alpha,
            k=self.config.rrf_k,
        )
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            fusion_engine=self.fusion_engine,
            dense_k=self.config.retrieval_dense_k,
            sparse_k=self.config.retrieval_sparse_k,
            final_k=self.config.final_top_k,
        )

        # 6. Query Rewriter & Synthesizer
        self.query_rewriter = query_rewriter or QueryRewriter(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_model_name,
            temperature=self.config.rewriter_temperature,
            top_k=self.config.rewriter_top_k,
        )
        self.synthesizer = synthesizer or DetectiveSynthesizer(
            api_key=self.config.gemini_api_key,
            model_name=self.config.gemini_model_name,
            temperature=self.config.synthesizer_temperature,
            top_k=self.config.synthesizer_top_k,
        )

        # In-memory document chunk registry (for fast resolution & BM25)
        self._all_chunks: List[DocumentChunk] = []
        self._chunk_map: Dict[str, DocumentChunk] = {}

        # Always start with a clean, empty vector store (do not retain previous user files)
        self.reset()

    def _warmup_from_vector_store(self) -> None:
        """Loads existing chunks from persisted Chroma DB into BM25 and chunk map."""
        try:
            persisted_chunks = self.vector_store.get_all_chunks()
            if persisted_chunks:
                self._all_chunks = persisted_chunks
                self._chunk_map = {c.id: c for c in persisted_chunks}
                self.sparse_retriever.index_documents(self._all_chunks)
                self.hybrid_retriever.set_indexed_chunks(self._all_chunks)
                logger.info("Warmed up RAG Detective with %d chunks from vector store.", len(persisted_chunks))
        except Exception as e:
            logger.debug("Vector store warmup skipped: %s", e)

    def ingest_file(self, file_path: Union[str, Path], reset: bool = True) -> IngestionResponse:
        """
        Ingests a document file (PDF, Markdown, or text) into dense and sparse indexes.

        Args:
            file_path: Path to target file.
            reset: Whether to clear the existing collection and create a new empty vector store first.

        Returns:
            IngestionResponse with chunk counts and metadata.
        """
        if reset:
            self.reset()
        start_chunk_idx = len(self._all_chunks) + 1
        new_chunks = self.ingestion.process_file(file_path, start_chunk_idx=start_chunk_idx)
        return self._register_and_index_chunks(new_chunks, source_name=str(file_path))

    def ingest_markdown(
        self, markdown_text: str, source_name: str = "direct_markdown", reset: bool = True
    ) -> IngestionResponse:
        """
        Ingests raw markdown text into dense and sparse indexes.
        """
        if reset:
            self.reset()
        start_chunk_idx = len(self._all_chunks) + 1
        new_chunks = self.ingestion.process_markdown(
            markdown_text, source_name=source_name, start_chunk_idx=start_chunk_idx
        )
        return self._register_and_index_chunks(new_chunks, source_name=source_name)

    def _register_and_index_chunks(
        self,
        chunks: List[DocumentChunk],
        source_name: str,
        embeddings: Optional[List[List[float]]] = None,
    ) -> IngestionResponse:
        """Indexes chunks into Chroma and BM25 (optionally with precomputed embeddings)."""
        if not chunks:
            return IngestionResponse(
                source=source_name,
                total_chunks=0,
                chunk_ids=[],
                metadata={"message": "No chunks extracted."},
            )

        max_chunks = getattr(self.config, "max_chunks_per_doc", 1500)
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            if embeddings is not None:
                embeddings = embeddings[:max_chunks]

        # 1. Index in Chroma Vector Store
        if embeddings is not None:
            chunk_ids = self.vector_store.add_chunks_with_embeddings(chunks, embeddings)
        else:
            chunk_ids = self.vector_store.add_chunks(chunks)

        # 2. Update chunk registry
        self._all_chunks.extend(chunks)
        for chunk in chunks:
            self._chunk_map[chunk.id] = chunk

        # 3. Re-index BM25 sparse retriever with all corpus chunks
        self.sparse_retriever.index_documents(self._all_chunks)
        self.hybrid_retriever.set_indexed_chunks(self._all_chunks)

        logger.info(
            "Engine registered and indexed %d new chunks from '%s' (total corpus: %d chunks).",
            len(chunks),
            source_name,
            len(self._all_chunks),
        )

        return IngestionResponse(
            source=source_name,
            total_chunks=len(chunks),
            chunk_ids=chunk_ids,
            metadata={
                "corpus_total_chunks": len(self._all_chunks),
                "collection_name": self.vector_store.collection_name,
            },
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        alpha: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Direct search interface returning retrieved chunks.

        Args:
            query: The search query string.
            top_k: Number of search results to return.
            mode: 'hybrid', 'dense', or 'sparse'.
            alpha: Weight for sparse retrieval in hybrid mode.
        """
        k = top_k or self.config.final_top_k
        if mode == "dense":
            return self.dense_retriever.retrieve(query, top_k=k)
        elif mode == "sparse":
            return self.sparse_retriever.retrieve(query, top_k=k)
        else:
            return self.hybrid_retriever.retrieve_hybrid(
                dense_query=query, sparse_query=query, top_k=k, alpha=alpha
            )

    def investigate(
        self,
        request: Union[InvestigationRequest, str],
    ) -> InvestigationResult:
        """
        Executes the end-to-end RAG Detective pipeline.

        Pipeline Stages:
        1. Query Rewriting & Intent Decomposition (Google Gemini)
        2. Dual Dense and Sparse Retrieval
        3. Weighted Reciprocal Rank Fusion (Weighted RRF)
        4. Context-Bounded Evidence Synthesis (Google Gemini)

        Args:
            request: InvestigationRequest instance or raw query string.

        Returns:
            Structured InvestigationResult containing answer and evidence citations.
        """
        t_start = time.perf_counter()

        if isinstance(request, str):
            req = InvestigationRequest(user_query=request)
        else:
            req = request

        # Step 1: Query Rewriting
        rewrite_result = self.query_rewriter.rewrite(
            user_message=req.user_query,
            chat_history=req.chat_history,
            document_context=req.document_context,
        )

        # Handle non-retrieval chit-chat
        if not rewrite_result.needs_retrieval:
            return InvestigationResult(
                query=req.user_query,
                needs_retrieval=False,
                needs_clarification=False,
                clarifying_question=None,
                sub_queries=[],
                answer="Hello! I am RAG Detective. How can I help investigate your documents today?",
                evidence=[],
                retrieved_chunk_ids=[],
                metadata={"total_time_sec": round(time.perf_counter() - t_start, 3)},
            )

        # Handle ambiguous queries requiring clarification
        if rewrite_result.needs_clarification:
            return InvestigationResult(
                query=req.user_query,
                needs_retrieval=True,
                needs_clarification=True,
                clarifying_question=rewrite_result.clarifying_question,
                sub_queries=rewrite_result.sub_queries,
                answer=rewrite_result.clarifying_question
                or "Your query requires clarification before I can investigate the documents.",
                evidence=[],
                retrieved_chunk_ids=[],
                metadata={"total_time_sec": round(time.perf_counter() - t_start, 3)},
            )

        # Step 2 & 3: Retrieval & Fusion for each sub-query
        sub_queries = rewrite_result.sub_queries or [
            SubQuery(dense=req.user_query, sparse=req.user_query)
        ]

        fused_candidates: List[SearchResult] = []
        seen_ids = set()

        for sq in sub_queries:
            sub_results = self.hybrid_retriever.retrieve_hybrid(
                dense_query=sq.dense,
                sparse_query=sq.sparse,
                top_k=req.top_k,
                alpha=req.alpha,
            )
            for r in sub_results:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    fused_candidates.append(r)

        # Re-sort candidates by score descending and take top_k
        fused_candidates.sort(key=lambda item: item.score, reverse=True)
        final_evidence = fused_candidates[: req.top_k]

        # Step 4: Detective Synthesis
        synthesized_answer = self.synthesizer.synthesize(
            user_query=req.user_query,
            retrieved_docs=final_evidence,
        )

        t_elapsed = round(time.perf_counter() - t_start, 3)

        return InvestigationResult(
            query=req.user_query,
            needs_retrieval=True,
            needs_clarification=False,
            clarifying_question=None,
            sub_queries=sub_queries,
            answer=synthesized_answer,
            evidence=final_evidence,
            retrieved_chunk_ids=[e.chunk_id for e in final_evidence],
            metadata={
                "total_time_sec": t_elapsed,
                "evidence_count": len(final_evidence),
                "alpha_used": req.alpha,
                "k_constant": req.k,
            },
        )

    def reset(self) -> None:
        """Clears the whole corpus: Chroma collection, BM25 index and chunk registry."""
        self.vector_store.reset_collection()
        self._all_chunks = []
        self._chunk_map = {}
        self.sparse_retriever.index_documents([])
        self.hybrid_retriever.set_indexed_chunks([])
        logger.info("RAG Detective corpus reset.")

    def create_session_engine(self, collection_name: str) -> "RAGDetectiveEngine":
        """
        Creates a lightweight, session-isolated RAGDetectiveEngine that shares the loaded
        embedding model and LLM clients with this base engine, but owns a dedicated
        Chroma collection, BM25 index, and chunk registry.
        """
        session_vector_store = ChromaVectorStore(
            collection_name=collection_name,
            embedding_manager=self.embedding_manager,
            persist_directory=self.config.chroma_persist_dir,
            distance_metric=self.config.chroma_distance_metric,
        )
        return RAGDetectiveEngine(
            config=self.config,
            embedding_manager=self.embedding_manager,
            vector_store=session_vector_store,
            query_rewriter=self.query_rewriter,
            synthesizer=self.synthesizer,
        )

    def destroy(self) -> None:
        """Deletes the underlying vector store collection and clears in-memory state."""
        self.vector_store.delete_collection()
        self._all_chunks = []
        self._chunk_map = {}
        self.sparse_retriever.index_documents([])
        self.hybrid_retriever.set_indexed_chunks([])

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics regarding active indexes and models."""
        return {
            "total_corpus_chunks": len(self._all_chunks),
            "chroma_collection": self.vector_store.collection_name,
            "embedding_model": self.config.embedding_model_name,
            "llm_model": self.config.gemini_model_name,
            "rrf_alpha": self.config.rrf_alpha,
            "rrf_k": self.config.rrf_k,
            "bm25_indexed_docs": self.sparse_retriever.N,
            "bm25_avgdl": self.sparse_retriever.avgdl,
        }
