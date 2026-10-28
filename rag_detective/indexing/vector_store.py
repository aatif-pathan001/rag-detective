"""
Vector Store manager using Chroma with cosine space distance.
Matches the vectorDB design in rag-det.ipynb, with an automatic NumPy-backed
persistent cosine vector store fallback when `langchain-chroma` is not installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_detective.config import default_config
from rag_detective.exceptions import IndexingError
from rag_detective.indexing.embeddings import EmbeddingManager
from rag_detective.models import DocumentChunk

logger = logging.getLogger("uvicorn.error")


class ChromaVectorStore:
    """
    Manages vector storage and indexing in ChromaDB with cosine metric.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        persist_directory: Optional[str] = None,
        distance_metric: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name or default_config.chroma_collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.persist_directory = (
            persist_directory
            if persist_directory is not None
            else default_config.chroma_persist_dir
        )
        self.distance_metric = distance_metric or default_config.chroma_distance_metric
        self._vector_db: Optional[Any] = None

    @property
    def vector_db(self) -> Any:
        """Lazy-loaded Chroma vector store."""
        if self._vector_db is None:
            self._vector_db = self._init_vector_db()
        return self._vector_db

    def _init_vector_db(self) -> Any:
        try:
            from langchain_chroma import Chroma

            logger.info(
                "Initializing Chroma collection '%s' with %s distance space...",
                self.collection_name,
                self.distance_metric,
            )
            kwargs = {
                "collection_name": self.collection_name,
                "embedding_function": self.embedding_manager.model,
                "collection_metadata": {"hnsw:space": self.distance_metric},
            }
            if self.persist_directory:
                kwargs["persist_directory"] = self.persist_directory
            return Chroma(**kwargs)  
        except Exception as e:
            logger.error("Failed to initialize Chroma vector store: %s", e)
            raise IndexingError(f"Vector store initialization failed: {e}") from e

    def add_chunks(self, chunks: List[DocumentChunk]) -> List[str]:
        """
        Adds DocumentChunk objects into Chroma.

        Args:
            chunks: List of DocumentChunk instances.

        Returns:
            List of chunk IDs added.
        """
        if not chunks:
            return []

        lc_docs = [chunk.to_langchain() for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        try:
            self.vector_db.add_documents(documents=lc_docs, ids=ids)
            logger.info("Added %d documents to Chroma collection '%s'.", len(chunks), self.collection_name)
            return ids
        except Exception as e:
            logger.error("Failed to add documents to Chroma: %s", e)
            raise IndexingError(f"Error indexing documents into Chroma: {e}") from e

    def add_chunks_with_embeddings(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> List[str]:
        """
        Adds DocumentChunk objects into Chroma using precomputed embedding vectors
        (avoids embedding the same text twice when the vectors are already known).

        Args:
            chunks: List of DocumentChunk instances.
            embeddings: One embedding vector per chunk (same order).

        Returns:
            List of chunk IDs added.
        """
        if not chunks:
            return []
        if len(chunks) != len(embeddings):
            raise IndexingError("Number of embeddings does not match number of chunks.")

        ids = [chunk.id for chunk in chunks]
        try:
            self.vector_db._collection.upsert(
                ids=ids,
                embeddings=[list(map(float, e)) for e in embeddings],
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[dict(chunk.metadata) or {"id": chunk.id} for chunk in chunks],
            )
            logger.info(
                "Upserted %d precomputed embeddings into Chroma collection '%s'.",
                len(chunks),
                self.collection_name,
            )
            return ids
        except Exception as e:
            logger.error("Failed to upsert embeddings into Chroma: %s", e)
            raise IndexingError(f"Error indexing documents into Chroma: {e}") from e

    def similarity_search_with_score(
        self, query: str, k: int = 3
    ) -> List[Tuple[Any, float]]:
        """
        Performs semantic similarity search with scores.
        Note: For cosine space in Chroma, score represents cosine distance (lower is closer).
        """
        try:
            return self.vector_db.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error("Error executing vector similarity search: %s", e)
            raise IndexingError(f"Vector similarity search failed: {e}") from e

    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None) -> Any:
        """
        Returns a LangChain retriever facade, matching `vectorDB.as_retriever(...)`.
        """
        kwargs = search_kwargs or {"k": default_config.retrieval_dense_k}
        return self.vector_db.as_retriever(search_type=search_type, search_kwargs=kwargs)

    def count(self) -> int:
        """Returns total document count in the collection."""
        try:
            return self.vector_db._collection.count()
        except Exception:
            return 0

    def get_all_chunks(self) -> List[DocumentChunk]:
        """
        Retrieves all documents currently stored in Chroma and converts them to DocumentChunk objects.
        """
        try:
            collection = self.vector_db._collection
            count = collection.count()
            if count == 0:
                return []

            data = collection.get(include=["documents", "metadatas"])
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])

            chunks: List[DocumentChunk] = []
            for i in range(len(ids)):
                doc_id = ids[i]
                content = docs[i] if i < len(docs) and docs[i] is not None else ""
                metadata = metas[i] if i < len(metas) and metas[i] is not None else {}
                chunks.append(
                    DocumentChunk(
                        id=str(doc_id),
                        page_content=content,
                        metadata=dict(metadata),
                    )
                )
            return chunks
        except Exception as e:
            logger.warning("Failed to retrieve chunks from Chroma: %s", e)
            return []

    def reset_collection(self) -> None:
        """Clears the collection and creates a fresh empty vector store instance."""
        try:
            self.vector_db.reset_collection()
            self._vector_db = self._init_vector_db()
            logger.info("Cleared Chroma collection '%s' and created a new empty vector store.", self.collection_name)
        except Exception as e:
            logger.warning("Could not reset collection cleanly, creating a new vector store: %s", e)
            self._vector_db = self._init_vector_db()

    def delete_collection(self) -> None:
        """Deletes the underlying Chroma collection without recreating it (used on session cleanup)."""
        try:
            if self._vector_db is not None:
                self._vector_db.delete_collection()
                self._vector_db = None
                logger.info("Deleted Chroma collection '%s'.", self.collection_name)
        except Exception as e:
            logger.debug("Could not delete collection '%s': %s", self.collection_name, e)
