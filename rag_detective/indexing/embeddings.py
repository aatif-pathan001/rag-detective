"""
Embeddings manager for RAG Detective.
Uses ONNX Runtime via `fastembed` (`BAAI/bge-small-en-v1.5`) as the primary lightweight
embedding engine (~110 MB RAM, no PyTorch required), with automatic fallback to
`langchain-huggingface` (PyTorch) and a deterministic 384-d hashed embedder when offline.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

from rag_detective.config import default_config
from rag_detective.exceptions import IndexingError

logger = logging.getLogger("uvicorn.error")


class _FastTokenizerAdapter:
    """
    Adapter around Rust `tokenizers.Tokenizer` (used by ONNX / FastEmbed) so it exposes
    the same interface (`tokenizer(text)` and `convert_ids_to_tokens(ids)`) expected by
    `PipelineTracer._wordpieces()` for the dashboard's Tokenizer visualization panel.
    """

    def __init__(self, rust_tokenizer: Any) -> None:
        self._tk = rust_tokenizer
        self._last_id_to_tok: Dict[int, str] = {}

    def __call__(
        self, text: str, add_special_tokens: bool = True, truncation: bool = False
    ) -> Dict[str, List[int]]:
        encoded = self._tk.encode(text, add_special_tokens=add_special_tokens)
        ids = [int(i) for i in encoded.ids]
        toks = list(encoded.tokens)
        self._last_id_to_tok = dict(zip(ids, toks))
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        out: List[str] = []
        for i in ids:
            tok = self._last_id_to_tok.get(int(i))
            if tok is None and hasattr(self._tk, "id_to_token"):
                tok = self._tk.id_to_token(int(i))
            out.append(tok or str(i))
        return out


# Make the class name read cleanly in the dashboard's `{{ tokNote }}` label
_FastTokenizerAdapter.__name__ = "BertTokenizer (ONNX)"


class _ONNXFastEmbedder:
    """
    Lightweight ONNX Runtime wrapper for `BAAI/bge-small-en-v1.5` via `fastembed`.

    Why ONNX (`fastembed`) instead of PyTorch (`langchain-huggingface`)?
    1. Memory: Eliminates the ~300 MB PyTorch + Transformers runtime overhead, keeping total
       process RAM around ~120 MB (fits easily inside 512 MB free-tier cloud containers).
    2. Speed: Uses graph-optimized / quantized ONNX weights executed by C++ `onnxruntime`
       and Rust `tokenizers`, running 2-3x faster on CPU.
    3. Compatibility: Implements LangChain's `embed_documents()` and `embed_query()` methods
       so `ChromaVectorStore` and `PipelineTracer` work without any changes.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", normalize: bool = True) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.normalize = normalize
        # Initializes the ONNX session and downloads the compact ONNX model weights
        self._fe = TextEmbedding(model_name=model_name)

        # Expose `.client.tokenizer` so `PipelineTracer._tokenizer()` can inspect WordPiece tokens
        raw_tk = getattr(getattr(self._fe, "model", None), "tokenizer", None)
        self.tokenizer = _FastTokenizerAdapter(raw_tk) if raw_tk is not None else None
        self.client = self

    def _l2_normalize(self, vec: List[float]) -> List[float]:
        if not self.normalize:
            return vec
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Encodes a batch of document chunks into 384-d float vectors using ONNX Runtime."""
        if not texts:
            return []
        return [self._l2_normalize([float(x) for x in emb]) for emb in self._fe.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        """Encodes a single search query into a 384-d float vector using ONNX Runtime."""
        for emb in self._fe.query_embed(text):
            return self._l2_normalize([float(x) for x in emb])
        return []



class _HashedEmbedder:
    """
    Deterministic 384-d FNV-1a unigram + bigram embedding fallback for offline environments.
    """

    DIM = 384
    _WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'’-][A-Za-z0-9]+)*")

    def __init__(self) -> None:
        self.tokenizer = None
        self.client = self

    @staticmethod
    def _fnv(s: str) -> int:
        h = 2166136261
        for ch in s:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def _encode_one(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        toks = [w.lower() for w in self._WORD_RE.findall(text or "") if len(w) > 1]
        for i, tok in enumerate(toks):
            h = self._fnv(tok)
            vec[h % self.DIM] += 1.0 if ((h >> 16) & 1) else -1.0
            if i > 0:
                h2 = self._fnv(f"{toks[i - 1]}_{tok}")
                vec[h2 % self.DIM] += 0.5 if ((h2 >> 16) & 1) else -0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._encode_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._encode_one(text)


class EmbeddingManager:
    """
    Manages loading and lifecycle of HuggingFace embedding models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name or default_config.embedding_model_name
        self.device = device or default_config.embedding_device
        self.normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else default_config.embedding_normalize
        )
        self._embed_model: Optional[Any] = None

    @property
    def model(self) -> Any:
        """Lazy-loaded embeddings model instance."""

        if self._embed_model is None:
            self._embed_model = self._init_model()
        return self._embed_model

    def _init_model(self) -> Any:
        # 1. Primary Engine: Lightweight ONNX Runtime (`fastembed`) — low RAM (~110 MB), fast CPU inference
        try:
            logger.info(
                "Initializing ONNX FastEmbed model '%s' on CPU...",
                self.model_name,
            )
            return _ONNXFastEmbedder(
                model_name=self.model_name,
                normalize=self.normalize_embeddings,
            )
        except Exception as onnx_err:
            logger.info(
                "ONNX FastEmbed unavailable (%s); trying langchain-huggingface fallback...",
                onnx_err,
            )

        # 2. Secondary Fallback: PyTorch HuggingFaceEmbeddings (if installed)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(
                "Initializing HuggingFaceEmbeddings with model '%s' on %s...",
                self.model_name,
                self.device,
            )
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": self.normalize_embeddings},
            )
        except Exception as e:
            logger.warning("Embedding models unavailable (%s); falling back to Python 384-d hashed embedder.", e)
            return _HashedEmbedder()
