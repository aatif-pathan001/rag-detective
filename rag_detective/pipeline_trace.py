"""
Stage-by-stage pipeline tracing for the RAG Detective web frontend.

The frontend visualises every stage of the RAG pipeline (extraction, chunking,
embedding, vector storage, BM25 indexing, query rewriting, dual retrieval, rank
fusion and synthesis). This module runs those stages with the *Python* backend
components owned by :class:`RAGDetectiveEngine` and yields one JSON-serialisable
event per step, so the API layer can stream them to the browser as NDJSON.

No retrieval logic lives in the browser: the frontend only renders these events.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from rag_detective.engine import RAGDetectiveEngine
from rag_detective.fusion.rrf import WeightedRRF
from rag_detective.models import DocumentChunk, SearchResult, SubQuery

logger = logging.getLogger("uvicorn.error")

Event = Dict[str, Any]

# Display-only stopword list (used for word-cloud / cluster labels, never for retrieval).
_DISPLAY_STOPWORDS = set(
    (
        "a an the and or but if of to in on at by for with from as is are was were be been being it its "
        "this that these those there here which who whom whose what when where why how not no nor so than "
        "too very can will just do does did done have has had having i me my we our you your he him his she "
        "her they them their up down out over under again further then once all any both each few more most "
        "other some such only own same should now per via into about against between through during before "
        "after above below off also may must shall would could us one two"
    ).split()
)

_HEADER_KEY = re.compile(r"^Header\s+(\d)$")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_LIST_OR_TABLE = re.compile(r"^([-*+]|\d+\.)\s|^\|")


def _ms(t0: float) -> int:
    return int(round((time.perf_counter() - t0) * 1000))


def _file_kind(name: str) -> str:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext == "pdf":
        return "pdf"
    if ext == "docx":
        return "docx"
    if ext in ("md", "markdown"):
        return "md"
    return "txt"


def markdown_to_blocks(text: str) -> List[Dict[str, Any]]:
    """Turns Markdown into display blocks ({kind: h1|h2|h3|p, text, raw}) for the extract view."""
    out: List[Dict[str, Any]] = []
    para: List[str] = []

    def flush() -> None:
        if para:
            out.append({"kind": "p", "text": " ".join(para), "page": None})
            para.clear()

    for raw in text.splitlines():
        line = raw.strip()
        m = _MD_HEADING.match(line)
        if m:
            flush()
            out.append({"kind": f"h{min(3, len(m.group(1)))}", "text": m.group(2).strip(), "page": None})
        elif not line:
            flush()
        elif _MD_LIST_OR_TABLE.match(line):
            flush()
            out.append({"kind": "p", "text": line, "page": None, "raw": True})
        else:
            para.append(line)
    flush()
    return out


def chunk_headers(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Maps MarkdownHeaderTextSplitter metadata ('Header 1'...) to ordered {h1: .., h2: ..}."""
    found: List[Tuple[int, str]] = []
    for key, value in (metadata or {}).items():
        m = _HEADER_KEY.match(str(key))
        if m and value:
            found.append((int(m.group(1)), str(value)))
    return {f"h{lvl}": val for lvl, val in sorted(found)}


def pca3(vectors: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float]]:
    """Projects vectors to 3-D with PCA. Returns (coords scaled to [-1, 1], explained variance ratios)."""
    import numpy as np

    X = np.asarray(vectors, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return [], [0.0, 0.0, 0.0]
    C = X - X.mean(axis=0, keepdims=True)
    if n == 1 or not np.any(C):
        return [[0.0, 0.0, 0.0] for _ in range(n)], [0.0, 0.0, 0.0]

    _, S, Vt = np.linalg.svd(C, full_matrices=False)
    comps = Vt[:3]
    coords = C @ comps.T
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    total = float((S ** 2).sum()) or 1.0
    explained = [float(s ** 2 / total) for s in S[:3]] + [0.0] * max(0, 3 - len(S))
    mx = np.maximum(np.abs(coords).max(axis=0), 1e-9)
    coords = coords / mx
    return [[round(float(v), 5) for v in row] for row in coords], explained


def kmeans_cosine(vectors: Sequence[Sequence[float]], k: int, iters: int = 25) -> List[int]:
    """Spherical k-means with farthest-point initialisation (deterministic)."""
    import numpy as np

    X = np.asarray(vectors, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return []
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    k = max(1, min(k, n))

    centers = [X[0]]
    while len(centers) < k:
        sims = np.max(np.stack([X @ c for c in centers], axis=1), axis=1)
        centers.append(X[int(np.argmin(sims))])
    Cm = np.stack(centers)

    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        assign = np.argmax(X @ Cm.T, axis=1)
        for ci in range(k):
            members = X[assign == ci]
            if len(members):
                m = members.sum(axis=0)
                nm = np.linalg.norm(m)
                Cm[ci] = m / (nm or 1)
    return [int(a) for a in assign]


def top_neighbors(vectors: Sequence[Sequence[float]], top_n: int = 5) -> List[List[List[Any]]]:
    """Computes top-n cosine neighbors [[idx, similarity], ...] for each vector."""
    import numpy as np

    X = np.asarray(vectors, dtype=np.float64)
    n = X.shape[0]
    if n <= 1:
        return [[] for _ in range(n)]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    sim = X @ X.T
    out: List[List[List[Any]]] = []
    for i in range(n):
        order = [j for j in np.argsort(-sim[i]) if j != i][:top_n]
        out.append([[int(j), round(float(sim[i, j]), 4)] for j in order])
    return out


class PipelineTracer:
    """Runs the RAG Detective pipeline stage by stage and yields UI events."""

    def __init__(self, engine: RAGDetectiveEngine, embed_batch_size: int = 8, max_tokens_shown: int = 160) -> None:
        self.engine = engine
        self.embed_batch_size = embed_batch_size
        self.max_tokens_shown = max_tokens_shown

    # ------------------------------------------------------------------ helpers
    def _tokenizer(self) -> Any:
        """Returns the HuggingFace tokenizer behind the embedding model (if reachable)."""
        model = self.engine.embedding_manager.model
        st = getattr(model, "_client", None) or getattr(model, "client", None)
        return getattr(st, "tokenizer", None)

    def _wordpieces(self, tokenizer: Any, text: str) -> Tuple[List[List[Any]], int]:
        if tokenizer is not None:
            try:
                ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
                toks = tokenizer.convert_ids_to_tokens(ids)
                return [[t, int(i)] for t, i in zip(toks, ids)][: self.max_tokens_shown], len(ids)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Tokenizer failed, falling back to BM25 tokens: %s", e)
        words = self.engine.sparse_retriever._tokenize(text)
        return [[w, -1] for w in words[: self.max_tokens_shown]], len(words)

    def _bm25_snapshot(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Serialises the BM25 inverted index built by the backend."""
        bm25 = self.engine.sparse_retriever
        tokenize = bm25._tokenize
        transcript = bm25.doc_list
        dls = [len(tokenize(getattr(d, "page_content", "") or "")) for d in transcript]
        postings: Dict[str, Dict[str, Dict[str, int]]] = {}
        idfs: Dict[str, float] = {}
        
        for term, docs in bm25.inv_idx.items():
            
            postings[term] = {
                str(transcript[doc_idx - 1].id): {"freq": int(info["freq"]), "dl": dls[doc_idx - 1]}
                for doc_idx, info in docs.items()
            }
            idfs[term] = round(bm25._idf(term), 4)
        by_df = sorted(postings.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        top_terms = [
            [term, len(p)]
            for term, p in by_df
            if term not in _DISPLAY_STOPWORDS and len(term) > 1 and not term.isdigit()
        ][:40]
        return {
            "N": bm25.N,
            "avgdl": bm25.avgdl,
            "k1": bm25.k1,
            "b": bm25.b,
            "terms": len(postings),
            "tokens": int(sum(dls)),
            "postings": postings,
            "idfs": idfs,
            "top_terms": top_terms,
        }

    def _cluster_labels(self, chunks: List[DocumentChunk], assign: List[int], k: int) -> List[Dict[str, Any]]:
        bm25 = self.engine.sparse_retriever
        labels = []
        for ci in range(k):
            counts: Dict[str, int] = {}
            size = 0
            for chunk, a in zip(chunks, assign):
                if a != ci:
                    continue
                size += 1
                for t in bm25._tokenize(chunk.page_content):
                    if t in _DISPLAY_STOPWORDS or len(t) < 2 or t[0].isdigit():
                        continue
                    counts[t] = counts.get(t, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: kv[1] * bm25._idf(kv[0]), reverse=True)[:2]
            labels.append({"label": " · ".join(t for t, _ in ranked) or "misc", "size": size})
        return labels

    # ---------------------------------------------------------------- ingestion
    def ingest(self, data: bytes, filename: str, reset: bool = True) -> Iterator[Event]:
        """
        Ingests a document and yields one event per pipeline stage.

        Event shapes:
            {"event": "stage", "stage": <name>, "status": "running"}
            {"event": "stage", "stage": <name>, "status": "done", "ms": int, "data": {...}}
            {"event": "embed_progress", "offset": int, "vectors": [[...]], "tokens": [...]}
            {"event": "log", "message": str}
            {"event": "error", "stage": <name>, "message": str}
            {"event": "complete", "ms": int}
        """
        engine = self.engine
        t_total = time.perf_counter()
        stage = "upload"
        try:
            # 1. Upload / fingerprint ------------------------------------------------
            yield {"event": "stage", "stage": "upload", "status": "running"}
            t0 = time.perf_counter()
            sha = hashlib.sha256(data).hexdigest()
            kind = _file_kind(filename)
            yield {
                "event": "stage", "stage": "upload", "status": "done", "ms": _ms(t0),
                "data": {"name": filename, "size": len(data), "kind": kind, "sha": sha},
            }

            # Always clear previous collection and create a new empty vector store for every new case
            engine.reset()
            yield {"event": "log", "message": f"corpus reset · new empty vector store '{engine.vector_store.collection_name}'"}

            # 2. Extract (PDF/DOCX/MD/TXT -> Markdown) --------------------------------
            stage = "extract"
            yield {"event": "stage", "stage": "extract", "status": "running"}
            t0 = time.perf_counter()
            suffix = Path(filename).suffix or ".txt"
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / f"upload{suffix}"
                tmp_path.write_bytes(data)
                markdown_text = engine.ingestion.loader.load(tmp_path)
            blocks = markdown_to_blocks(markdown_text)
            if not blocks:
                raise ValueError("no extractable text in this document")
            note = {
                "pdf": f"{type(engine.ingestion.loader).__name__} · pdfmuse → Markdown (pypdf fallback)",
                "docx": "python-docx · Word heading styles → #, ##, ###",
                "md": "native Markdown · headings read from # markers",
                "txt": "plain text read as Markdown",
            }[kind]
            yield {
                "event": "stage", "stage": "extract", "status": "done", "ms": _ms(t0),
                "data": {"lines": blocks, "note": note, "chars": len(markdown_text)},
            }

            # 3. Chunk (MarkdownHeaderTextSplitter) ----------------------------------
            stage = "chunk"
            yield {"event": "stage", "stage": "chunk", "status": "running"}
            t0 = time.perf_counter()
            start_idx = len(engine._all_chunks) + 1
            if hasattr(engine.ingestion.splitter, "start_index"):
                engine.ingestion.splitter.start_index = start_idx
            chunks = engine.ingestion.splitter.split(markdown_text, source_name=filename)
            if not chunks:
                raise ValueError("the splitter produced no chunks")
            max_chunks = getattr(engine.config, "max_chunks_per_doc", 1500)
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
                yield {
                    "event": "log",
                    "message": f"capped document at {max_chunks} chunks to prevent resource exhaustion",
                }
            chunk_ms = _ms(t0)
            chunk_payload = [
                {
                    "id": c.id,
                    "text": c.page_content,
                    "meta": chunk_headers(c.metadata),
                    "chars": len(c.page_content),
                    # rough estimate; replaced by the real tokenizer count during embedding
                    "tok": round(len(c.page_content.split()) * 1.3),
                    "page": c.metadata.get("page"),
                }
                for c in chunks
            ]
            yield {
                "event": "stage", "stage": "chunk", "status": "done", "ms": chunk_ms,
                "data": {"chunks": chunk_payload, "splitter": type(engine.ingestion.splitter).__name__},
            }

            # 4. Embed (HuggingFace sentence-transformer) -----------------------------
            stage = "embed"
            yield {
                "event": "stage", "stage": "embed", "status": "running",
                "data": {"model": engine.config.embedding_model_name, "device": engine.config.embedding_device},
            }
            t0 = time.perf_counter()
            model = engine.embedding_manager.model  # lazy load (may download weights)
            tokenizer = None
            try:
                tokenizer = self._tokenizer()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Tokenizer unavailable: %s", e)
            yield {"event": "log", "message": f"embedding model ready · {engine.config.embedding_model_name}"}
            vectors: List[List[float]] = []
            B = self.embed_batch_size
            for i in range(0, len(chunks), B):
                batch = chunks[i : i + B]
                embs = model.embed_documents([c.page_content for c in batch])
                embs = [[float(x) for x in e] for e in embs]
                vectors.extend(embs)
                views = [self._wordpieces(tokenizer, c.page_content) for c in batch]
                yield {
                    "event": "embed_progress",
                    "offset": i,
                    "total": len(chunks),
                    "vectors": [[round(x, 5) for x in e] for e in embs],
                    "tokens": [v[0] for v in views],
                    "token_counts": [v[1] for v in views],
                }
            dim = len(vectors[0]) if vectors else 0
            yield {
                "event": "stage", "stage": "embed", "status": "done", "ms": _ms(t0),
                "data": {"count": len(vectors), "dim": dim, "model": engine.config.embedding_model_name,
                         "tokenizer": type(tokenizer).__name__ if tokenizer is not None else None},
            }

            # 5. Store (Chroma) + PCA / k-means for the 3-D view ----------------------
            stage = "store"
            yield {"event": "stage", "stage": "store", "status": "running"}
            t0 = time.perf_counter()
            # Persists into Chroma and rebuilds the BM25 index over the corpus.
            engine._register_and_index_chunks(chunks, source_name=filename, embeddings=vectors)
            coords, explained = pca3(vectors)
            n = len(chunks)
            k = 1 if n < 4 else max(2, min(6, round(math.sqrt(n / 2))))
            assign = kmeans_cosine(vectors, k)
            legend = self._cluster_labels(chunks, assign, k)
            neighbors = top_neighbors(vectors, top_n=5)
            yield {
                "event": "stage", "stage": "store", "status": "done", "ms": _ms(t0),
                "data": {
                    "coords": coords, "explained": explained, "assign": assign, "legend": legend,
                    "neighbors": neighbors,
                    "collection": engine.vector_store.collection_name,
                    "persist_dir": engine.vector_store.persist_directory,
                    "distance": engine.vector_store.distance_metric,
                    "count": engine.vector_store.count(),
                },
            }

            # 6. Index (BM25 inverted index, built by the backend) --------------------
            stage = "index"
            yield {"event": "stage", "stage": "index", "status": "running"}
            t0 = time.perf_counter()
            snapshot = self._bm25_snapshot(chunks)
            yield {"event": "stage", "stage": "index", "status": "done", "ms": _ms(t0), "data": snapshot}

            yield {"event": "complete", "ms": _ms(t_total)}
        except Exception as e:
            logger.exception("Ingestion failed at stage '%s'", stage)
            yield {"event": "error", "stage": stage, "message": str(e) or type(e).__name__}

    # ------------------------------------------------------------- investigation
    def investigate(
        self,
        user_query: str,
        top_k: int = 5,
        alpha: float = 0.3,
        k: int = 60,
        candidate_k: Optional[int] = None,
        chat_history: Optional[str] = None,
        document_context: Optional[str] = None,
    ) -> Iterator[Event]:
        """
        Runs query rewriting -> dense + sparse retrieval -> weighted RRF -> synthesis,
        yielding the intermediate results of every stage.

        `alpha` is the sparse (BM25) weight in Weighted RRF, matching the backend convention.
        """
        engine = self.engine
        stage = "rewrite"
        cand_k = max(1, candidate_k or max(top_k, engine.config.retrieval_dense_k))
        try:
            if not engine._all_chunks:
                raise ValueError("no documents are indexed yet — ingest a document first")

            # 1. Rewrite --------------------------------------------------------------
            yield {"event": "stage", "stage": "rewrite", "status": "running"}
            t0 = time.perf_counter()
            if document_context is None:
                heads: List[str] = []
                for c in engine._all_chunks:
                    for v in chunk_headers(c.metadata).values():
                        if v not in heads:
                            heads.append(v)
                document_context = "Section headings: " + "; ".join(heads[:30]) if heads else ""
            rw = engine.query_rewriter.rewrite(
                user_message=user_query, chat_history=chat_history, document_context=document_context
            )
            sub_queries = rw.sub_queries or [SubQuery(dense=user_query, sparse=user_query)]
            passthrough = len(sub_queries) == 1 and sub_queries[0].dense.strip() == user_query.strip()
            mode = "fallback" if (engine.query_rewriter.client is None or passthrough) else engine.config.gemini_model_name
            tokenize = engine.sparse_retriever._tokenize
            terms: List[str] = []
            for sq in sub_queries:
                for t in tokenize(sq.sparse):
                    if t not in terms:
                        terms.append(t)
            yield {
                "event": "stage", "stage": "rewrite", "status": "done", "ms": _ms(t0),
                "data": {
                    "needs_retrieval": rw.needs_retrieval,
                    "needs_clarification": rw.needs_clarification,
                    "clarifying_question": rw.clarifying_question,
                    "sub_queries": [sq.model_dump() for sq in sub_queries],
                    "dense": " · ".join(sq.dense for sq in sub_queries),
                    "sparse": " · ".join(sq.sparse for sq in sub_queries),
                    "terms": terms,
                    "mode": mode,
                },
            }

            if not rw.needs_retrieval or rw.needs_clarification:
                answer = (
                    rw.clarifying_question or "Your query requires clarification before I can investigate the documents."
                    if rw.needs_clarification
                    else "Hello! I am RAG Detective. How can I help investigate your documents today?"
                )
                yield {"event": "skip", "reason": "clarification" if rw.needs_clarification else "no_retrieval",
                       "answer": answer}
                return

            # 2. Dense + sparse retrieval ----------------------------------------------
            stage = "recall"
            yield {"event": "stage", "stage": "recall", "status": "running"}
            per_sq: List[Tuple[List[SearchResult], List[SearchResult]]] = []
            t_dense = t_sparse = 0.0
            for sq in sub_queries:
                t0 = time.perf_counter()
                d = engine.dense_retriever.retrieve(sq.dense, top_k=cand_k)
                t_dense += time.perf_counter() - t0
                t0 = time.perf_counter()
                s = engine.sparse_retriever.retrieve(sq.sparse, top_k=cand_k)
                t_sparse += time.perf_counter() - t0
                per_sq.append((d, s))

            def merge(lists: List[List[SearchResult]]) -> List[Dict[str, Any]]:
                best: Dict[str, float] = {}
                for lst in lists:
                    for r in lst:
                        best[r.chunk_id] = max(best.get(r.chunk_id, float("-inf")), r.score)
                return [{"chunk_id": cid, "score": sc} for cid, sc in sorted(best.items(), key=lambda kv: -kv[1])]

            dense_list = merge([d for d, _ in per_sq])
            sparse_list = merge([s for _, s in per_sq])
            yield {
                "event": "stage", "stage": "recall", "status": "done",
                "ms": int(round((t_dense + t_sparse) * 1000)),
                "data": {"dense": dense_list, "sparse": sparse_list,
                         "dense_ms": round(t_dense * 1000, 1), "sparse_ms": round(t_sparse * 1000, 1),
                         "candidate_k": cand_k},
            }

            # 3. Weighted reciprocal rank fusion ---------------------------------------
            stage = "fuse"
            yield {"event": "stage", "stage": "fuse", "status": "running"}
            t0 = time.perf_counter()
            fusion = WeightedRRF(alpha=alpha, k=k)
            fused_rows: List[Dict[str, Any]] = []
            evidence_pool: Dict[str, SearchResult] = {}
            seen = set()
            for d, s in per_sq:
                fused = fusion.fuse(d, s, chunk_map=engine._chunk_map, top_k=len(d) + len(s))
                d_rank = {r.chunk_id: i + 1 for i, r in enumerate(d)}
                s_rank = {r.chunk_id: i + 1 for i, r in enumerate(s)}
                for r in fused:
                    if r.chunk_id in seen:
                        continue
                    seen.add(r.chunk_id)
                    dR, sR = d_rank.get(r.chunk_id, 0), s_rank.get(r.chunk_id, 0)
                    fused_rows.append({
                        "chunk_id": r.chunk_id, "score": r.score,
                        "cd": (1.0 - alpha) / (k + dR) if dR else 0.0,
                        "cs": alpha / (k + sR) if sR else 0.0,
                        "dR": dR, "sR": sR,
                    })
                    evidence_pool[r.chunk_id] = r
            fused_rows.sort(key=lambda row: row["score"], reverse=True)
            final_ids = [row["chunk_id"] for row in fused_rows[:top_k]]
            evidence = [evidence_pool[cid] for cid in final_ids]
            yield {
                "event": "stage", "stage": "fuse", "status": "done", "ms": _ms(t0),
                "data": {"fused": fused_rows, "alpha": alpha, "k": k, "top_k": top_k, "evidence_ids": final_ids},
            }

            # 4. Synthesis -----------------------------------------------------------------
            stage = "synth"
            yield {"event": "stage", "stage": "synth", "status": "running"}
            t0 = time.perf_counter()
            answer = engine.synthesizer.synthesize(user_query=user_query, retrieved_docs=evidence, cite_sources=True)
            synth_mode = getattr(engine.synthesizer, "last_mode", engine.config.gemini_model_name)
            yield {
                "event": "stage", "stage": "synth", "status": "done", "ms": _ms(t0),
                "data": {"answer": answer, "mode": synth_mode},
            }
            yield {"event": "complete"}
        except Exception as e:
            logger.exception("Investigation failed at stage '%s'", stage)
            yield {"event": "error", "stage": stage, "message": str(e) or type(e).__name__}
