"""
REST API route definitions for RAG Detective backend.
Includes per-session isolation and sliding-window IP rate limiting.
"""

from collections import OrderedDict
import hashlib
import json
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover - optional until create_router is called
    APIRouter = File = Form = HTTPException = Request = UploadFile = StreamingResponse = None  # type: ignore

from rag_detective.engine import RAGDetectiveEngine
from rag_detective.models import (
    IngestionResponse,
    InvestigationRequest,
    InvestigationResult,
    SearchResult,
)
from rag_detective.pipeline_trace import PipelineTracer

logger = logging.getLogger("uvicorn.error")

_SAFE_SESSION_RE = re.compile(r"[^a-zA-Z0-9_-]")
_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
_READ_CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks


async def _read_validated_upload(file: UploadFile, max_bytes: int) -> Tuple[bytes, str, str]:
    """
    Validates file extension and reads upload stream in bounded chunks up to `max_bytes`.
    Prevents unbounded memory exhaustion (DoS) from oversized uploads.
    """
    raw_name = Path(file.filename or "document.txt").name
    suffix = Path(raw_name).suffix.lower() or ".txt"
    if suffix not in _ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Allowed formats: {allowed_str}",
        )

    chunks: List[bytes] = []
    total = 0
    while True:
        piece = await file.read(_READ_CHUNK_SIZE)
        if not piece:
            break
        total += len(piece)
        if total > max_bytes:
            max_mb = round(max_bytes / (1024 * 1024), 1)
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the maximum allowed upload size of {max_mb:g} MB.",
            )
        chunks.append(piece)

    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return data, raw_name, suffix


def _client_ip(request: Request) -> str:
    """Extracts the client IP address, respecting X-Forwarded-For behind proxies."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _session_id_from_request(request: Request) -> str:
    """Resolves a sanitized session identifier from X-Session-Id header or client fingerprint."""
    raw = (request.headers.get("x-session-id") or "").strip()
    cleaned = _SAFE_SESSION_RE.sub("", raw)[:32]
    if len(cleaned) >= 6:
        return cleaned
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")
    return hashlib.sha256(f"{ip}:{ua}".encode("utf-8")).hexdigest()[:16]


class IPRateLimiter:
    """Thread-safe sliding-window rate limiter keyed by (ip, bucket)."""

    def __init__(self, window_seconds: int = 300) -> None:
        self.window = max(1, window_seconds)
        self._hits: Dict[Tuple[str, str], List[float]] = {}
        self._lock = threading.Lock()

    def check(self, ip: str, bucket: str, max_requests: int) -> None:
        now = time.monotonic()
        cutoff = now - self.window
        key = (ip, bucket)
        with self._lock:
            timestamps = [t for t in self._hits.get(key, []) if t > cutoff]
            if len(timestamps) >= max_requests:
                retry_after = max(1, int(round(self.window - (now - timestamps[0]))))
                self._hits[key] = timestamps
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({max_requests} {bucket} requests per {self.window}s). Please retry in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            timestamps.append(now)
            self._hits[key] = timestamps


class SessionEngineManager:
    """
    Manages isolated RAGDetectiveEngine instances per user session so concurrent
    visitors never overwrite or read each other's uploaded documents.
    """

    def __init__(self, base_engine: RAGDetectiveEngine) -> None:
        self.base_engine = base_engine
        self.ttl = max(60, base_engine.config.session_ttl_seconds)
        self.max_sessions = max(1, base_engine.config.max_active_sessions)
        self._sessions: "OrderedDict[str, Tuple[RAGDetectiveEngine, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def _evict_expired_locked(self, now: float) -> None:
        expired = [sid for sid, (_, ts) in self._sessions.items() if now - ts > self.ttl]
        for sid in expired:
            eng, _ = self._sessions.pop(sid)
            eng.destroy()
            logger.info("Expired idle session '%s'.", sid)

    def get(self, session_id: str) -> RAGDetectiveEngine:
        now = time.monotonic()
        with self._lock:
            self._evict_expired_locked(now)
            if session_id in self._sessions:
                eng, _ = self._sessions.pop(session_id)
                self._sessions[session_id] = (eng, now)
                return eng

            while len(self._sessions) >= self.max_sessions:
                oldest_sid, (oldest_eng, _) = self._sessions.popitem(last=False)
                oldest_eng.destroy()
                logger.info("Evicted least-recently-used session '%s'.", oldest_sid)

            collection_name = f"case_{session_id[:24]}"
            eng = self.base_engine.create_session_engine(collection_name=collection_name)
            self._sessions[session_id] = (eng, now)
            logger.info("Created isolated session engine for '%s' (collection '%s').", session_id, collection_name)
            return eng


class TextIngestRequest(BaseModel):
    markdown_text: str = Field(
        ..., min_length=1, max_length=10 * 1024 * 1024, description="Markdown or text content to ingest (max 10 MB)"
    )
    source_name: Optional[str] = Field("api_input.md", max_length=255, description="Identifier for this document")


class DirectSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(3, ge=1, le=50, description="Max results")
    mode: Literal["hybrid", "dense", "sparse"] = Field("hybrid", description="'hybrid', 'dense', or 'sparse'")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sparse weight for hybrid")


class CaseInvestigateRequest(BaseModel):
    user_query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    top_k: int = Field(5, ge=1, le=50, description="Fused chunks passed to the synthesizer")
    alpha: float = Field(0.3, ge=0.0, le=1.0, description="Sparse (BM25) weight in Weighted RRF")
    k: int = Field(60, ge=1, le=1000, description="RRF smoothing constant")
    candidate_k: Optional[int] = Field(None, ge=1, le=100, description="Candidates per retriever")
    chat_history: Optional[str] = Field(None, max_length=10000, description="Prior conversation turns")


def create_router(engine: RAGDetectiveEngine):
    """
    Creates and returns a FastAPI APIRouter configured with the provided RAGDetectiveEngine instance.
    """
    if APIRouter is None:
        raise ImportError(
            "FastAPI is required for the API layer. Install with `pip install fastapi uvicorn python-multipart`"
        )

    router = APIRouter(prefix="/api/v1", tags=["RAG Detective"])
    sessions = SessionEngineManager(engine)
    limiter = IPRateLimiter(window_seconds=engine.config.rate_limit_window_seconds)
    ingest_limit = engine.config.rate_limit_ingest_max
    query_limit = engine.config.rate_limit_query_max
    max_upload_bytes = engine.config.max_upload_size_bytes

    def _session_engine(req: Request) -> RAGDetectiveEngine:
        return sessions.get(_session_id_from_request(req))

    @router.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "RAG Detective Backend"}

    @router.get("/stats")
    async def get_stats(req: Request) -> Dict[str, Any]:
        """Returns indexed corpus statistics and active configuration for the current session."""
        return _session_engine(req).get_stats()

    @router.post("/investigate", response_model=InvestigationResult)
    async def investigate_lead(request: InvestigationRequest, req: Request) -> InvestigationResult:
        """
        Executes an end-to-end detective investigation:
        Query rewriting -> Dual Dense/Sparse search -> Weighted RRF -> Synthesized grounded answer.
        """
        limiter.check(_client_ip(req), "query", query_limit)
        try:
            return _session_engine(req).investigate(request)
        except Exception as e:
            logger.error("Investigation error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/search", response_model=List[SearchResult])
    async def search_chunks(request: DirectSearchRequest, req: Request) -> List[SearchResult]:
        """Direct search across indexed documents in the current session."""
        limiter.check(_client_ip(req), "query", query_limit)
        try:
            return _session_engine(req).search(
                query=request.query,
                top_k=request.top_k,
                mode=request.mode,
                alpha=request.alpha,
            )
        except Exception as e:
            logger.error("Search error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/ingest/text", response_model=IngestionResponse)
    async def ingest_raw_text(request: TextIngestRequest, req: Request) -> IngestionResponse:
        """Ingests raw markdown/plain text content directly into the session's pipeline."""
        limiter.check(_client_ip(req), "ingest", ingest_limit)
        try:
            return _session_engine(req).ingest_markdown(
                markdown_text=request.markdown_text,
                source_name=request.source_name or "api_input.md",
            )
        except Exception as e:
            logger.error("Text ingestion error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/ingest/file", response_model=IngestionResponse)
    async def ingest_document_file(
        req: Request,
        file: UploadFile = File(...),
    ) -> IngestionResponse:
        """Uploads and ingests a PDF, DOCX, Markdown, or text file (max 10 MB) into the current session."""
        limiter.check(_client_ip(req), "ingest", ingest_limit)
        content, raw_name, suffix = await _read_validated_upload(file, max_upload_bytes)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / f"upload{suffix}"
                tmp_path.write_bytes(content)
                response = _session_engine(req).ingest_file(tmp_path)
                response.source = raw_name
                return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error("File ingestion error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # ------------------------------------------------------------------
    # Frontend (RAG Detective web UI) endpoints.
    # Each streams newline-delimited JSON events, one per pipeline stage,
    # so the browser can visualise the Python pipeline as it runs.
    # ------------------------------------------------------------------
    def _ndjson(events) -> StreamingResponse:
        def gen():
            for ev in events:
                yield json.dumps(ev, ensure_ascii=False) + "\n"

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/cases")
    async def open_case(req: Request, file: UploadFile = File(...), reset: bool = Form(True)):
        """
        Opens a new case in the user's isolated session: validates file size (max 10 MB),
        clears the session's collection, creates a new empty vector store, and ingests the
        uploaded document through the full Python pipeline while streaming NDJSON.
        """
        limiter.check(_client_ip(req), "ingest", ingest_limit)
        data, raw_name, _ = await _read_validated_upload(file, max_upload_bytes)
        tracer = PipelineTracer(_session_engine(req))
        return _ndjson(tracer.ingest(data, raw_name, reset=True))

    @router.post("/cases/reset")
    async def reset_case(req: Request) -> Dict[str, str]:
        """Clears the current session's collection and initializes a new empty vector store."""
        session_eng = _session_engine(req)
        session_eng.reset()
        return {"status": "reset", "collection": session_eng.vector_store.collection_name}

    @router.post("/cases/investigate")
    async def investigate_case(request: CaseInvestigateRequest, req: Request):
        """
        Runs rewrite -> dense + sparse retrieval -> weighted RRF -> synthesis over the
        current session's indexed corpus and streams per-stage events (NDJSON).
        """
        limiter.check(_client_ip(req), "query", query_limit)
        tracer = PipelineTracer(_session_engine(req))
        return _ndjson(
            tracer.investigate(
                user_query=request.user_query,
                top_k=request.top_k,
                alpha=request.alpha,
                k=request.k,
                candidate_k=request.candidate_k,
                chat_history=request.chat_history,
            )
        )

    return router
