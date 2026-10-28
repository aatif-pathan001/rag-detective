"""
FastAPI application factory for RAG Detective backend.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from rag_detective.config import RAGDetectiveConfig, default_config
from rag_detective.engine import RAGDetectiveEngine

logger = logging.getLogger("uvicorn.error")


def create_app(
    config: Optional[RAGDetectiveConfig] = None,
    engine: Optional[RAGDetectiveEngine] = None,
):
    """
    Constructs and configures the FastAPI application for RAG Detective.
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise ImportError(
            "FastAPI is required to run the REST backend. Please install via: `pip install fastapi uvicorn python-multipart`"
        ) from e

    app = FastAPI(
        title="RAG Detective Backend API",
        description="Modular Hybrid RAG Engine with Query Rewriting, BM25, Chroma Vector Search, and Weighted RRF",
        version="1.0.0",
    )

    # Enable CORS for frontend clients (Streamlit, React, Vue, Next.js, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rag_engine = engine or RAGDetectiveEngine(config=config or default_config)

    from rag_detective.api.routes import create_router
    app.include_router(create_router(rag_engine))

    api_info = {
        "name": "RAG Detective API",
        "docs": "/docs",
        "version": "1.0.0",
        "status": "online",
    }

    # Serve the RAG Detective web frontend (../frontend) at /ui when it is present.
    frontend_dir = Path(os.getenv("RAG_DETECTIVE_FRONTEND_DIR", Path(__file__).resolve().parents[2] / "frontend"))
    frontend_page = os.getenv("RAG_DETECTIVE_FRONTEND_PAGE", "RAG Detective v4.dc.html")
    if (frontend_dir / frontend_page).is_file():
        from fastapi.responses import RedirectResponse
        from fastapi.staticfiles import StaticFiles

        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")
        ui_url = "/ui/" + quote(frontend_page)
        api_info["ui"] = ui_url

        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url=ui_url)
    else:
        @app.get("/")
        async def root():
            return api_info

    @app.get("/api")
    async def api_root():
        return api_info

    return app


# Module-level instance for uvicorn (e.g., uvicorn rag_detective.api.app:app)
def get_app():
    return create_app()


try:
    app = get_app()
except Exception:
    # If fastapi is not installed in the current environment yet, app will be None
    app = None
