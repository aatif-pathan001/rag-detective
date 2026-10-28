"""
Entrypoint for Hugging Face Spaces (Gradio SDK mode) and local execution.
Starts the RAG Detective FastAPI server (which serves both the web UI and REST API) on port 7860.
"""

import os
import uvicorn
from rag_detective.api.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    uvicorn.run(app, host="0.0.0.0", port=port)
