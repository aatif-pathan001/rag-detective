"""
Detective Synthesizer: Grounded LLM reasoning agent for RAG Detective.
Synthesizes answers strictly derived from retrieved context evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from rag_detective.config import default_config
from rag_detective.exceptions import SynthesisError
from rag_detective.generation.prompts import build_detective_prompt
from rag_detective.models import SearchResult

logger = logging.getLogger("uvicorn.error")


class DetectiveSynthesizer:
    """
    Synthesizes factual answers strictly derived from retrieved evidence documents.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or default_config.gemini_api_key
        self.model_name = model_name or default_config.gemini_model_name
        self.temperature = (
            temperature if temperature is not None else default_config.synthesizer_temperature
        )
        self.top_k = top_k or default_config.synthesizer_top_k
        self.last_mode: str = self.model_name
        self._client = None

    @property
    def client(self):
        """Lazy-loaded Google GenAI client."""
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_client(self):
        try:
            from google import genai
            return genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.warning("Could not initialize google.genai Client: %s", e)
            return None

    def synthesize(
        self,
        user_query: str,
        retrieved_docs: Sequence[Any],
        cite_sources: bool = False,
    ) -> str:
        """
        Synthesizes an answer based strictly on retrieved documents.

        Args:
            user_query: The question to answer.
            retrieved_docs: List of SearchResult or DocumentChunk objects.
            cite_sources: If True, asks the model to cite Document IDs inline, e.g. [Chunk_3].

        Returns:
            Grounded textual answer string.
        """
        if not retrieved_docs:
            return "I do not know. No relevant evidence was found in the indexed documents."

        client = self.client
        if client is None:
            logger.warning("Google GenAI client unavailable.")

        input_prompt = build_detective_prompt(user_query, retrieved_docs, cite_sources=cite_sources)
        logger.debug("Prompting Detective Synthesizer with %d documents...", len(retrieved_docs))

        try:
            from google.genai.types import GenerateContentConfig

            response = client.models.generate_content(
                model=self.model_name,
                config=GenerateContentConfig(
                    temperature=self.temperature,
                    top_k=self.top_k,
                ),
                contents=input_prompt,
            )

            answer = response.text or ""
            self.last_mode = self.model_name
            return answer.strip()

        except Exception as e:
            logger.warning("Gemini synthesis failed (%s).", e)
            raise SynthesisError("LLM synthesis failed :", e ) from e
