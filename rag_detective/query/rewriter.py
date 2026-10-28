"""
Query Rewriter powered by Google GenAI Gemini model.
Translates conversational user queries into dual-intent (dense and sparse) search representations.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from rag_detective.config import default_config
from rag_detective.exceptions import QueryRewritingError
from rag_detective.models import QueryRewriteResult, SubQuery
from rag_detective.query.prompts import QR_SYSTEM_PROMPT, build_rewrite_prompt

logger = logging.getLogger("uvicorn.error")


class QueryRewriter:
    """
    LLM-powered query analyzer and search intent generator.
    Translates colloquial questions into precision dense vectors and BM25 keywords.
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
            temperature if temperature is not None else default_config.rewriter_temperature
        )
        self.top_k = top_k or default_config.rewriter_top_k
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

    def rewrite(
        self,
        user_message: str,
        chat_history: Optional[str] = None,
        document_context: Optional[str] = None,
    ) -> QueryRewriteResult:
        """
        Rewrites a user message into dense and sparse sub-queries.

        Args:
            user_message: The raw input question from the user.
            chat_history: Optional conversation transcript.
            document_context: Optional brief context of indexed documents.

        Returns:
            QueryRewriteResult containing retrieval decision, clarification needs, and sub-queries.
        """
        if not user_message or not user_message.strip():
            return QueryRewriteResult(
                needs_retrieval=False,
                needs_clarification=False,
                sub_queries=[],
            )

        client = self.client
        if client is None:
            logger.warning("GenAI client unavailable; falling back to direct pass-through query.")

        prompt_content = build_rewrite_prompt(
            user_message=user_message,
            chat_history=chat_history or "",
            document_context=document_context or "",
        )

        try:
            from google.genai.types import GenerateContentConfig

            response = client.models.generate_content(
                model=self.model_name,
                config=GenerateContentConfig(
                    system_instruction=QR_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    top_k=self.top_k,
                ),
                contents=prompt_content,
            )

            raw_text = response.text or ""
            parsed_data = self._clean_and_parse_json(raw_text)
            return QueryRewriteResult.model_validate(parsed_data)

        except Exception as e:
            logger.error("Query rewrite LLM call failed: %s", e)
            raise QueryRewritingError("Query rewrite LLM cal failed: %s",e) from e

    def _clean_and_parse_json(self, text: str) -> dict:
        """Strips markdown fences and parses raw JSON."""
        cleaned = text.strip()
        # Remove markdown ```json ... ``` wrapper if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Find first '{' and last '}'
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        return json.loads(cleaned)
