"""
Generation and Detective synthesis module for RAG Detective.
"""

from rag_detective.generation.prompts import build_detective_prompt
from rag_detective.generation.synthesizer import DetectiveSynthesizer

__all__ = ["build_detective_prompt", "DetectiveSynthesizer"]
