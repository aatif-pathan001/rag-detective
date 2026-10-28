"""
Prompt construction for Detective Synthesizer.
Strict context-bounded reasoning from rag-det.ipynb prototype.
"""

from __future__ import annotations

from typing import Any, Sequence

from rag_detective.models import DocumentChunk, SearchResult


def build_detective_prompt(
    user_query: str,
    retrieved_docs: Sequence[Any],
    cite_sources: bool = False,
) -> str:
    """
    Constructs the grounded LLM synthesis prompt matching LLM_Prompt() in rag-det.ipynb.

    Args:
        user_query: The user's original query.
        retrieved_docs: Ranked context documents (SearchResult, DocumentChunk, or LangChain Document).
        cite_sources: If True, instructs the model to cite supporting Document IDs in square
                      brackets after each claim (e.g. [Chunk_3]).

    Returns:
        Structured prompt with bounded <context> and <user_query> XML tags.
    """
    context_blocks = []
    for doc in retrieved_docs:
        if isinstance(doc, SearchResult):
            doc_id = doc.chunk_id
            content = doc.content
        elif isinstance(doc, DocumentChunk):
            doc_id = doc.id
            content = doc.page_content
        else:
            doc_id = getattr(doc, "id", None) or getattr(doc, "metadata", {}).get("id", "Unknown")
            content = getattr(doc, "page_content", "")

        context_blocks.append(f"Document ID: {doc_id}\nContent: {content}")

    context_str = "\n\n".join(context_blocks)

    citation_rule = (
        "After each claim, cite the supporting Document ID in square brackets, "
        "e.g. [Chunk_3]. Use plain prose without markdown.\n"
        if cite_sources
        else ""
    )

    prompt = (
        "Answer the user query strictly based on the context provided. "
        'If the context does not contain the answer, respond with "I do not know."\n'
        f"{citation_rule}\n"
        f"<user_query>\n{user_query}\n</user_query>\n\n"
        f"<context>\n{context_str}\n</context>"
    )
    return prompt
