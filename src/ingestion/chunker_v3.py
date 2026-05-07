"""
Semantic chunker using LangChain's SemanticChunker with OpenAI embeddings.

Uses semantic similarity to find optimal chunk boundaries instead of fixed tokens.
"""

import hashlib
from langchain_text_splitters import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from config import ChunkingConfig, get_config
from src.ingestion.parser import ParsedElement
from src.ingestion.chunker_v2 import (
    _build_context_prefix,
    _count_tokens,
    _enforce_embedding_ceiling,
    Chunk,
)


def _generate_chunk_id(doc_name: str, content: str, index: int) -> str:
    """Generate unique chunk ID."""
    raw = f"{doc_name}::{index}::{content[:50]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _create_chunk(
    content: str,
    doc_name: str,
    doc_path: str,
    section_path: list[str],
    element_id: str,
    language: str,
    index: int,
) -> Chunk:
    """Helper to create a Chunk object."""
    prefix = _build_context_prefix(doc_name, section_path)
    full_content = prefix + "\n\n" + content
    
    return Chunk(
        chunk_id=_generate_chunk_id(doc_name, content, index),
        doc_name=doc_name,
        doc_path=doc_path,
        element_type="text",
        content=full_content,
        section_path=section_path,
        source_element_ids=[element_id],
        language=language,
        token_estimate=_count_tokens(full_content),
    )


def chunk_elements_semantic(
    elements: list[ParsedElement],
    config: ChunkingConfig,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
) -> list[Chunk]:
    """
    Split documents into semantic chunks using configured embeddings.

    Args:
        elements: List of ParsedElement from parser
        config: ChunkingConfig
        breakpoint_threshold_type: "percentile", "standard_deviation", or "interquartile"
        breakpoint_threshold_amount: Threshold value (e.g., 95 for 95th percentile)

    Returns:
        List of Chunk objects
    """
    # Get embedding model from global config
    app_config = get_config()
    embedding_model = app_config.embedding.model
    
    # Initialize semantic chunker with configured embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    chunks = []
    chunk_index = 0

    for elem in elements:
        if not elem.content.strip():
            continue

        # Skip headings
        if elem.element_type == "heading":
            continue

        # Tables: emit as-is
        if elem.element_type == "table":
            prefix = _build_context_prefix(elem.doc_name, elem.section_path)
            chunk = Chunk(
                chunk_id=_generate_chunk_id(elem.doc_name, elem.content, chunk_index),
                doc_name=elem.doc_name,
                doc_path=elem.doc_path,
                element_type="table",
                content=prefix + "\n\n" + elem.content,
                section_path=list(elem.section_path),
                source_element_ids=[elem.element_id],
                language=elem.language,
                token_estimate=_count_tokens(elem.content),
                table_data=elem.table_data,
            )
            chunk = _enforce_embedding_ceiling(chunk)
            chunks.append(chunk)
            chunk_index += 1
            continue

        # Text: split semantically
        try:
            semantic_docs = splitter.create_documents([elem.content])
            for semantic_doc in semantic_docs:
                chunk = _create_chunk(
                    content=semantic_doc.page_content,
                    doc_name=elem.doc_name,
                    doc_path=elem.doc_path,
                    section_path=list(elem.section_path),
                    element_id=elem.element_id,
                    language=elem.language,
                    index=chunk_index,
                )
                chunk = _enforce_embedding_ceiling(chunk)
                chunks.append(chunk)
                chunk_index += 1
        except Exception as e:
            print(f"⚠ Semantic chunking failed for {elem.doc_name}: {e}. Skipping.")
            continue

    return chunks
