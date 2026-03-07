"""
Section-aware semantic chunker.
Chunks at paragraph/clause boundaries, preserves tables as atomic units,
and attaches metadata (doc name, section path, element type, language).
"""

import hashlib
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.ingestion.parser import ParsedElement


@dataclass
class Chunk:
    """An atomic unit ready for embedding and indexing."""
    chunk_id: str
    doc_name: str
    doc_path: str
    element_type: str           # "text", "table", "table_summary"
    content: str                # The text to embed
    section_path: list[str]
    source_element_ids: list[str]  # Traceability: which ParsedElements this came from
    metadata: dict = field(default_factory=dict)
    language: str = "unknown"
    token_estimate: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English, ~2 for Arabic."""
    '''
    Currently simple heurtic ma bein statistical estimates of bpe (byte pair encoding)
    #English bpe tends to average 4 chars per token, while Arabic can be closer to 2.
    #3 is middle ground beinhom later use same tokenzier as embedding model for more accurate estimate and chunking

    
    '''
 
    if not text:
        return 0
    # Simple heuristic
    return max(len(text) // 3, len(text.split()))


# Hard ceiling: chunks must never exceed the embedding model context window.
# nomic-embed-text = 8192 tokens.  We use 7500 as a safety margin.
_EMBEDDING_TOKEN_LIMIT = 7500

# Tables are allowed up to 4× the paragraph limit before being split.
_TABLE_TOKEN_CEILING_FACTOR = 4


def _split_text_at_sentences(text: str, max_tokens: int) -> list[str]:
    """Split a long text into pieces that each fit within *max_tokens*.

    Strategy:
      1. Split on sentence boundaries (period / exclamation / question followed by whitespace).
      2. Greedily accumulate sentences until the next one would exceed the budget.
      3. If a single sentence still exceeds max_tokens, fall back to a hard character cut.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        sent_tokens = _estimate_tokens(sent)
        if buf and buf_tokens + sent_tokens > max_tokens:
            pieces.append(" ".join(buf))
            buf, buf_tokens = [], 0
        # Single sentence bigger than the limit → hard-cut
        if sent_tokens > max_tokens:
            char_limit = max_tokens * 3  # inverse of the //3 heuristic
            for start in range(0, len(sent), char_limit):
                pieces.append(sent[start:start + char_limit])
        else:
            buf.append(sent)
            buf_tokens += sent_tokens

    if buf:
        pieces.append(" ".join(buf))
    return pieces or [text]  # never return empty


def _split_table_rows(table_md: str, max_tokens: int) -> list[str]:
    """Split a markdown table into sub-tables that each fit within *max_tokens*.

    Each sub-table keeps the header row so the LLM can interpret columns.
    """
    lines = table_md.strip().splitlines()
    if len(lines) < 3:
        return [table_md]  # too small to split meaningfully

    header_line = lines[0]
    separator_line = lines[1]
    header_block = f"{header_line}\n{separator_line}"
    header_tokens = _estimate_tokens(header_block)
    row_budget = max_tokens - header_tokens
    if row_budget < 10:
        row_budget = max_tokens  # degenerate case — just try

    data_rows = lines[2:]
    parts: list[str] = []
    current_rows: list[str] = []
    current_tokens = 0

    for row in data_rows:
        row_tokens = _estimate_tokens(row)
        if current_rows and current_tokens + row_tokens > row_budget:
            parts.append(f"{header_block}\n" + "\n".join(current_rows))
            current_rows, current_tokens = [], 0
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        parts.append(f"{header_block}\n" + "\n".join(current_rows))

    return parts or [table_md]


def _generate_chunk_id(doc_name: str, chunk_type: str, content: str, index: int) -> str:
    raw = f"{doc_name}::{chunk_type}::{index}::{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_elements(
    elements: list[ParsedElement],
    max_chunk_tokens: int = 512,
    overlap_tokens: int = 50,
    preserve_tables: bool = True,
) -> list[Chunk]:
    """
    Convert parsed elements into chunks for embedding.
    
    Strategy:
    - Headings mesh standalone chunks (they provide section context).
    - Paragraphs are grouped by section leghayet max_chunk_tokens is reached.
    - Tables are kept as single chunks when possible; oversized tables are split by rows.
    - Each chunk carries full section path for citation.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    # Group elements by document + section
    current_doc = None
    current_section = None
    paragraph_buffer: list[ParsedElement] = []

    def _flush_paragraphs():
        nonlocal chunk_index
        if not paragraph_buffer:
            return

        # Merge paragraphs into chunks respecting token limit
        current_texts: list[str] = []
        current_ids: list[str] = []
        current_tokens = 0

        for elem in paragraph_buffer:
            elem_tokens = _estimate_tokens(elem.content)

            # ── FIX: split oversized single paragraphs ──
            if elem_tokens > max_chunk_tokens:
                warnings.warn(
                    f"Oversized paragraph ({elem_tokens} est. tokens) in "
                    f"{elem.doc_name} / {' > '.join(elem.section_path or ['root'])}. "
                    f"Splitting at sentence boundaries."
                )
                sub_texts = _split_text_at_sentences(elem.content, max_chunk_tokens)
                for sub in sub_texts:
                    # Treat each piece as its own pseudo-element
                    if current_tokens + _estimate_tokens(sub) > max_chunk_tokens and current_texts:
                        # flush current buffer first
                        merged = "\n\n".join(current_texts)
                        section_ctx = paragraph_buffer[0].section_path
                        section_prefix = " > ".join(section_ctx) if section_ctx else ""
                        if section_prefix:
                            content_with_ctx = f"[{paragraph_buffer[0].doc_name} | {section_prefix}]\n\n{merged}"
                        else:
                            content_with_ctx = f"[{paragraph_buffer[0].doc_name}]\n\n{merged}"
                        chunks.append(Chunk(
                            chunk_id=_generate_chunk_id(paragraph_buffer[0].doc_name, "text", merged, chunk_index),
                            doc_name=paragraph_buffer[0].doc_name,
                            doc_path=paragraph_buffer[0].doc_path,
                            element_type="text",
                            content=content_with_ctx,
                            section_path=list(section_ctx),
                            source_element_ids=list(current_ids),
                            language=paragraph_buffer[0].language,
                            token_estimate=_estimate_tokens(content_with_ctx),
                            metadata={"section_path_str": " > ".join(section_ctx) if section_ctx else "root"},
                        ))
                        chunk_index += 1
                        current_texts, current_ids, current_tokens = [], [], 0
                    current_texts.append(sub)
                    current_ids.append(elem.element_id)
                    current_tokens += _estimate_tokens(sub)
                continue  # already handled — skip normal append below

            if current_tokens + elem_tokens > max_chunk_tokens and current_texts:
                # Emit current chunk
                merged = "\n\n".join(current_texts)
                section_ctx = paragraph_buffer[0].section_path
                section_prefix = " > ".join(section_ctx) if section_ctx else ""
                if section_prefix:
                    content_with_ctx = f"[{paragraph_buffer[0].doc_name} | {section_prefix}]\n\n{merged}"
                else:
                    content_with_ctx = f"[{paragraph_buffer[0].doc_name}]\n\n{merged}"

                chunk = Chunk(
                    chunk_id=_generate_chunk_id(paragraph_buffer[0].doc_name, "text", merged, chunk_index),
                    doc_name=paragraph_buffer[0].doc_name,
                    doc_path=paragraph_buffer[0].doc_path,
                    element_type="text",
                    content=content_with_ctx,
                    section_path=list(section_ctx),
                    source_element_ids=list(current_ids),
                    language=paragraph_buffer[0].language,
                    token_estimate=_estimate_tokens(content_with_ctx),
                    metadata={
                        "section_path_str": " > ".join(section_ctx) if section_ctx else "root",
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

                # Overlap: keep last element
                if overlap_tokens > 0 and current_texts:
                    last_text = current_texts[-1]
                    last_id = current_ids[-1]
                    current_texts = [last_text]
                    current_ids = [last_id]
                    current_tokens = _estimate_tokens(last_text)
                else:
                    current_texts = []
                    current_ids = []
                    current_tokens = 0

            current_texts.append(elem.content)
            current_ids.append(elem.element_id)
            current_tokens += elem_tokens

        # Flush remainder
        if current_texts:
            merged = "\n\n".join(current_texts)
            section_ctx = paragraph_buffer[0].section_path
            section_prefix = " > ".join(section_ctx) if section_ctx else ""
            if section_prefix:
                content_with_ctx = f"[{paragraph_buffer[0].doc_name} | {section_prefix}]\n\n{merged}"
            else:
                content_with_ctx = f"[{paragraph_buffer[0].doc_name}]\n\n{merged}"

            chunk = Chunk(
                chunk_id=_generate_chunk_id(paragraph_buffer[0].doc_name, "text", merged, chunk_index),
                doc_name=paragraph_buffer[0].doc_name,
                doc_path=paragraph_buffer[0].doc_path,
                element_type="text",
                content=content_with_ctx,
                section_path=list(section_ctx),
                source_element_ids=list(current_ids),
                language=paragraph_buffer[0].language,
                token_estimate=_estimate_tokens(content_with_ctx),
                metadata={
                    "section_path_str": " > ".join(section_ctx) if section_ctx else "root",
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        paragraph_buffer.clear()

    for elem in elements:
        # Flush when document or section changes
        if elem.doc_name != current_doc or elem.section_path != current_section:
            _flush_paragraphs()
            current_doc = elem.doc_name
            current_section = elem.section_path

        if elem.element_type == "heading":
            # Headings are not standalone chunks — they set section context
            continue

        elif elem.element_type == "table" and preserve_tables:
            # Flush any pending paragraphs first
            _flush_paragraphs()

            section_prefix = " > ".join(elem.section_path) if elem.section_path else ""
            table_token_ceiling = max_chunk_tokens * _TABLE_TOKEN_CEILING_FACTOR
            table_tokens = _estimate_tokens(elem.content)

            # ── FIX: split oversized tables by rows ──
            if table_tokens > table_token_ceiling:
                warnings.warn(
                    f"Oversized table ({table_tokens} est. tokens) in "
                    f"{elem.doc_name} / {section_prefix or 'root'}. "
                    f"Splitting into sub-tables."
                )
                parts = _split_table_rows(elem.content, table_token_ceiling)
                for part_idx, part_md in enumerate(parts):
                    suffix = f" (part {part_idx+1}/{len(parts)})" if len(parts) > 1 else ""
                    if section_prefix:
                        content_with_ctx = f"[{elem.doc_name} | {section_prefix} | TABLE{suffix}]\n\n{part_md}"
                    else:
                        content_with_ctx = f"[{elem.doc_name} | TABLE{suffix}]\n\n{part_md}"
                    chunk = Chunk(
                        chunk_id=_generate_chunk_id(elem.doc_name, "table", part_md, chunk_index),
                        doc_name=elem.doc_name,
                        doc_path=elem.doc_path,
                        element_type="table",
                        content=content_with_ctx,
                        section_path=list(elem.section_path),
                        source_element_ids=[elem.element_id],
                        language=elem.language,
                        token_estimate=_estimate_tokens(content_with_ctx),
                        metadata={
                            "section_path_str": section_prefix or "root",
                            "table_data": elem.table_data,
                            "table_part": f"{part_idx+1}/{len(parts)}",
                        },
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                # Table fits — keep it as a single chunk
                if section_prefix:
                    content_with_ctx = f"[{elem.doc_name} | {section_prefix} | TABLE]\n\n{elem.content}"
                else:
                    content_with_ctx = f"[{elem.doc_name} | TABLE]\n\n{elem.content}"

                chunk = Chunk(
                    chunk_id=_generate_chunk_id(elem.doc_name, "table", elem.content, chunk_index),
                    doc_name=elem.doc_name,
                    doc_path=elem.doc_path,
                    element_type="table",
                    content=content_with_ctx,
                    section_path=list(elem.section_path),
                    source_element_ids=[elem.element_id],
                    language=elem.language,
                    token_estimate=_estimate_tokens(content_with_ctx),
                    metadata={
                        "section_path_str": section_prefix or "root",
                        "table_data": elem.table_data,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1

        else:
            # Regular paragraph — buffer it
            paragraph_buffer.append(elem)

    # Flush remaining
    _flush_paragraphs()

    return chunks
