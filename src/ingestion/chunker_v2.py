"""
Section-aware semantic chunker.

Input:  list[ParsedElement] from src/ingestion/parser.py
Output: list[Chunk] ready for embedding and KG ingestion

Strategy:
  - Headings flush the text buffer and reset section context.
  - Paragraphs are sentence-split, accumulated until the token budget is reached,
    then flushed with a true sliding-window overlap tail.
  - Tables are emitted atomically; oversized tables are split row-by-row.
  - The context prefix "[doc | section]" is counted toward the token budget so
    emitted chunks never exceed max_chunk_tokens.
"""

import hashlib
import re
import warnings
from functools import lru_cache
from dataclasses import dataclass, field, asdict

from config import ChunkingConfig, get_config
from src.ingestion.parser import ParsedElement


# Hard ceiling for the embedding model (nomic-embed-text = 8192 tokens).
_EMBEDDING_TOKEN_LIMIT = 7500
# Tables may be up to this many times larger than a paragraph before being split.
_TABLE_TOKEN_CEILING_FACTOR = 4


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """An atomic unit ready for embedding and KG indexing."""
    chunk_id: str
    doc_name: str
    doc_path: str
    element_type: str                   # "text" | "table"
    content: str                        # includes context prefix
    section_path: list[str]
    source_element_ids: list[str]
    language: str = "unknown"
    token_estimate: int = 0
    table_data: dict | None = None      # structured table: {headers, rows, shape}
    access_level: int | None = None     # stamped by build_index.py after chunking
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_tokenizer():
    """Load and cache the same tokenizer configuration used by the embedder."""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Exact tokenization requires transformers. "
            "Install dependencies from requirements.txt."
        ) from e

    cfg = get_config()
    tokenizer_model = cfg.embedding.tokenizer_model or cfg.embedding.model
    return AutoTokenizer.from_pretrained(tokenizer_model)


def _count_tokens(text: str) -> int:
    """Count tokens using the embedding tokenizer for exact budgeting."""
    if not text:
        return 0
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _encode_tokens(text: str) -> list[int]:
    """Encode text with the embedding tokenizer and return raw token ids."""
    if not text:
        return []
    tokenizer = _get_tokenizer()
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _build_context_prefix(
    doc_name: str,
    section_path: list[str],
    label: str | None = None,
) -> str:
    parts = [doc_name]
    if section_path:
        parts.append(" > ".join(section_path))
    if label:
        parts.append(label)
    return "[" + " | ".join(parts) + "]"


def _generate_chunk_id(doc_name: str, kind: str, content: str, index: int) -> str:
    raw = f"{doc_name}::{kind}::{index}::{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _enforce_embedding_ceiling(chunk: Chunk) -> Chunk:
    """Truncate any chunk that slips past the embedding model's context window."""
    if chunk.token_estimate <= _EMBEDDING_TOKEN_LIMIT:
        return chunk
    warnings.warn(
        f"Chunk {chunk.chunk_id} ({chunk.doc_name}) exceeds embedding ceiling "
        f"({chunk.token_estimate} > {_EMBEDDING_TOKEN_LIMIT}). Truncating. "
        f"Investigate splitter configuration."
    )
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(chunk.content, add_special_tokens=False)
    truncated_ids = token_ids[:_EMBEDDING_TOKEN_LIMIT]
    chunk.content = tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()
    chunk.token_estimate = _count_tokens(chunk.content)
    return chunk


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def _split_paragraph_to_sentences(
    text: str,
    effective_budget: int,
) -> list[tuple[str, int]]:
    """
    Split text into (sentence, token_count) pairs each ≤ effective_budget.

    Sentences that still exceed the budget after splitting on punctuation are
    split directly with the bge-m3 tokenizer so every emitted piece is bounded
    by the exact token budget.
    """
    raw = re.split(r'(?<=[.!?؟])\s+', text.strip())
    result: list[tuple[str, int]] = []
    tokenizer = _get_tokenizer()

    for sent in raw:
        sent = sent.strip()
        if not sent:
            continue
        tokens = _count_tokens(sent)
        if tokens <= effective_budget:
            result.append((sent, tokens))
            continue

        token_ids = _encode_tokens(sent)
        for start in range(0, len(token_ids), effective_budget):
            piece_ids = token_ids[start:start + effective_budget]
            piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
            if piece:
                result.append((piece, len(piece_ids)))

    return result or [(text.strip(), _count_tokens(text.strip()))]


def _split_table_rows(table_md: str, effective_budget: int) -> list[str]:
    """
    Split a markdown table into sub-tables each ≤ effective_budget tokens.
    Every sub-table repeats the header + separator rows.
    """
    lines = table_md.strip().splitlines()
    if len(lines) < 3:
        return [table_md]

    header_block = f"{lines[0]}\n{lines[1]}"
    header_tokens = _count_tokens(header_block)
    row_budget = effective_budget - header_tokens

    if row_budget < 10:
        warnings.warn(
            f"Table header alone ({header_tokens} tokens) nearly fills the budget "
            f"({effective_budget} tokens). Splitting one row per chunk."
        )
        row_budget = max(1, effective_budget)

    parts: list[str] = []
    current_rows: list[str] = []
    current_tokens = 0

    for row in lines[2:]:
        row_tokens = _count_tokens(row)
        if current_rows and current_tokens + row_tokens > row_budget:
            parts.append(header_block + "\n" + "\n".join(current_rows))
            current_rows, current_tokens = [], 0
        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        parts.append(header_block + "\n" + "\n".join(current_rows))

    return parts or [table_md]


# ---------------------------------------------------------------------------
# Table chunk emitter
# ---------------------------------------------------------------------------

def _emit_table_chunk(
    elem: ParsedElement,
    part_md: str,
    part_idx: int,
    n_parts: int,
    chunk_index: int,
) -> Chunk:
    suffix = f" (part {part_idx + 1}/{n_parts})" if n_parts > 1 else ""
    prefix = _build_context_prefix(elem.doc_name, elem.section_path, f"TABLE{suffix}")
    content = f"{prefix}\n\n{part_md}"
    chunk = Chunk(
        chunk_id=_generate_chunk_id(elem.doc_name, "table", part_md, chunk_index),
        doc_name=elem.doc_name,
        doc_path=elem.doc_path,
        element_type="table",
        content=content,
        section_path=list(elem.section_path),
        source_element_ids=[elem.element_id],
        language=elem.language,
        token_estimate=_count_tokens(content),
        table_data=elem.table_data,
    )
    return _enforce_embedding_ceiling(chunk)


# ---------------------------------------------------------------------------
# Text chunk builder — sentence-level buffer with sliding-window overlap
# ---------------------------------------------------------------------------

class _ChunkBuilder:
    """
    Accumulates sentences for a single (doc, section) window.

    flush() emits a Chunk and retains a sliding-window tail (≤ overlap_tokens)
    in the buffer so the next chunk begins with some overlapping context.
    """

    def __init__(
        self,
        doc_name: str,
        doc_path: str,
        section_path: list[str],
        language: str,
        config: ChunkingConfig,
    ) -> None:
        self.doc_name = doc_name
        self.doc_path = doc_path
        self.section_path = list(section_path)
        self.language = language
        self._config = config

        self._prefix = _build_context_prefix(doc_name, section_path)
        prefix_tokens = _count_tokens(self._prefix)
        # Sentence budget is the token allowance *after* the prefix is counted.
        self.effective_budget = max(config.max_chunk_tokens - prefix_tokens, 50)

        # Each entry: (sentence_text, source_element_id, token_count)
        self._buffer: list[tuple[str, str, int]] = []
        self._buffer_tokens: int = 0

    def is_empty(self) -> bool:
        return not self._buffer

    def can_fit(self, sentence_tokens: int) -> bool:
        return self._buffer_tokens + sentence_tokens <= self.effective_budget

    def add_sentence(self, text: str, source_id: str, tokens: int) -> None:
        self._buffer.append((text, source_id, tokens))
        self._buffer_tokens += tokens

    def flush(self, chunk_index: int) -> tuple["Chunk | None", int]:
        if not self._buffer:
            return None, chunk_index

        texts = [s[0] for s in self._buffer]
        seen: set[str] = set()
        source_ids: list[str] = []
        for _, sid, _ in self._buffer:
            if sid not in seen:
                seen.add(sid)
                source_ids.append(sid)

        content = self._prefix + "\n\n" + " ".join(texts)
        chunk = Chunk(
            chunk_id=_generate_chunk_id(self.doc_name, "text", content, chunk_index),
            doc_name=self.doc_name,
            doc_path=self.doc_path,
            element_type="text",
            content=content,
            section_path=self.section_path,
            source_element_ids=source_ids,
            language=self.language,
            token_estimate=_count_tokens(content),
        )
        chunk = _enforce_embedding_ceiling(chunk)

        # Sliding-window overlap: retain the longest tail ≤ overlap_tokens.
        if self._config.overlap_tokens > 0:
            carry: list[tuple[str, str, int]] = []
            carry_tokens = 0
            for entry in reversed(self._buffer):
                if carry_tokens + entry[2] <= self._config.overlap_tokens:
                    carry.insert(0, entry)
                    carry_tokens += entry[2]
                else:
                    break
            self._buffer = carry
            self._buffer_tokens = carry_tokens
        else:
            self._buffer = []
            self._buffer_tokens = 0

        return chunk, chunk_index + 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_elements(elements: list[ParsedElement], config: ChunkingConfig) -> list[Chunk]:
    """
    Convert parsed document elements into embeddable chunks.

    Headings flush the current text buffer and update section context without
    producing a chunk of their own. Paragraphs are sentence-split and
    accumulated within the effective token budget. Tables are emitted
    atomically and split by rows only when oversized.
    """
    chunks: list[Chunk] = []
    chunk_index = 0
    current_doc: str | None = None
    current_section: list[str] | None = None
    builder: _ChunkBuilder | None = None

    for elem in elements:
        if not elem.content.strip():
            continue

        # Flush and reset on document or section boundary
        if elem.doc_name != current_doc or elem.section_path != current_section:
            if builder is not None:
                chunk, chunk_index = builder.flush(chunk_index)
                if chunk is not None:
                    chunks.append(chunk)
            current_doc = elem.doc_name
            current_section = elem.section_path
            builder = _ChunkBuilder(
                doc_name=elem.doc_name,
                doc_path=elem.doc_path,
                section_path=elem.section_path,
                language=elem.language,
                config=config,
            )

        if elem.element_type == "heading":
            continue

        if elem.element_type == "table":
            # Flush any pending text before the table
            chunk, chunk_index = builder.flush(chunk_index)  # type: ignore[union-attr]
            if chunk is not None:
                chunks.append(chunk)

            prefix = _build_context_prefix(elem.doc_name, elem.section_path)
            table_ceiling = max(
                config.max_chunk_tokens * _TABLE_TOKEN_CEILING_FACTOR - _count_tokens(prefix),
                50,
            )

            if _count_tokens(elem.content) > table_ceiling:
                warnings.warn(
                    f"Oversized table in {elem.doc_name} / "
                    f"{' > '.join(elem.section_path) or 'root'}. Splitting by rows."
                )
                parts = _split_table_rows(elem.content, table_ceiling)
            else:
                parts = [elem.content]

            for part_idx, part_md in enumerate(parts):
                chunks.append(_emit_table_chunk(elem, part_md, part_idx, len(parts), chunk_index))
                chunk_index += 1

        else:
            # Paragraph, list_item, or any other text element
            sentences = _split_paragraph_to_sentences(elem.content, builder.effective_budget)  # type: ignore[union-attr]
            for sent_text, sent_tokens in sentences:
                if not builder.can_fit(sent_tokens) and not builder.is_empty():  # type: ignore[union-attr]
                    chunk, chunk_index = builder.flush(chunk_index)  # type: ignore[union-attr]
                    if chunk is not None:
                        chunks.append(chunk)
                builder.add_sentence(sent_text, elem.element_id, sent_tokens)  # type: ignore[union-attr]

    # Final flush
    if builder is not None:
        chunk, chunk_index = builder.flush(chunk_index)
        if chunk is not None:
            chunks.append(chunk)

    return chunks
