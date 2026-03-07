"""
Document parser using Docling for structured extraction of Word documents.
Extracts sections, paragraphs, tables, and figures with full metadata.
"""

import json
import hashlib
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from docling.document_converter import DocumentConverter

# Module-level tracker for parse failures —
# lets build_index.py report skipped documents.
PARSE_FAILURES: list[tuple[str, str]] = []


@dataclass
class ParsedElement:
    """An atomic unit extracted from a document."""
    element_id: str
    doc_name: str
    doc_path: str
    element_type: str  # "paragraph", "table", "heading", "list_item", "figure"
    content: str
    section_path: list[str] = field(default_factory=list)  # ["Chapter 1", "Section 1.2", "1.2.1"]
    metadata: dict = field(default_factory=dict)
    language: str = "unknown"
    page_number: Optional[int] = None
    table_data: Optional[dict] = None  # Structured table if element_type == "table"

    def to_dict(self) -> dict:
        return asdict(self)


def _generate_element_id(doc_name: str, element_type: str, content: str, index: int) -> str:
    """Generate a deterministic unique ID for an element."""
    raw = f"{doc_name}::{element_type}::{index}::{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _detect_language(text: str) -> str:
    """Simple heuristic language detection (Arabic vs English)."""
    if not text.strip():
        return "unknown"
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    ratio = arabic_chars / max(len(text), 1)
    if ratio > 0.3:
        return "ar"
    return "en"


def parse_document(doc_path: str | Path) -> list[ParsedElement]:
    """
    Parse a single Word document into structured elements.
    
    Returns a list of ParsedElement objects with full metadata.
    """
    doc_path = Path(doc_path)
    doc_name = doc_path.stem

    # Initialize Docling converter
    converter = DocumentConverter()

    # Convert document
    result = converter.convert(str(doc_path))
    doc = result.document

    elements: list[ParsedElement] = []
    current_section_path: list[str] = []
    element_index = 0

    # Iterate over Docling's document items
    for item, _level in doc.iterate_items():
        item_type = type(item).__name__

        if item_type == "SectionHeaderItem" or (hasattr(item, 'label') and 'heading' in str(getattr(item, 'label', '')).lower()):
            # Update section path based on heading level
            heading_text = item.text if hasattr(item, 'text') else str(item)
            heading_text = heading_text.strip()
            if heading_text:
                level = getattr(item, 'level', _level) or _level
                # Truncate section path to current level and add new heading
                if isinstance(level, int) and level > 0:
                    current_section_path = current_section_path[:level - 1]
                current_section_path.append(heading_text)

                #Adding heaidings fel document

                elem = ParsedElement(
                    element_id=_generate_element_id(doc_name, "heading", heading_text, element_index),
                    doc_name=doc_name,
                    doc_path=str(doc_path),
                    element_type="heading",
                    content=heading_text,
                    section_path=list(current_section_path),
                    language=_detect_language(heading_text),
                )
                elements.append(elem)
                element_index += 1

        elif hasattr(item, 'text') and item.text and item.text.strip():
            text = item.text.strip()
            #law mesh heading default to paragraph
            elem = ParsedElement(
                element_id=_generate_element_id(doc_name, "paragraph", text, element_index),
                doc_name=doc_name,
                doc_path=str(doc_path),
                element_type="paragraph",
                content=text,
                section_path=list(current_section_path),
                language=_detect_language(text),
            )
            elements.append(elem)
            element_index += 1

        # Handle tables
        if item_type == "TableItem" or (hasattr(item, 'export_to_markdown') and 'table' in item_type.lower()):
            try:
                table_md = item.export_to_markdown() if hasattr(item, 'export_to_markdown') else None
                if table_md:
                    table_data = None
                    if hasattr(item, 'export_to_dataframe'):
                        try:
                            df = item.export_to_dataframe()
                            table_data = {
                                "headers": list(df.columns),
                                "rows": df.values.tolist(),
                                "shape": list(df.shape),
                            }
                        except Exception:
                            pass
                    #Tables handled
                    elem = ParsedElement(
                        element_id=_generate_element_id(doc_name, "table", table_md, element_index),
                        doc_name=doc_name,
                        doc_path=str(doc_path),
                        element_type="table",
                        content=table_md,
                        section_path=list(current_section_path),
                        language=_detect_language(table_md),
                        table_data=table_data,
                    )
                    elements.append(elem)
                    element_index += 1
            except Exception as e:
                warnings.warn(
                    f"Table extraction failed in {doc_name} "
                    f"(section: {' > '.join(current_section_path) or 'root'}): {e}"
                )

    return elements


def parse_document_fallback(doc_path: str | Path) -> list[ParsedElement]:
    """
    Fallback parser using python-docx directly.
    Used when Docling has issues with specific document formats.
    """
    from docx import Document

    doc_path = Path(doc_path)
    doc_name = doc_path.stem
    doc = Document(str(doc_path))

    elements: list[ParsedElement] = []
    current_section_path: list[str] = []
    element_index = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect headings by style
        style_name = para.style.name.lower() if para.style else ""
        if "heading" in style_name:
            level = 1
            for ch in style_name:
                if ch.isdigit():
                    level = int(ch)
                    break
            current_section_path = current_section_path[:level - 1]
            current_section_path.append(text)

            elem = ParsedElement(
                element_id=_generate_element_id(doc_name, "heading", text, element_index),
                doc_name=doc_name,
                doc_path=str(doc_path),
                element_type="heading",
                content=text,
                section_path=list(current_section_path),
                language=_detect_language(text),
            )
        else:
            elem = ParsedElement(
                element_id=_generate_element_id(doc_name, "paragraph", text, element_index),
                doc_name=doc_name,
                doc_path=str(doc_path),
                element_type="paragraph",
                content=text,
                section_path=list(current_section_path),
                language=_detect_language(text),
            )

        elements.append(elem)
        element_index += 1

    # Extract tables
    for table in doc.tables:
        rows = []
        headers = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            if i == 0:
                headers = cells
            else:
                rows.append(cells)

        # Build markdown representation
        if headers:
            md_lines = ["| " + " | ".join(headers) + " |"]
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                md_lines.append("| " + " | ".join(row) + " |")
            table_md = "\n".join(md_lines)

            elem = ParsedElement(
                element_id=_generate_element_id(doc_name, "table", table_md, element_index),
                doc_name=doc_name,
                doc_path=str(doc_path),
                element_type="table",
                content=table_md,
                section_path=list(current_section_path),
                language=_detect_language(table_md),
                table_data={
                    "headers": headers,
                    "rows": rows,
                    "shape": [len(rows), len(headers)],
                },
            )
            elements.append(elem)
            element_index += 1

    return elements


def parse_all_documents(docs_dir: str | Path, output_dir: Optional[str | Path] = None) -> list[ParsedElement]:
    """
    Parse all Word documents in a directory.
    Optionally saves parsed output as JSON.
    """
    docs_dir = Path(docs_dir)
    PARSE_FAILURES.clear()  # Reset from previous runs
    all_elements: list[ParsedElement] = []

    doc_files = list(docs_dir.glob("*.docx")) + list(docs_dir.glob("*.doc"))
    print(f"Found {len(doc_files)} documents in {docs_dir}")

    for doc_file in sorted(doc_files):
        if doc_file.name.startswith("~$"):
            continue  # Skip temp Word files
        print(f"  Parsing: {doc_file.name}")
        try:
            elements = parse_document(doc_file)
        except Exception as e:
            print(f"    Docling failed ({e}), trying fallback parser...")
            try:
                elements = parse_document_fallback(doc_file)
            except Exception as e2:
                warnings.warn(f"Both parsers failed for {doc_file.name}: {e2}. Skipping.")
                PARSE_FAILURES.append((doc_file.name, str(e2)))
                continue

        all_elements.extend(elements)
        print(f"    Extracted {len(elements)} elements")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "parsed_elements.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in all_elements], f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_elements)} elements to {output_path}")

    # ── Report any parse failures ──
    if PARSE_FAILURES:
        warnings.warn(
            f"{len(PARSE_FAILURES)} document(s) failed to parse: "
            + ", ".join(name for name, _ in PARSE_FAILURES)
        )

    return all_elements
