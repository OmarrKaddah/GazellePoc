import warnings

"""
Document parser using Docling for structured extraction of Word documents.
Extracts sections, paragraphs, tables, and figures with full metadata.
"""

import json
import hashlib
import importlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from docling.document_converter import DocumentConverter
from config import get_config

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


def parse_document_docling(doc_path: str | Path) -> list[ParsedElement]:
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
        elif item_type == "TableItem" or (hasattr(item, 'export_to_markdown') and 'table' in item_type.lower()):
            try:
                table_md = item.export_to_markdown() if hasattr(item, 'export_to_markdown') else None
                if table_md:
                    print(f"Extracted table from {doc_name}: {table_md}")
                    table_data = None
                    if hasattr(item, 'export_to_dataframe'):
                        try:
                            df = item.export_to_dataframe()
                            print(f"Converted table to DataFrame =\n {df}")
                            table_data = {
                                "headers": [str(col) for col in df.columns],
                                "rows": [[str(cell) for cell in row] for row in df.values.tolist()],
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


def parse_document_unstructured(doc_path: str | Path) -> list[ParsedElement]:
    """
    Parse a single Word document using unstructured.

    Supports .docx directly and falls back to auto partition for other formats.
    """
    partition_docx_fn = None
    partition_auto_fn = None

    try:
        partition_docx_fn = importlib.import_module("unstructured.partition.docx").partition_docx
    except Exception:
        pass

    try:
        partition_auto_fn = importlib.import_module("unstructured.partition.auto").partition
    except Exception:
        pass

    if partition_docx_fn is None and partition_auto_fn is None:
        raise ImportError(
            "unstructured is not installed. Install with `pip install unstructured`."
        )

    doc_path = Path(doc_path)
    doc_name = doc_path.stem
    suffix = doc_path.suffix.lower()

    if suffix == ".docx" and partition_docx_fn is not None:
        raw_elements = partition_docx_fn(filename=str(doc_path))
    elif partition_auto_fn is not None:
        raw_elements = partition_auto_fn(filename=str(doc_path))
    else:
        raise ImportError(
            "unstructured auto partition is unavailable for non-.docx documents."
        )

    elements: list[ParsedElement] = []
    current_section_path: list[str] = []
    element_index = 0

    for item in raw_elements:
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        item_type = str(getattr(item, "category", type(item).__name__)).lower()
        item_metadata = {}
        page_number = None

        raw_metadata = getattr(item, "metadata", None)
        if raw_metadata is not None:
            if hasattr(raw_metadata, "to_dict"):
                item_metadata = raw_metadata.to_dict()
            elif isinstance(raw_metadata, dict):
                item_metadata = raw_metadata
            page_number = item_metadata.get("page_number")

        if "title" in item_type or "header" in item_type:
            element_type = "heading"
            current_section_path.append(text)
            section_path = list(current_section_path)
        elif "table" in item_type:
            element_type = "table"
            section_path = list(current_section_path)
        elif "list" in item_type:
            element_type = "list_item"
            section_path = list(current_section_path)
        else:
            element_type = "paragraph"
            section_path = list(current_section_path)

        table_data = None
        if element_type == "table":
            table_cells = item_metadata.get("table_as_cells")
            if isinstance(table_cells, list) and table_cells:
                headers = [str(cell).strip() for cell in table_cells[0]]
                rows = [[str(cell).strip() for cell in row] for row in table_cells[1:]]
                table_data = {
                    "headers": headers,
                    "rows": rows,
                    "shape": [len(rows), len(headers)],
                }

        elem = ParsedElement(
            element_id=_generate_element_id(doc_name, element_type, text, element_index),
            doc_name=doc_name,
            doc_path=str(doc_path),
            element_type=element_type,
            content=text,
            section_path=section_path,
            metadata=item_metadata,
            language=_detect_language(text),
            page_number=page_number,
            table_data=table_data,
        )
        elements.append(elem)
        element_index += 1

    return elements


def parse_document(doc_path: str | Path, parser_backend: Optional[str] = None) -> list[ParsedElement]:
    """Parse a single document using the configured parser backend."""
    backend = (parser_backend or get_config().doc_parser).strip().lower()

    if backend == "docling":
        return parse_document_docling(doc_path)
    if backend == "unstructured":
        return parse_document_unstructured(doc_path)

    raise ValueError(f"Unsupported parser backend '{backend}'. Use 'docling' or 'unstructured'.")


def parse_all_documents(docs_dir: str | Path, output_dir: Optional[str | Path] = None) -> list[ParsedElement]:
    """
    Parse all Word documents in a directory.
    Optionally saves parsed output as JSON.
    """
    docs_dir = Path(docs_dir)
    PARSE_FAILURES.clear()  # Reset from previous runs
    all_elements: list[ParsedElement] = []


    doc_files = list(docs_dir.glob("*.docx")) + list(docs_dir.glob("*.doc"))
    all_files = list(sorted(set(doc_files)))
    print(f"Found {len(all_files)} documents in {docs_dir}")
    print(doc_files)
    for doc_file in sorted(all_files):
        if doc_file.name.startswith("~$"):
            continue  # Skip temp Word files
        print(f"  Parsing: {doc_file.name}")
        try:
            elements = parse_document(doc_file)
        except Exception as e:
            warnings.warn(f"Document parser failed for {doc_file.name}: {e}. Skipping.")
            PARSE_FAILURES.append((doc_file.name, str(e)))
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
