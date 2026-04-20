"""
OCR JSON parser that groups individual OCR results into meaningful sentences/paragraphs.

Strategy:
1. Group OCR items into lines based on spatial proximity (y-coordinate)
2. Group lines into paragraphs based on y-gap heuristics
3. Convert to ParsedElement objects similar to Word document parsing
4. Handle Arabic RTL text and confidence filtering
"""

import json
import hashlib
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ParsedElement:
    """An atomic unit extracted from a document."""
    element_id: str
    doc_name: str
    doc_path: str
    element_type: str  # "paragraph", "heading", "section"
    content: str
    section_path: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    language: str = "unknown"
    page_number: Optional[int] = None
    table_data: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


# Module-level tracker for parse failures
PARSE_FAILURES: list[tuple[str, str]] = []


def _generate_element_id(doc_name: str, element_type: str, content: str, index: int) -> str:
    """Generate a deterministic unique ID for an element."""
    raw = f"{doc_name}::{element_type}::{index}::{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _detect_language(text: str) -> str:
    """Detect language (Arabic vs English)."""
    if not text.strip():
        return "unknown"
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    ratio = arabic_chars / max(len(text), 1)
    return "ar" if ratio > 0.3 else "en"


def _get_bbox_y_center(bbox: list) -> float:
    """Extract y-coordinate center from bounding box coordinates.
    
    bbox format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    if not bbox or len(bbox) < 2:
        return 0
    # Average the y-coordinates of all 4 points
    y_coords = [point[1] for point in bbox if len(point) > 1]
    return sum(y_coords) / len(y_coords) if y_coords else 0


def _get_bbox_x_left(bbox: list) -> float:
    """Extract leftmost x-coordinate from bounding box."""
    if not bbox or len(bbox) < 2:
        return 0
    x_coords = [point[0] for point in bbox if len(point) > 0]
    return min(x_coords) if x_coords else 0


def _group_ocr_items_into_lines(ocr_items: list[dict], y_threshold: float = 30) -> list[list[dict]]:
    """
    Group OCR items into lines based on y-coordinate proximity.
    
    Items with similar y-coordinates are grouped together as a line.
    Args:
        ocr_items: List of OCR result items
        y_threshold: Maximum y-distance to consider items on same line
    
    Returns:
        List of lines, where each line is a list of OCR items
    """
    if not ocr_items:
        return []
    
    # Sort by y-coordinate
    sorted_items = sorted(ocr_items, key=lambda item: _get_bbox_y_center(item.get("bbox", [])))
    
    lines: list[list[dict]] = []
    current_line: list[dict] = []
    current_y: Optional[float] = None
    
    for item in sorted_items:
        item_y = _get_bbox_y_center(item.get("bbox", []))
        
        if current_y is None:
            current_y = item_y
            current_line = [item]
        elif abs(item_y - current_y) <= y_threshold:
            # Same line
            current_line.append(item)
        else:
            # New line
            if current_line:
                lines.append(current_line)
            current_line = [item]
            current_y = item_y
    
    if current_line:
        lines.append(current_line)
    
    return lines


def _sort_line_rtl(line_items: list[dict]) -> list[dict]:
    """
    Sort items within a line for RTL (Arabic) reading order.
    
    Items should be sorted from right to left (highest x to lowest x).
    """
    return sorted(line_items, key=lambda item: -_get_bbox_x_left(item.get("bbox", [])))


def _merge_ocr_items(items: list[dict]) -> str:
    """Merge multiple OCR items into a single string."""
    texts = [item.get("text", "").strip() for item in items if item.get("text", "").strip()]
    return " ".join(texts)


def _group_lines_into_paragraphs(
    lines: list[list[dict]], 
    min_line_gap: float = 50,
    max_lines_per_paragraph: int = 10
) -> list[list[list[dict]]]:
    """
    Group lines into paragraphs based on vertical gap heuristics.
    
    Args:
        lines: List of lines (each line is a list of OCR items)
        min_line_gap: Minimum vertical gap to start a new paragraph
        max_lines_per_paragraph: Maximum lines per paragraph before forcing split
    
    Returns:
        List of paragraphs (each paragraph is a list of lines)
    """
    if not lines:
        return []
    
    paragraphs: list[list[list[dict]]] = []
    current_paragraph: list[list[dict]] = []
    last_line_y: Optional[float] = None
    current_line_count = 0
    
    for line in lines:
        if not line:
            continue
        
        # Get y-coordinate of this line
        line_y = _get_bbox_y_center(line[0].get("bbox", []))
        
        # Check if we should start a new paragraph
        should_new_paragraph = False
        
        if last_line_y is None:
            # First line
            should_new_paragraph = False
        elif line_y - last_line_y > min_line_gap:
            # Large vertical gap detected
            should_new_paragraph = True
        elif current_line_count >= max_lines_per_paragraph:
            # Paragraph is getting too long
            should_new_paragraph = True
        
        if should_new_paragraph and current_paragraph:
            paragraphs.append(current_paragraph)
            current_paragraph = []
            current_line_count = 0
        
        current_paragraph.append(line)
        last_line_y = line_y
        current_line_count += 1
    
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs


def _detect_is_heading(text: str, confidence: float) -> bool:
    """
    Heuristic to detect if text is a heading.
    
    Headings typically:
    - Are short (<100 chars)
    - Have high confidence
    - Are Arabic words like "المادة الأولى" (Article 1)
    """
    if confidence < 0.85:
        return False
    
    # Check for common heading patterns
    heading_keywords = [
        "المادة",  # Article
        "الفصل",   # Chapter
        "القسم",   # Section
        "الباب",   # Part
    ]
    
    return any(keyword in text for keyword in heading_keywords)


def parse_ocr_json(json_path: str | Path) -> list[ParsedElement]:
    """
    Parse OCR JSON output into structured ParsedElement objects.
    
    Groups individual OCR items into meaningful lines and paragraphs,
    then converts them to ParsedElement objects.
    """
    json_path = Path(json_path)
    doc_name = json_path.stem
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON: {e}")
    
    elements: list[ParsedElement] = []
    element_index = 0
    documents = data.get("documents", [])
    
    if not documents:
        return elements
    
    for doc in documents:
        if doc.get("status") != "success":
            continue
        
        raw_ocr = doc.get("raw_ocr", {})
        ocr_items = raw_ocr.get("arabic_results", [])
        
        if not ocr_items:
            continue
        
        # Filter by confidence threshold
        high_confidence_items = [
            item for item in ocr_items 
            if item.get("confidence", 0) >= 0.4
        ]
        
        if not high_confidence_items:
            continue
        
        # Step 1: Group items into lines
        lines = _group_ocr_items_into_lines(high_confidence_items, y_threshold=30)
        
        # Step 2: Sort each line for RTL
        sorted_lines = [_sort_line_rtl(line) for line in lines]
        
        # Step 3: Group lines into paragraphs
        paragraphs = _group_lines_into_paragraphs(sorted_lines, min_line_gap=50)
        
        # Step 4: Convert to ParsedElement objects
        current_section_path: list[str] = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Merge all lines in paragraph
            paragraph_texts: list[str] = []
            avg_confidence = 0
            total_confidence = 0
            item_count = 0
            
            for line in paragraph:
                line_text = _merge_ocr_items(line)
                paragraph_texts.append(line_text)
                # Track confidence
                for item in line:
                    total_confidence += item.get("confidence", 0)
                    item_count += 1
            
            if item_count > 0:
                avg_confidence = total_confidence / item_count
            
            full_text = "\n".join(paragraph_texts)
            
            # Check if this is a heading
            is_heading = _detect_is_heading(full_text, avg_confidence)
            
            if is_heading:
                element_type = "heading"
                # Update section path
                current_section_path = [full_text]
            else:
                element_type = "paragraph"
            
            elem = ParsedElement(
                element_id=_generate_element_id(doc_name, element_type, full_text, element_index),
                doc_name=doc_name,
                doc_path=str(json_path),
                element_type=element_type,
                content=full_text,
                section_path=list(current_section_path),
                language=_detect_language(full_text),
                metadata={
                    "source": "ocr",
                    "average_confidence": round(avg_confidence, 3),
                    "line_count": len(paragraph),
                    "item_count": item_count,
                }
            )
            
            elements.append(elem)
            element_index += 1
    
    return elements


def parse_all_ocr_documents(
    docs_dir: str | Path,
    output_dir: Optional[str | Path] = None
) -> list[ParsedElement]:
    """
    Parse all OCR JSON files in a directory.
    
    Args:
        docs_dir: Directory containing .json OCR files
        output_dir: Optional directory to save parsed elements
    
    Returns:
        List of ParsedElement objects
    """
    docs_dir = Path(docs_dir)
    PARSE_FAILURES.clear()
    all_elements: list[ParsedElement] = []
    
    # Find all JSON files
    ocr_files = list(docs_dir.glob("*.json"))
    
    if not ocr_files:
        print(f"Found 0 OCR JSON files in {docs_dir}")
        return all_elements
    
    print(f"Found {len(ocr_files)} OCR JSON files in {docs_dir}")
    
    for ocr_file in sorted(ocr_files):
        print(f"  Parsing: {ocr_file.name}")
        
        try:
            elements = parse_ocr_json(ocr_file)
            all_elements.extend(elements)
            print(f"    Extracted {len(elements)} elements")
        
        except Exception as e:
            error_msg = str(e)
            warnings.warn(f"OCR parser failed for {ocr_file.name}: {error_msg}")
            PARSE_FAILURES.append((ocr_file.name, error_msg))
    
    # Save parsed elements if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "parsed_ocr_elements.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in all_elements], f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(all_elements)} OCR elements to {output_path}")
    
    # Report failures
    if PARSE_FAILURES:
        warnings.warn(
            f"{len(PARSE_FAILURES)} OCR file(s) failed to parse: "
            + ", ".join(name for name, _ in PARSE_FAILURES)
        )
    
    return all_elements

    return all_elements