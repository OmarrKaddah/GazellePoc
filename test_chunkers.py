"""
Test chunker_v2 and chunker_v3 independently on text files.

Usage:
    python test_chunkers.py <path_to_txt_file> [<path_to_another_txt_file> ...]
    
Example:
    python test_chunkers.py data/raw_docs/document.txt
    python test_chunkers.py data/raw_docs/*.txt
"""

import sys
from pathlib import Path

from config import get_config
from src.ingestion.parser import ParsedElement
from src.ingestion.chunker_v2 import chunk_elements
from src.ingestion.chunker_v3 import chunk_elements_semantic


def load_txt_files(file_paths: list[str]) -> list[ParsedElement]:
    """Load text files and convert to ParsedElements."""
    elements = []
    
    for file_path in file_paths:
        p = Path(file_path)
        if not p.exists():
            print(f"⚠ File not found: {file_path}")
            continue
        
        doc_name = p.name
        content = p.read_text(encoding="utf-8")
        
        if not content.strip():
            print(f"⚠ File is empty: {file_path}")
            continue
        
        element = ParsedElement(
            element_id=f"{doc_name}_0",
            doc_name=doc_name,
            doc_path=str(p.absolute()),
            element_type="text",
            content=content,
            section_path=["Content"],
            language="en",
        )
        elements.append(element)
        print(f"✓ Loaded: {doc_name} ({len(content)} chars)")
    
    return elements


def print_chunk_stats(chunks, title):
    """Print statistics about chunks."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    
    if chunks:
        print(f"\nChunk breakdown by type:")
        type_counts = {}
        for chunk in chunks:
            type_counts[chunk.element_type] = type_counts.get(chunk.element_type, 0) + 1
        for ctype, count in type_counts.items():
            print(f"  {ctype}: {count}")
        
        print(f"\nToken statistics:")
        token_counts = [c.token_estimate for c in chunks]
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print(f"  Avg: {sum(token_counts) / len(token_counts):.1f}")
        print(f"  Total: {sum(token_counts)}")
        
        print(f"\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n  Chunk {i}:")
            print(f"    ID: {chunk.chunk_id}")
            print(f"    Type: {chunk.element_type}")
            print(f"    Tokens: {chunk.token_estimate}")
            print(f"    Content preview: {chunk.content[:100]}...")

import json


def save_chunks(chunks, output_dir: str, chunker_name: str):
    """Save chunks to json files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        chunk_data = {
            "chunk_id": chunk.chunk_id,
            "element_type": chunk.element_type,
            "token_estimate": chunk.token_estimate,
            "doc_name": chunk.doc_name,
            "content": chunk.content,
            "metadata": getattr(chunk, "metadata", {}),
        }

        file_name = f"{chunker_name}_chunk_{i+1}.json"

        with open(out_path / file_name, "w", encoding="utf-8") as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(chunks)} chunks to {out_path}")

def main():
    config = get_config()
    
    # Get file paths from command line
    if len(sys.argv) < 2:
        print("Usage: python test_chunkers.py <path_to_txt_file> [<path_to_another_txt_file> ...]")
        print("\nExample:")
        print("  python test_chunkers.py data/raw_docs/document.txt")
        print("  python test_chunkers.py test_files/*.txt")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    
    print("Testing Chunker V2 vs V3")
    print("=" * 60)
    
    # Load text files
    print(f"\nLoading {len(file_paths)} file(s)...")
    elements = load_txt_files(file_paths)
    
    if not elements:
        print("❌ No elements loaded. Check file paths.")
        sys.exit(1)
    
    print(f"✓ Total elements: {len(elements)}\n")
    
    # Test V2
    print("\n[1/2] Testing chunker_v2...")
    try:
        chunks_v2 = chunk_elements(elements, config.chunking)
        print_chunk_stats(chunks_v2, "CHUNKER V2 Results")
    except Exception as e:
        print(f"❌ V2 failed: {e}")
        chunks_v2 = []
    
    print_chunk_stats(chunks_v2, "CHUNKER V2 Results")
    save_chunks(chunks_v2, "test_results/chunks/v2", "v2")
    # Test V3
    print("\n[2/2] Testing chunker_v3...")
    try:
        chunks_v3 = chunk_elements_semantic(
            elements,
            config.chunking,
            breakpoint_threshold_type=config.chunking.semantic_breakpoint_type,
            breakpoint_threshold_amount=config.chunking.semantic_breakpoint_amount,
        )
        print_chunk_stats(chunks_v3, "CHUNKER V3 Results")
    except Exception as e:
        print(f"❌ V3 failed: {e}")
        chunks_v3 = []

    print_chunk_stats(chunks_v3, "CHUNKER V3 Results")
    save_chunks(chunks_v3, "test_results/chunks/v3", "v3")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"V2 chunks: {len(chunks_v2)}")
    print(f"V3 chunks: {len(chunks_v3)}")
    
    if chunks_v2 and chunks_v3:
        v2_tokens = sum(c.token_estimate for c in chunks_v2)
        v3_tokens = sum(c.token_estimate for c in chunks_v3)
        print(f"V2 total tokens: {v2_tokens}")
        print(f"V3 total tokens: {v3_tokens}")
        print(f"Difference: {abs(v2_tokens - v3_tokens)} tokens")


if __name__ == "__main__":
    main()
