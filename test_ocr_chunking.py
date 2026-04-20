"""
Test script: Parse OCR JSON and chunk output only.
Saves results to ocr_chunking_output.txt
"""

from pathlib import Path
from src.ingestion.ocr_structuring import parse_all_ocr_documents
from src.ingestion.chunker import chunk_elements
from config import get_config

# Get config
config = get_config()
data_dir = Path(config.data_dir)

# Output file
output_file = Path("ocr_chunking_output.txt")
output = []

# Parse OCR documents
output.append("=" * 60)
output.append("Testing OCR → Chunking Pipeline")
output.append("=" * 60)

output.append(f"\n[1] Parsing OCR JSON files from {data_dir}...")
elements = parse_all_ocr_documents(data_dir)
output.append(f"✓ Parsed {len(elements)} elements")

if not elements:
    output.append("No elements found. Check data_dir for .json files.")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    exit(1)

# Show parsed elements
output.append(f"\n[2] Parsed Elements Summary:")
for i, elem in enumerate(elements[:5]):  # Show first 5
    output.append(f"\n  Element {i+1}:")
    output.append(f"    Type: {elem.element_type}")
    output.append(f"    Content: {elem.content[:100]}...")
    output.append(f"    Confidence: {elem.metadata.get('average_confidence', 'N/A')}")

if len(elements) > 5:
    output.append(f"\n  ... and {len(elements) - 5} more elements")

# Chunk the elements
output.append(f"\n[3] Chunking with token limit {config.chunking.max_chunk_tokens}...")
chunks = chunk_elements(
    elements,
    max_chunk_tokens=config.chunking.max_chunk_tokens,
    overlap_tokens=config.chunking.overlap_tokens,
    preserve_tables=config.chunking.preserve_tables,
)
output.append(f"✓ Created {len(chunks)} chunks")

# Show chunks
output.append(f"\n[4] Chunks Summary:")
for i, chunk in enumerate(chunks[:5]):  # Show first 5
    output.append(f"\n  Chunk {i+1}:")
    output.append(f"    Type: {chunk.element_type}")
    output.append(f"    Tokens: {chunk.token_estimate}")
    output.append(f"    Content: {chunk.content[:100]}...")

if len(chunks) > 5:
    output.append(f"\n  ... and {len(chunks) - 5} more chunks")

# Statistics
output.append(f"\n[5] Statistics:")
output.append(f"  Total input elements: {len(elements)}")
output.append(f"  Total chunks: {len(chunks)}")
element_types = {t: sum(1 for e in elements if e.element_type == t) for t in set(e.element_type for e in elements)}
chunk_types = {t: sum(1 for c in chunks if c.element_type == t) for t in set(c.element_type for c in chunks)}
output.append(f"  Element types: {element_types}")
output.append(f"  Chunk types: {chunk_types}")

output.append("\n" + "=" * 60)
output.append("✓ Test complete!")
output.append("=" * 60)

# Write to file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print(f"✓ Output saved to {output_file.resolve()}")
