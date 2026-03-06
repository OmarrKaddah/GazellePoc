"""
Build the complete index: parse documents, chunk, embed, extract entities, build KG.
Run this once before launching the app.

Usage:
    python build_index.py
"""

import json
import os
from pathlib import Path

from config import get_config
from src.ingestion.parser import parse_all_documents
from src.ingestion.chunker import chunk_elements, Chunk
from src.ingestion.embedder import VectorStore
from src.graph.entity_extractor import extract_entities_from_chunks, align_entities
from src.graph.kg_builder import KnowledgeGraph


def main():
    # Load env vars from .env if present
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    config = get_config()

    print("=" * 60)
    print("Banking Knowledge AI — Index Builder")
    print(f"Environment: {config.env}")
    print(f"LLM: {config.fast_llm.provider}/{config.fast_llm.model}")
    print(f"Embedding: {config.embedding.provider}/{config.embedding.model}")
    print("=" * 60)

    data_dir = Path(config.data_dir)
    parsed_dir = Path(config.parsed_dir)
    graph_dir = Path(config.graph_dir)

    # ── Step 1: Parse documents ──
    print("\n[1/5] Parsing documents...")
    elements = parse_all_documents(data_dir, output_dir=parsed_dir)
    print(f"  Total elements: {len(elements)}")
    print(f"  Types: { {t: sum(1 for e in elements if e.element_type == t) for t in set(e.element_type for e in elements)} }")

    # ── Step 2: Chunk ──
    print("\n[2/5] Chunking...")
    chunks = chunk_elements(
        elements,
        max_chunk_tokens=config.chunking.max_chunk_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
        preserve_tables=config.chunking.preserve_tables,
    )
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Types: { {t: sum(1 for c in chunks if c.element_type == t) for t in set(c.element_type for c in chunks)} }")

    # Save chunks for later loading
    chunks_path = parsed_dir / "chunks.json"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
    print(f"  Saved chunks to {chunks_path}")

    # ── Step 3: Embed and index ──
    print("\n[3/5] Embedding and indexing into vector store...")
    vector_store = VectorStore(config)
    vector_store.index_chunks(chunks)
    info = vector_store.get_collection_info()
    print(f"  Collection: {info}")

    # ── Step 4: Extract entities ──
    print("\n[4/5] Extracting entities...")
    entities, relations = extract_entities_from_chunks(chunks, config, use_fast_llm=True)

    # Entity alignment
    canonical_map = align_entities(entities, similarity_threshold=config.graph.similarity_threshold)

    # ── Step 5: Build knowledge graph ──
    print("\n[5/5] Building knowledge graph...")
    kg = KnowledgeGraph(config)
    kg.add_chunks(chunks)
    kg.add_entities(entities, relations, canonical_map)
    kg.add_cross_document_links()

    stats = kg.get_stats()
    print(f"  Graph stats: {json.dumps(stats, indent=2)}")

    # Save graph
    graph_path = graph_dir / "knowledge_graph.json"
    kg.save(graph_path)

    print("\n" + "=" * 60)
    print("Index build complete!")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Entities: {len(entities)}")
    print(f"  Relations: {len(relations)}")
    print(f"  Graph nodes: {stats['total_nodes']}")
    print(f"  Graph edges: {stats['total_edges']}")
    print(f"\nTo launch the app:")
    print(f"  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
