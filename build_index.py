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
from src.ingestion.parser import parse_all_documents, PARSE_FAILURES
from src.ingestion.chunker_v2 import chunk_elements, Chunk
from src.ingestion.embedder import VectorStore
from src.graph.entity_extractor import extract_entities_from_chunks, align_entities, FAILED_EXTRACTIONS
from src.graph.kg_builder import KnowledgeGraph
from src.auth.rbac import get_access_level


def main():
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
    chunks = chunk_elements(elements, config.chunking)
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Types: { {t: sum(1 for c in chunks if c.element_type == t) for t in set(c.element_type for c in chunks)} }")

    # ── Step 2b: Stamp RBAC access levels onto chunks ──
    for chunk in chunks:
        chunk.access_level = get_access_level(chunk.doc_name)
    level_counts: dict[int, int] = {}
    for c in chunks:
        lvl = c.access_level or 0
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    print(f"  Access levels: {level_counts}  (1=Public, 2=Confidential, 3=Restricted)")

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

    # ── Health Check ──
    print(f"\n{'─' * 60}")
    print("  HEALTH CHECK")
    print(f"{'─' * 60}")

    max_tok = config.chunking.max_chunk_tokens
    oversized_chunks = [c for c in chunks if c.token_estimate > max_tok]
    danger_zone = [c for c in chunks if c.token_estimate > 7500]

    # Documents with zero entities
    doc_names = set(c.doc_name for c in chunks)
    entity_docs = set(e.source_doc for e in entities)
    docs_without_entities = doc_names - entity_docs

    issues_found = 0

    if PARSE_FAILURES:
        issues_found += 1
        print(f"  ⚠ Parse failures:         {len(PARSE_FAILURES)} doc(s) skipped")
        for name, err in PARSE_FAILURES:
            print(f"      • {name}: {err[:80]}")

    if oversized_chunks:
        issues_found += 1
        print(f"  ⚠ Oversized chunks:       {len(oversized_chunks)} chunk(s) > {max_tok} tokens")
        for c in oversized_chunks[:5]:
            print(f"      • {c.doc_name} [{c.element_type}] = {c.token_estimate} tokens")

    if danger_zone:
        issues_found += 1
        print(f"  ⚠ Embedding danger zone:  {len(danger_zone)} chunk(s) > 7500 tokens (near model limit)")

    if FAILED_EXTRACTIONS:
        issues_found += 1
        rate = len(FAILED_EXTRACTIONS) / max(len(chunks), 1) * 100
        print(f"  ⚠ Entity extraction fails: {len(FAILED_EXTRACTIONS)}/{len(chunks)} chunks ({rate:.1f}%)")

    if docs_without_entities:
        issues_found += 1
        print(f"  ⚠ Docs without entities:  {len(docs_without_entities)}")
        for dn in sorted(docs_without_entities):
            print(f"      • {dn}")

    if stats.get("dropped_relations", 0) > 0:
        issues_found += 1
        print(f"  ⚠ Dropped relations:      {stats['dropped_relations']}")

    if stats.get("orphan_entities", 0) > 0:
        issues_found += 1
        print(f"  ⚠ Orphan entities:        {stats['orphan_entities']}")

    if issues_found == 0:
        print("  ✓ All clear — no silent failures detected.")

    print(f"{'─' * 60}")
    print(f"\nTo launch the app:")
    print(f"  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
