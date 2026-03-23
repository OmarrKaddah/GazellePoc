"""
Build the complete index: parse documents, chunk, embed, extract entities, build KG.
Logs all output to a log file as well as printing to console.

Usage:
    python build_index_with_log.py
"""

import json
import os
from pathlib import Path
import sys
from datetime import datetime

from config import get_config
from src.ingestion.parser import parse_all_documents, PARSE_FAILURES
from src.ingestion.chunker import chunk_elements, Chunk
from src.ingestion.embedder import VectorStore
from src.graph.entity_extractor import extract_entities_from_chunks, align_entities, FAILED_EXTRACTIONS
from src.graph.kg_builder import KnowledgeGraph
from src.auth.rbac import get_access_level

class Logger:
    def __init__(self, log_path):
        self.log_file = open(log_path, "a", encoding="utf-8")
    def log(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()
    def close(self):
        self.log_file.close()

def main():
    # Hardcoded paths and config values
    data_dir = Path("test_data/raw_docs_test")
    parsed_dir = Path("test_data/parsed_test")
    graph_dir = Path("test_data/graph_test")
    log_dir = Path("test_data/test_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"build_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = Logger(log_path)

    # Hardcoded chunking and model config
    max_chunk_tokens = 2048
    overlap_tokens = 128
    preserve_tables = True
    similarity_threshold = 0.85

    logger.log("=" * 60)
    logger.log("Banking Knowledge AI — Index Builder (LOGGED)")
    logger.log("Environment: HARDCODED")
    logger.log("LLM: openai/gpt-3.5-turbo")
    logger.log("Embedding: openai/text-embedding-ada-002")
    logger.log("=" * 60)

    # ── Step 1: Parse documents ──
    logger.log("\n[1/5] Parsing documents...")
    elements = parse_all_documents(data_dir, output_dir=parsed_dir)
    logger.log(f"  Total elements: {len(elements)}")
    logger.log(f"  Types: { {t: sum(1 for e in elements if e.element_type == t) for t in set(e.element_type for e in elements)} }")

    # ── Step 2: Chunk ──
    logger.log("\n[2/5] Chunking...")
    chunks = chunk_elements(
        elements,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
        preserve_tables=preserve_tables,
    )
    logger.log(f"  Total chunks: {len(chunks)}")
    logger.log(f"  Types: { {t: sum(1 for c in chunks if c.element_type == t) for t in set(c.element_type for c in chunks)} }")

    # ── Step 2b: Stamp RBAC access levels onto chunk metadata ──
    for chunk in chunks:
        chunk.metadata["access_level"] = get_access_level(chunk.doc_name)
    level_counts = {}
    for c in chunks:
        lvl = c.metadata["access_level"]
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    logger.log(f"  Access levels: {level_counts}  (1=Public, 2=Confidential, 3=Restricted)")

    # Save chunks for later loading
    chunks_path = parsed_dir / "chunks.json"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
    logger.log(f"  Saved chunks to {chunks_path}")

    # ── Step 3: Embed and index ──
    logger.log("\n[3/5] Embedding and indexing into vector store...")
    vector_store = VectorStore(None)  # Pass None or update VectorStore to not require config
    vector_store.index_chunks(chunks)
    info = vector_store.get_collection_info()
    logger.log(f"  Collection: {info}")

    # ── Step 4: Extract entities ──
    logger.log("\n[4/5] Extracting entities...")
    entities, relations = extract_entities_from_chunks(chunks, None, use_fast_llm=True)

    # Entity alignment
    canonical_map = align_entities(entities, similarity_threshold=similarity_threshold)

    # ── Step 5: Build knowledge graph ──
    logger.log("\n[5/5] Building knowledge graph...")
    kg = KnowledgeGraph(None)
    kg.add_chunks(chunks)
    kg.add_entities(entities, relations, canonical_map)
    kg.add_cross_document_links()

    stats = kg.get_stats()
    logger.log(f"  Graph stats: {json.dumps(stats, indent=2)}")

    # Save graph
    graph_path = graph_dir / "knowledge_graph.json"
    kg.save(graph_path)

    logger.log("\n" + "=" * 60)
    logger.log("Index build complete!")
    logger.log(f"  Chunks: {len(chunks)}")
    logger.log(f"  Entities: {len(entities)}")
    logger.log(f"  Relations: {len(relations)}")
    logger.log(f"  Graph nodes: {stats['total_nodes']}")
    logger.log(f"  Graph edges: {stats['total_edges']}")

    # ── Health Check ──
    logger.log(f"\n{'─' * 60}")
    logger.log("  HEALTH CHECK")
    logger.log(f"{'─' * 60}")

    max_tok = max_chunk_tokens
    oversized_chunks = [c for c in chunks if c.token_estimate > max_tok]
    danger_zone = [c for c in chunks if c.token_estimate > 7500]

    # Documents with zero entities
    doc_names = set(c.doc_name for c in chunks)
    entity_docs = set(e.source_doc for e in entities)
    docs_without_entities = doc_names - entity_docs

    issues_found = 0

    if PARSE_FAILURES:
        issues_found += 1
        logger.log(f"  ⚠ Parse failures:         {len(PARSE_FAILURES)} doc(s) skipped")
        for name, err in PARSE_FAILURES:
            logger.log(f"      • {name}: {err[:80]}")

    if oversized_chunks:
        issues_found += 1
        logger.log(f"  ⚠ Oversized chunks:       {len(oversized_chunks)} chunk(s) > {max_tok} tokens")
        for c in oversized_chunks[:5]:
            logger.log(f"      • {c.doc_name} [{c.element_type}] = {c.token_estimate} tokens")

    if danger_zone:
        issues_found += 1
        logger.log(f"  ⚠ Embedding danger zone:  {len(danger_zone)} chunk(s) > 7500 tokens (near model limit)")

    if FAILED_EXTRACTIONS:
        issues_found += 1
        rate = len(FAILED_EXTRACTIONS) / max(len(chunks), 1) * 100
        logger.log(f"  ⚠ Entity extraction fails: {len(FAILED_EXTRACTIONS)}/{len(chunks)} chunks ({rate:.1f}%)")

    if docs_without_entities:
        issues_found += 1
        logger.log(f"  ⚠ Docs without entities:  {len(docs_without_entities)}")
        for dn in sorted(docs_without_entities):
            logger.log(f"      • {dn}")

    if stats.get("dropped_relations", 0) > 0:
        issues_found += 1
        logger.log(f"  ⚠ Dropped relations:      {stats['dropped_relations']}")

    if stats.get("orphan_entities", 0) > 0:
        issues_found += 1
        logger.log(f"  ⚠ Orphan entities:        {stats['orphan_entities']}")

    if issues_found == 0:
        logger.log("  ✓ All clear — no silent failures detected.")

    logger.log(f"{'─' * 60}")
    logger.log(f"\nTo launch the app:")
    logger.log(f"  streamlit run app.py")
    logger.log("=" * 60)
    logger.close()

if __name__ == "__main__":
    main()
