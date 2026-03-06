"""
Run the full evaluation suite: single-doc + multi-hop queries,
vector-only vs hybrid, with LLM-as-Judge scoring.

Usage:
  python run_eval.py                  # run all 30 queries
  python run_eval.py --quick          # run first 5 from each set
  python run_eval.py --dataset single # single-doc only
  python run_eval.py --dataset multi  # multi-hop only
"""

import argparse
import json
import sys
from pathlib import Path

from config import get_config
from src.evaluation.evaluator import (
    load_eval_dataset,
    run_evaluation,
    print_report,
)
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import VectorStore
from src.graph.kg_builder import KnowledgeGraph
from src.retrieval.hybrid_retriever import HybridRetriever

EVAL_DIR = Path(__file__).parent / "src" / "evaluation"
SINGLE_PATH = EVAL_DIR / "eval_dataset.json"
MULTI_PATH = EVAL_DIR / "eval_multihop.json"
OUTPUT_DIR = Path(__file__).parent / "eval_reports"


def main():
    parser = argparse.ArgumentParser(description="Banking Knowledge AI – Evaluation")
    parser.add_argument(
        "--dataset",
        choices=["single", "multi", "all"],
        default="all",
        help="Which dataset to evaluate (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run first 5 queries per dataset (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for JSON report",
    )
    args = parser.parse_args()

    config = get_config()

    # Load queries
    queries = []
    if args.dataset in ("single", "all") and SINGLE_PATH.exists():
        single = load_eval_dataset(SINGLE_PATH)
        print(f"Loaded {len(single)} single-doc queries")
        queries.extend(single)
    if args.dataset in ("multi", "all") and MULTI_PATH.exists():
        multi = load_eval_dataset(MULTI_PATH)
        print(f"Loaded {len(multi)} multi-hop queries")
        queries.extend(multi)

    if not queries:
        print("No evaluation queries found. Check paths:")
        print(f"  {SINGLE_PATH}")
        print(f"  {MULTI_PATH}")
        sys.exit(1)

    if args.quick:
        # Take up to 5 from each type
        single_q = [q for q in queries if not q.expected_docs or len(q.expected_docs) <= 1]
        multi_q = [q for q in queries if q.expected_docs and len(q.expected_docs) > 1]
        queries = single_q[:5] + multi_q[:5]
        print(f"Quick mode: running {len(queries)} queries")

    print(f"\nTotal queries: {len(queries)}")
    print(f"Judge model: {config.fast_llm.model}")
    print(f"Main LLM: {config.main_llm.model}")

    # Build retriever (assumes index is already built via build_index.py)
    graph_path = Path(config.graph_dir) / "knowledge_graph.json"
    chunks_path = Path(config.parsed_dir) / "chunks.json"

    if not (graph_path.exists() and chunks_path.exists()):
        print("Index not built. Run 'python build_index.py' first.")
        sys.exit(1)

    # Load graph
    kg = KnowledgeGraph(config)
    kg.load(graph_path)
    print(f"Graph: {kg.get_stats()['total_nodes']} nodes, {kg.get_stats()['total_edges']} edges")

    # Load chunks + vector store
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [Chunk(**c) for c in chunks_data]
    vector_store = VectorStore(config)
    vector_store.index_chunks(chunks)

    retriever = HybridRetriever(vector_store, kg, config)

    # Output path
    if args.output:
        save_path = Path(args.output)
    else:
        import time
        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = OUTPUT_DIR / f"eval_report_{ts}.json"

    # Run
    report = run_evaluation(
        eval_queries=queries,
        retriever=retriever,
        config=config,
        save_path=save_path,
    )

    # Print summary
    print_report(report)


if __name__ == "__main__":
    main()
