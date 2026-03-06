"""
Evaluation framework.
Compares: Vector-Only RAG (A) vs Hybrid Vector+Graph (B).
Measures: retrieval recall, answer correctness, citation accuracy, hallucination rate.
"""

import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from config import get_config
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResponse
from src.generation.aggregator import generate_response


@dataclass
class EvalQuery:
    """A single evaluation query with expected answer."""
    query_id: str
    question: str
    expected_answer: str
    source_doc: str
    source_section: str
    reasoning_type: str  # "single_hop", "multi_hop", "table_lookup", "cross_reference", "conditional"
    language: str = "en"
    expected_entities: list[str] = None

    def __post_init__(self):
        if self.expected_entities is None:
            self.expected_entities = []


@dataclass
class EvalResult:
    """Evaluation result for a single query."""
    query_id: str
    question: str
    reasoning_type: str
    # Retrieval metrics
    retrieval_recall: float  # Did we retrieve the right document/section?
    retrieval_precision: float  # How many retrieved chunks were relevant?
    # Answer metrics
    answer_generated: str
    answer_correct: bool  # Manual or LLM-judged
    citation_accurate: bool  # Did citations match actual sources?
    hallucination_detected: bool  # Answer contains claims not in evidence?
    # System metrics
    evidence_sufficient: bool
    confidence_score: float
    latency_ms: float
    # Mode
    retrieval_mode: str  # "vector_only" or "hybrid"

    def to_dict(self) -> dict:
        return asdict(self)


def load_eval_dataset(path: str | Path) -> list[EvalQuery]:
    """Load evaluation dataset from JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalQuery(**q) for q in data]


def save_eval_dataset(queries: list[EvalQuery], path: str | Path):
    """Save evaluation dataset to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(q) for q in queries], f, ensure_ascii=False, indent=2)


def evaluate_single_query(
    eval_query: EvalQuery,
    retriever: HybridRetriever,
    use_graph: bool = True,
    config=None,
) -> EvalResult:
    """
    Run a single evaluation query through the pipeline and measure results.
    """
    config = config or get_config()
    mode = "hybrid" if use_graph else "vector_only"

    # Time the retrieval + generation
    start = time.time()

    # Retrieve
    retrieval = retriever.retrieve(
        query=eval_query.question,
        use_graph=use_graph,
    )

    # Generate
    response = generate_response(
        query=eval_query.question,
        retrieval=retrieval,
        config=config,
    )

    latency = (time.time() - start) * 1000

    # ── Measure retrieval recall ──
    # Did we retrieve chunks from the expected document/section?
    expected_doc = eval_query.source_doc.lower()
    expected_section = eval_query.source_section.lower()

    doc_hit = any(
        expected_doc in r.doc_name.lower()
        for r in retrieval.results
    )
    section_hit = any(
        expected_section in " > ".join(r.section_path).lower()
        for r in retrieval.results
    )
    retrieval_recall = 1.0 if (doc_hit and section_hit) else (0.5 if doc_hit else 0.0)

    # Retrieval precision (rough: what fraction of results are from expected doc)
    relevant_count = sum(
        1 for r in retrieval.results
        if expected_doc in r.doc_name.lower()
    )
    retrieval_precision = relevant_count / max(len(retrieval.results), 1)

    # ── Answer correctness ──
    # Simple: check if key terms from expected answer appear in generated answer
    answer_text = response.get("answer", "")
    expected_lower = eval_query.expected_answer.lower()
    answer_lower = answer_text.lower()

    # Check for key numeric values and terms
    expected_words = set(expected_lower.split())
    answer_words = set(answer_lower.split())
    overlap = len(expected_words & answer_words) / max(len(expected_words), 1)
    answer_correct = overlap > 0.3  # 30% word overlap as rough heuristic

    # ── Citation accuracy ──
    citation_accurate = doc_hit  # If the right doc was retrieved, citations should be correct

    # ── Hallucination detection ──
    # Simple: if the system said evidence was insufficient but still answered, that's a problem.
    # More sophisticated: check if answer contains specific claims not traceable to retrieved chunks.
    hallucination_detected = False
    if not retrieval.evidence_sufficient and answer_text and "cannot answer" not in answer_lower:
        hallucination_detected = True

    return EvalResult(
        query_id=eval_query.query_id,
        question=eval_query.question,
        reasoning_type=eval_query.reasoning_type,
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
        answer_generated=answer_text[:500],  # Truncate for storage
        answer_correct=answer_correct,
        citation_accurate=citation_accurate,
        hallucination_detected=hallucination_detected,
        evidence_sufficient=retrieval.evidence_sufficient,
        confidence_score=retrieval.confidence_score,
        latency_ms=latency,
        retrieval_mode=mode,
    )


def run_evaluation(
    eval_queries: list[EvalQuery],
    retriever: HybridRetriever,
    config=None,
) -> dict:
    """
    Run full comparative evaluation: vector-only vs hybrid.
    Returns aggregated metrics for both modes.
    """
    config = config or get_config()

    results_vector = []
    results_hybrid = []

    for i, eq in enumerate(eval_queries):
        print(f"Evaluating query {i + 1}/{len(eval_queries)}: {eq.question[:60]}...")

        # Run vector-only
        r_vec = evaluate_single_query(eq, retriever, use_graph=False, config=config)
        results_vector.append(r_vec)

        # Run hybrid
        r_hyb = evaluate_single_query(eq, retriever, use_graph=True, config=config)
        results_hybrid.append(r_hyb)

    # Aggregate
    report = {
        "vector_only": _aggregate_results(results_vector),
        "hybrid": _aggregate_results(results_hybrid),
        "per_query_type": _aggregate_by_type(results_vector, results_hybrid),
        "detailed_results": {
            "vector_only": [r.to_dict() for r in results_vector],
            "hybrid": [r.to_dict() for r in results_hybrid],
        },
    }

    return report


def _aggregate_results(results: list[EvalResult]) -> dict:
    """Aggregate metrics across all results."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "num_queries": n,
        "avg_retrieval_recall": sum(r.retrieval_recall for r in results) / n,
        "avg_retrieval_precision": sum(r.retrieval_precision for r in results) / n,
        "answer_accuracy": sum(1 for r in results if r.answer_correct) / n,
        "citation_accuracy": sum(1 for r in results if r.citation_accurate) / n,
        "hallucination_rate": sum(1 for r in results if r.hallucination_detected) / n,
        "abstention_rate": sum(1 for r in results if not r.evidence_sufficient) / n,
        "avg_confidence": sum(r.confidence_score for r in results) / n,
        "avg_latency_ms": sum(r.latency_ms for r in results) / n,
    }


def _aggregate_by_type(
    results_vector: list[EvalResult],
    results_hybrid: list[EvalResult],
) -> dict:
    """Aggregate metrics broken down by reasoning type (the key thesis analysis)."""
    types = set(r.reasoning_type for r in results_vector)
    breakdown = {}

    for rtype in types:
        vec_subset = [r for r in results_vector if r.reasoning_type == rtype]
        hyb_subset = [r for r in results_hybrid if r.reasoning_type == rtype]

        breakdown[rtype] = {
            "vector_only": _aggregate_results(vec_subset),
            "hybrid": _aggregate_results(hyb_subset),
            "improvement": {
                "retrieval_recall_delta": (
                    _aggregate_results(hyb_subset).get("avg_retrieval_recall", 0)
                    - _aggregate_results(vec_subset).get("avg_retrieval_recall", 0)
                ),
                "answer_accuracy_delta": (
                    _aggregate_results(hyb_subset).get("answer_accuracy", 0)
                    - _aggregate_results(vec_subset).get("answer_accuracy", 0)
                ),
                "hallucination_rate_delta": (
                    _aggregate_results(hyb_subset).get("hallucination_rate", 0)
                    - _aggregate_results(vec_subset).get("hallucination_rate", 0)
                ),
            },
        }

    return breakdown


def print_report(report: dict):
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT: Vector-Only vs Hybrid (Vector + Graph)")
    print("=" * 70)

    for mode in ["vector_only", "hybrid"]:
        m = report.get(mode, {})
        label = "Vector-Only RAG" if mode == "vector_only" else "Hybrid RAG (Vector + Graph)"
        print(f"\n{'─' * 40}")
        print(f"  {label}")
        print(f"{'─' * 40}")
        print(f"  Queries:              {m.get('num_queries', 0)}")
        print(f"  Retrieval Recall:     {m.get('avg_retrieval_recall', 0):.3f}")
        print(f"  Retrieval Precision:  {m.get('avg_retrieval_precision', 0):.3f}")
        print(f"  Answer Accuracy:      {m.get('answer_accuracy', 0):.3f}")
        print(f"  Citation Accuracy:    {m.get('citation_accuracy', 0):.3f}")
        print(f"  Hallucination Rate:   {m.get('hallucination_rate', 0):.3f}")
        print(f"  Abstention Rate:      {m.get('abstention_rate', 0):.3f}")
        print(f"  Avg Confidence:       {m.get('avg_confidence', 0):.3f}")
        print(f"  Avg Latency (ms):     {m.get('avg_latency_ms', 0):.0f}")

    # Per-type breakdown
    per_type = report.get("per_query_type", {})
    if per_type:
        print(f"\n{'=' * 70}")
        print("BREAKDOWN BY QUERY TYPE (This is the thesis-critical comparison)")
        print(f"{'=' * 70}")
        for qtype, data in per_type.items():
            imp = data.get("improvement", {})
            print(f"\n  [{qtype}]")
            print(f"    Retrieval Recall: Vector={data['vector_only'].get('avg_retrieval_recall', 0):.3f} → Hybrid={data['hybrid'].get('avg_retrieval_recall', 0):.3f} (Δ={imp.get('retrieval_recall_delta', 0):+.3f})")
            print(f"    Answer Accuracy:  Vector={data['vector_only'].get('answer_accuracy', 0):.3f} → Hybrid={data['hybrid'].get('answer_accuracy', 0):.3f} (Δ={imp.get('answer_accuracy_delta', 0):+.3f})")
            print(f"    Hallucination:    Vector={data['vector_only'].get('hallucination_rate', 0):.3f} → Hybrid={data['hybrid'].get('hallucination_rate', 0):.3f} (Δ={imp.get('hallucination_rate_delta', 0):+.3f})")

    print(f"\n{'=' * 70}")
