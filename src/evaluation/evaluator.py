"""
Evaluation framework  --  v2 (LLM-as-Judge)
Compares: Vector-Only RAG (A) vs Hybrid Vector+Graph (B).

Metrics (industry-standard RAG evaluation axes):
- Retrieval: multi-doc recall, section recall, precision, entity recall
- Answer quality (LLM-judged): faithfulness, correctness, citation verification
- Safety: hallucination detection (LLM-judged)
- System: confidence, latency
"""

import json
import re
import time
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

from openai import OpenAI

from config import get_config, Config, LLMConfig
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalResponse
from src.generation.aggregator import generate_response


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class EvalQuery:
    """A single evaluation query -- supports BOTH dataset schemas.

    Schema A (eval_dataset.json):  source_doc + source_section + expected_answer
    Schema B (eval_multihop.json): expected_docs + reasoning_chain + hops_required
    """
    query_id: str
    question: str
    reasoning_type: str
    language: str = "en"
    # Schema A fields (single-doc / few-doc)
    expected_answer: str = ""
    source_doc: str = ""
    source_section: str = ""
    expected_entities: list[str] = None
    # Schema B fields (multi-hop)
    expected_docs: list[str] = None
    reasoning_chain: str = ""
    hops_required: int = 0

    def __post_init__(self):
        if self.expected_entities is None:
            self.expected_entities = []
        if self.expected_docs is None:
            self.expected_docs = []
        # Normalise: if source_doc is set but expected_docs is empty,
        # populate expected_docs from source_doc for a uniform interface.
        if self.source_doc and not self.expected_docs:
            self.expected_docs = [self.source_doc]


@dataclass
class EvalResult:
    """Evaluation result for a single query."""
    query_id: str
    question: str
    reasoning_type: str
    retrieval_mode: str  # "vector_only" or "hybrid"
    # -- Retrieval metrics --
    retrieval_recall: float          # fraction of expected_docs retrieved
    retrieval_precision: float       # fraction of results from expected docs
    section_recall: float            # 1.0 if section matched, 0.0 otherwise
    entity_recall: float             # fraction of expected_entities found in chunks
    # -- LLM-judged answer metrics (0.0 - 1.0) --
    faithfulness: float              # every claim traceable to context?
    answer_correctness: float        # semantically matches expected answer?
    citation_accuracy: float         # each [N] citation is verifiable?
    hallucination_score: float       # 0 = clean, 1 = fully hallucinated
    # -- Legacy / system --
    answer_generated: str
    evidence_sufficient: bool
    confidence_score: float
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


# ===================================================================
# Dataset I/O
# ===================================================================

def load_eval_dataset(path: str | Path) -> list[EvalQuery]:
    """Load evaluation dataset from JSON -- auto-detects either schema."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries: list[EvalQuery] = []
    for item in data:
        # Only keep fields that EvalQuery.__init__ accepts
        valid_keys = set(EvalQuery.__dataclass_fields__.keys())
        filtered = {k: v for k, v in item.items() if k in valid_keys}
        queries.append(EvalQuery(**filtered))
    return queries


def save_eval_dataset(queries: list[EvalQuery], path: str | Path):
    """Save evaluation dataset to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(q) for q in queries], f, ensure_ascii=False, indent=2)


# ===================================================================
# LLM-as-Judge  (uses the FAST LLM to judge -- cheap + fast)
# ===================================================================

def _judge_call(llm_config: LLMConfig, system: str, user: str) -> str:
    """Single LLM call for judging -- with retry."""
    for attempt in range(3):
        try:
            if llm_config.provider == "ollama_chat":
                import ollama as ollama_client
                resp = ollama_client.chat(
                    model=llm_config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    options={"temperature": 0.0, "num_predict": 256},
                )
                return resp["message"]["content"]
            else:
                client = OpenAI(
                    api_key=llm_config.api_key or "not-needed",
                    base_url=llm_config.base_url,
                )
                resp = client.chat.completions.create(
                    model=llm_config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                return resp.choices[0].message.content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                warnings.warn(f"Judge LLM call failed: {e}")
                return ""


def _parse_score(text: str) -> float:
    """Extract a 0.0-1.0 score from judge output.
    Accepts '0.8', '0.8/1', '8/10', or bare integers 0-10."""
    if not text:
        return 0.0
    text = text.strip()
    # Try "X/10"
    m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
    if m:
        return min(float(m.group(1)) / 10.0, 1.0)
    # Try "X/1"
    m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*1\b', text)
    if m:
        return min(float(m.group(1)), 1.0)
    # Try bare float 0.0-1.0
    m = re.search(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', text)
    if m:
        return float(m.group(1))
    # Try bare int 0-10
    m = re.search(r'\b(\d{1,2})\b', text)
    if m:
        val = int(m.group(1))
        if val <= 10:
            return val / 10.0
    return 0.0


# -- Faithfulness --
_FAITHFULNESS_SYSTEM = (
    "You are an impartial evaluator. Score how faithful an answer is to "
    "the provided context. A faithful answer makes NO claims that are not "
    "directly supported by the context.\n"
    "Respond with ONLY a decimal score between 0.0 and 1.0.\n"
    "1.0 = every claim is supported. 0.0 = entirely unsupported."
)
_FAITHFULNESS_USER = (
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Score (0.0-1.0):"
)


def judge_faithfulness(llm_config: LLMConfig, context: str, answer: str) -> float:
    raw = _judge_call(
        llm_config, _FAITHFULNESS_SYSTEM,
        _FAITHFULNESS_USER.format(context=context[:6000], answer=answer[:2000]),
    )
    return _parse_score(raw)


# -- Answer correctness --
_CORRECTNESS_SYSTEM = (
    "You are an impartial evaluator. Score how semantically correct a generated "
    "answer is compared to the expected (gold) answer. Paraphrasing is fine -- "
    "focus on whether the KEY FACTS match (numbers, entities, thresholds, requirements).\n"
    "Respond with ONLY a decimal score between 0.0 and 1.0.\n"
    "1.0 = semantically identical. 0.0 = completely wrong."
)
_CORRECTNESS_USER = (
    "EXPECTED ANSWER:\n{expected}\n\n"
    "GENERATED ANSWER:\n{generated}\n\n"
    "Score (0.0-1.0):"
)


def judge_correctness(llm_config: LLMConfig, expected: str, generated: str) -> float:
    if not expected:
        return -1.0  # no ground truth -- cannot judge
    raw = _judge_call(
        llm_config, _CORRECTNESS_SYSTEM,
        _CORRECTNESS_USER.format(expected=expected[:2000], generated=generated[:2000]),
    )
    return _parse_score(raw)


# -- Citation verification --
_CITATION_SYSTEM = (
    "You are an impartial evaluator. The answer below contains numbered citations "
    "like [1], [2], etc. For each citation, check whether the referenced evidence "
    "actually supports the claim made.\n"
    "Respond with ONLY a decimal score between 0.0 and 1.0.\n"
    "1.0 = all citations are accurate. 0.0 = all citations are wrong or missing."
)
_CITATION_USER = (
    "EVIDENCE (numbered):\n{evidence}\n\n"
    "ANSWER WITH CITATIONS:\n{answer}\n\n"
    "Score (0.0-1.0):"
)


def judge_citations(llm_config: LLMConfig, evidence: str, answer: str) -> float:
    # If no citations in answer, score = 0
    if not re.search(r'\[\d+\]', answer):
        return 0.0
    raw = _judge_call(
        llm_config, _CITATION_SYSTEM,
        _CITATION_USER.format(evidence=evidence[:6000], answer=answer[:2000]),
    )
    return _parse_score(raw)


# -- Hallucination detection --
_HALLUCINATION_SYSTEM = (
    "You are an impartial evaluator. Determine how much of the answer contains "
    "claims that are NOT present in the provided context. Focus on specific facts: "
    "numbers, percentages, thresholds, entity names, timeframes.\n"
    "Respond with ONLY a decimal score between 0.0 and 1.0.\n"
    "0.0 = no hallucination (all claims grounded). 1.0 = entirely hallucinated."
)
_HALLUCINATION_USER = (
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Hallucination score (0.0-1.0):"
)


def judge_hallucination(llm_config: LLMConfig, context: str, answer: str) -> float:
    raw = _judge_call(
        llm_config, _HALLUCINATION_SYSTEM,
        _HALLUCINATION_USER.format(context=context[:6000], answer=answer[:2000]),
    )
    return _parse_score(raw)


# ===================================================================
# Core evaluation logic
# ===================================================================

def _build_evidence_str(retrieval: RetrievalResponse) -> str:
    """Build a numbered evidence string from retrieval results."""
    parts = []
    for i, r in enumerate(retrieval.results, 1):
        parts.append(f"[{i}] ({r.doc_name} | {' > '.join(r.section_path)})\n{r.content[:500]}")
    return "\n\n".join(parts)


def evaluate_single_query(
    eval_query: EvalQuery,
    retriever: HybridRetriever,
    use_graph: bool = True,
    config: Optional[Config] = None,
    judge_llm: Optional[LLMConfig] = None,
) -> EvalResult:
    """Run a single evaluation query and measure all metrics."""
    config = config or get_config()
    judge_llm = judge_llm or config.fast_llm
    mode = "hybrid" if use_graph else "vector_only"

    start = time.time()

    # -- Retrieve --
    retrieval = retriever.retrieve(query=eval_query.question, use_graph=use_graph)

    # -- Generate --
    response = generate_response(query=eval_query.question, retrieval=retrieval, config=config)
    latency = (time.time() - start) * 1000
    answer_text = response.get("answer", "")

    # -- Retrieval recall (multi-doc aware) --
    expected_docs_lower = [d.lower() for d in eval_query.expected_docs]
    retrieved_docs_lower = [r.doc_name.lower() for r in retrieval.results]

    if expected_docs_lower:
        hits = sum(
            1 for ed in expected_docs_lower
            if any(ed in rd for rd in retrieved_docs_lower)
        )
        retrieval_recall = hits / len(expected_docs_lower)
    else:
        retrieval_recall = 0.0

    # -- Section recall (if source_section provided) --
    section_recall = 0.0
    if eval_query.source_section:
        expected_section = eval_query.source_section.lower()
        section_hit = any(
            expected_section in " > ".join(r.section_path).lower()
            for r in retrieval.results
        )
        section_recall = 1.0 if section_hit else 0.0

    # -- Retrieval precision --
    if retrieval.results and expected_docs_lower:
        relevant_count = sum(
            1 for r in retrieval.results
            if any(ed in r.doc_name.lower() for ed in expected_docs_lower)
        )
        retrieval_precision = relevant_count / len(retrieval.results)
    else:
        retrieval_precision = 0.0

    # -- Entity recall --
    if eval_query.expected_entities:
        all_content = " ".join(r.content.lower() for r in retrieval.results)
        entity_hits = sum(
            1 for ent in eval_query.expected_entities
            if ent.lower() in all_content
        )
        entity_recall = entity_hits / len(eval_query.expected_entities)
    else:
        entity_recall = -1.0  # N/A -- no expected entities provided

    # -- LLM-as-Judge metrics --
    evidence_str = _build_evidence_str(retrieval)

    faithfulness = judge_faithfulness(judge_llm, evidence_str, answer_text)
    correctness = judge_correctness(judge_llm, eval_query.expected_answer, answer_text)
    citation_acc = judge_citations(judge_llm, evidence_str, answer_text)
    hallucination = judge_hallucination(judge_llm, evidence_str, answer_text)

    return EvalResult(
        query_id=eval_query.query_id,
        question=eval_query.question,
        reasoning_type=eval_query.reasoning_type,
        retrieval_mode=mode,
        retrieval_recall=round(retrieval_recall, 4),
        retrieval_precision=round(retrieval_precision, 4),
        section_recall=round(section_recall, 4),
        entity_recall=round(entity_recall, 4),
        faithfulness=round(faithfulness, 4),
        answer_correctness=round(correctness, 4),
        citation_accuracy=round(citation_acc, 4),
        hallucination_score=round(hallucination, 4),
        answer_generated=answer_text[:500],
        evidence_sufficient=retrieval.evidence_sufficient,
        confidence_score=round(retrieval.confidence_score, 4),
        latency_ms=round(latency, 1),
    )


# ===================================================================
# Batch evaluation
# ===================================================================

def run_evaluation(
    eval_queries: list[EvalQuery],
    retriever: HybridRetriever,
    config: Optional[Config] = None,
    save_path: Optional[str | Path] = None,
) -> dict:
    """
    Run full comparative evaluation: vector-only vs hybrid.
    Returns aggregated metrics for both modes.
    """
    config = config or get_config()
    judge_llm = config.fast_llm  # use the cheap model for judging

    results_vector: list[EvalResult] = []
    results_hybrid: list[EvalResult] = []

    total = len(eval_queries)
    for i, eq in enumerate(eval_queries):
        print(f"\n[{i+1}/{total}] {eq.query_id}: {eq.question[:60]}...")

        # Vector-only
        print(f"  -> vector_only ... ", end="", flush=True)
        r_vec = evaluate_single_query(eq, retriever, use_graph=False, config=config, judge_llm=judge_llm)
        print(f"recall={r_vec.retrieval_recall:.2f}  faith={r_vec.faithfulness:.2f}  correct={r_vec.answer_correctness:.2f}")
        results_vector.append(r_vec)

        # Hybrid
        print(f"  -> hybrid      ... ", end="", flush=True)
        r_hyb = evaluate_single_query(eq, retriever, use_graph=True, config=config, judge_llm=judge_llm)
        print(f"recall={r_hyb.retrieval_recall:.2f}  faith={r_hyb.faithfulness:.2f}  correct={r_hyb.answer_correctness:.2f}")
        results_hybrid.append(r_hyb)

    # -- Aggregate --
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_queries": total,
        "judge_model": judge_llm.model,
        "vector_only": _aggregate_results(results_vector),
        "hybrid": _aggregate_results(results_hybrid),
        "per_query_type": _aggregate_by_type(results_vector, results_hybrid),
        "detailed_results": {
            "vector_only": [r.to_dict() for r in results_vector],
            "hybrid": [r.to_dict() for r in results_hybrid],
        },
    }

    # Optionally save to disk
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation report to {save_path}")

    return report


# ===================================================================
# Aggregation helpers
# ===================================================================

def _safe_avg(values: list[float]) -> float:
    """Average ignoring -1 sentinel values (N/A)."""
    valid = [v for v in values if v >= 0]
    return sum(valid) / len(valid) if valid else -1.0


def _aggregate_results(results: list[EvalResult]) -> dict:
    """Aggregate metrics across all results."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "num_queries": n,
        # Retrieval
        "avg_retrieval_recall": round(sum(r.retrieval_recall for r in results) / n, 4),
        "avg_retrieval_precision": round(sum(r.retrieval_precision for r in results) / n, 4),
        "avg_section_recall": round(
            _safe_avg([r.section_recall for r in results if r.section_recall >= 0]), 4
        ),
        "avg_entity_recall": round(
            _safe_avg([r.entity_recall for r in results]), 4
        ),
        # LLM-judged
        "avg_faithfulness": round(sum(r.faithfulness for r in results) / n, 4),
        "avg_answer_correctness": round(
            _safe_avg([r.answer_correctness for r in results]), 4
        ),
        "avg_citation_accuracy": round(sum(r.citation_accuracy for r in results) / n, 4),
        "avg_hallucination_score": round(sum(r.hallucination_score for r in results) / n, 4),
        # System
        "abstention_rate": round(sum(1 for r in results if not r.evidence_sufficient) / n, 4),
        "avg_confidence": round(sum(r.confidence_score for r in results) / n, 4),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
    }


def _aggregate_by_type(
    results_vector: list[EvalResult],
    results_hybrid: list[EvalResult],
) -> dict:
    """Aggregate metrics broken down by reasoning type (thesis-critical)."""
    types = sorted(set(r.reasoning_type for r in results_vector + results_hybrid))
    breakdown = {}

    for rtype in types:
        vec_sub = [r for r in results_vector if r.reasoning_type == rtype]
        hyb_sub = [r for r in results_hybrid if r.reasoning_type == rtype]
        v = _aggregate_results(vec_sub)
        h = _aggregate_results(hyb_sub)

        delta_keys = [
            "avg_retrieval_recall", "avg_faithfulness",
            "avg_answer_correctness", "avg_hallucination_score",
        ]
        improvement = {}
        for k in delta_keys:
            vv = v.get(k, 0) if v.get(k, -1) >= 0 else 0
            hh = h.get(k, 0) if h.get(k, -1) >= 0 else 0
            improvement[f"{k}_delta"] = round(hh - vv, 4)

        breakdown[rtype] = {
            "vector_only": v,
            "hybrid": h,
            "improvement": improvement,
        }

    return breakdown


# ===================================================================
# Pretty-print report
# ===================================================================

def print_report(report: dict):
    """Print a formatted evaluation report to the terminal."""
    print("\n" + "=" * 72)
    print("  EVALUATION REPORT: Vector-Only vs Hybrid (Vector + Graph)")
    print(f"  Judge model: {report.get('judge_model', '?')}")
    print(f"  Queries: {report.get('num_queries', '?')}")
    print("=" * 72)

    for mode in ["vector_only", "hybrid"]:
        m = report.get(mode, {})
        label = "Vector-Only RAG" if mode == "vector_only" else "Hybrid RAG (Vector + Graph)"
        print(f"\n{'-' * 42}")
        print(f"  {label}")
        print(f"{'-' * 42}")
        print(f"  Retrieval Recall:      {m.get('avg_retrieval_recall', 0):.3f}")
        print(f"  Retrieval Precision:   {m.get('avg_retrieval_precision', 0):.3f}")
        _sr = m.get('avg_section_recall', -1)
        if _sr >= 0:
            print(f"  Section Recall:        {_sr:.3f}")
        _er = m.get('avg_entity_recall', -1)
        if _er >= 0:
            print(f"  Entity Recall:         {_er:.3f}")
        print(f"  ----------------------------")
        print(f"  Faithfulness (LLM):    {m.get('avg_faithfulness', 0):.3f}")
        _ac = m.get('avg_answer_correctness', -1)
        if _ac >= 0:
            print(f"  Answer Correct (LLM):  {_ac:.3f}")
        print(f"  Citation Acc (LLM):    {m.get('avg_citation_accuracy', 0):.3f}")
        print(f"  Hallucination (LLM):   {m.get('avg_hallucination_score', 0):.3f}  (lower is better)")
        print(f"  ----------------------------")
        print(f"  Abstention Rate:       {m.get('abstention_rate', 0):.3f}")
        print(f"  Avg Confidence:        {m.get('avg_confidence', 0):.3f}")
        print(f"  Avg Latency (ms):      {m.get('avg_latency_ms', 0):.0f}")

    # -- Per-type breakdown --
    per_type = report.get("per_query_type", {})
    if per_type:
        print(f"\n{'=' * 72}")
        print("  BREAKDOWN BY QUERY TYPE  (thesis-critical comparison)")
        print(f"{'=' * 72}")
        for qtype, data in per_type.items():
            imp = data.get("improvement", {})
            v = data.get("vector_only", {})
            h = data.get("hybrid", {})
            n_v = v.get("num_queries", 0)
            print(f"\n  [{qtype}]  (n={n_v})")
            print(f"    Retrieval Recall:  V={v.get('avg_retrieval_recall',0):.3f} -> H={h.get('avg_retrieval_recall',0):.3f}  (d={imp.get('avg_retrieval_recall_delta',0):+.3f})")
            print(f"    Faithfulness:      V={v.get('avg_faithfulness',0):.3f} -> H={h.get('avg_faithfulness',0):.3f}  (d={imp.get('avg_faithfulness_delta',0):+.3f})")
            ac_v = v.get('avg_answer_correctness', -1)
            ac_h = h.get('avg_answer_correctness', -1)
            if ac_v >= 0 and ac_h >= 0:
                print(f"    Correctness:       V={ac_v:.3f} -> H={ac_h:.3f}  (d={imp.get('avg_answer_correctness_delta',0):+.3f})")
            print(f"    Hallucination:     V={v.get('avg_hallucination_score',0):.3f} -> H={h.get('avg_hallucination_score',0):.3f}  (d={imp.get('avg_hallucination_score_delta',0):+.3f})  (lower better)")

    print(f"\n{'=' * 72}")
