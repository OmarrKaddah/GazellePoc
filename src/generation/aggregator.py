"""
LLM-as-Aggregator response generation.

Key architectural principle:
  The LLM does NOT reason about the answer. The graph traversal and retrieval
  engine have already determined WHAT the answer is (evidence nodes, paths, scores).
  The LLM's only job is to VERBALIZE the pre-assembled evidence into natural language.

This is what makes the system hallucination-resistant:
  - Reasoning = deterministic graph traversal (auditable)
  - Generation = LLM converts structured evidence to prose (constrained)
"""

from typing import Optional

from openai import OpenAI

from config import get_config, Config, LLMConfig
from src.retrieval.hybrid_retriever import RetrievalResponse


# ── System prompts ──

AGGREGATOR_SYSTEM_PROMPT = """You are a banking knowledge assistant. Your role is STRICTLY to aggregate and verbalize the provided evidence into a clear answer.

CRITICAL RULES:
1. ONLY use information from the provided evidence. NEVER add facts not present in the evidence.
2. QUOTE exact values (numbers, percentages, limits) directly from the evidence. Do not paraphrase numeric values.
3. CITE using numbered references like [1], [2], [3] matching the evidence numbers provided. Place the citation IMMEDIATELY after the claim it supports.
4. If the evidence is insufficient to fully answer the question, explicitly state what is missing.
5. If evidence is marked as INSUFFICIENT, respond ONLY with: "I cannot answer this question with sufficient confidence. The available evidence does not adequately cover this topic."
6. Respond in the same language as the user's question (Arabic or English).
7. Structure your response clearly. Do NOT add a separate evidence/sources section at the end — the numbered citations inline are sufficient.

RESPONSE FORMAT:
**Answer:** [Direct answer with inline numbered citations like [1], [2]]

**Confidence:** [High/Medium/Low based on evidence coverage]"""


AGGREGATOR_USER_PROMPT = """Question: {query}

Evidence Sufficiency: {sufficiency}
Confidence Score: {confidence}

Retrieved Evidence (ranked by relevance):
{evidence_block}

Connected Entities from Knowledge Graph:
{entities_block}

Evidence Paths (graph reasoning trails):
{paths_block}

Based ONLY on the above evidence, provide your answer. Quote exact values. Cite sources."""


def _get_llm_client(llm_config: LLMConfig):
    """Create an OpenAI-compatible client."""
    if llm_config.provider == "ollama_chat":
        return None  # Use ollama library directly
    return OpenAI(
        api_key=llm_config.api_key or "not-needed",
        base_url=llm_config.base_url,
    )


def _call_llm(llm_config: LLMConfig, system_msg: str, user_msg: str, stream: bool = False):
    """Call the LLM via appropriate provider."""
    if llm_config.provider == "ollama_chat":
        import ollama as ollama_client
        if stream:
            return ollama_client.chat(
                model=llm_config.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": llm_config.temperature},
                stream=True,
            )
        else:
            response = ollama_client.chat(
                model=llm_config.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": llm_config.temperature},
            )
            return response["message"]["content"]
    else:
        client = OpenAI(
            api_key=llm_config.api_key or "not-needed",
            base_url=llm_config.base_url,
        )
        if stream:
            return client.chat.completions.create(
                model=llm_config.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                stream=True,
            )
        else:
            response = client.chat.completions.create(
                model=llm_config.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
            return response.choices[0].message.content


def _format_evidence_block(retrieval: RetrievalResponse) -> str:
    """Format retrieved chunks into a numbered evidence block."""
    lines = []
    for i, r in enumerate(retrieval.results, 1):
        section = " > ".join(r.section_path) if r.section_path else "root"
        lines.append(
            f"[Evidence {i}] (Score: {r.final_score:.3f}, Type: {r.element_type})\n"
            f"  Source: {r.doc_name} | {section}\n"
            f"  Content: {r.content}\n"
        )
    return "\n".join(lines) if lines else "No evidence retrieved."


def _format_entities_block(retrieval: RetrievalResponse) -> str:
    """Format query entities and connected entities."""
    lines = []
    if retrieval.query_entities:
        lines.append(f"Query entities detected: {', '.join(retrieval.query_entities)}")
    for r in retrieval.results:
        if r.connected_entities:
            lines.append(f"  Chunk {r.chunk_id[:8]}... connected to: {', '.join(r.connected_entities)}")
    return "\n".join(lines) if lines else "No entity connections found."


def _format_paths_block(retrieval: RetrievalResponse) -> str:
    """Format evidence paths from graph traversal."""
    lines = []
    for r in retrieval.results:
        if r.evidence_path:
            path_str = " → ".join(
                f"[{step['from_type']}]{step['from']} --{step['relation']}--> [{step['to_type']}]{step['to']}"
                for step in r.evidence_path
            )
            lines.append(f"  Path: {path_str}")
    return "\n".join(lines) if lines else "No graph evidence paths available."


def generate_response(
    query: str,
    retrieval: RetrievalResponse,
    config: Optional[Config] = None,
    stream: bool = False,
    model_override: Optional[str] = None,
) -> dict:
    """
    Generate a response using the LLM as an aggregator over retrieved evidence.
    
    The LLM receives:
    - Pre-assembled evidence (chunks with scores and citations)
    - Entity connections from the knowledge graph
    - Evidence paths (graph reasoning trails)
    - Evidence sufficiency assessment
    
    The LLM does NOT:
    - Decide what the answer is (retrieval already did that)
    - Add information not in the evidence
    - Reason about relationships (graph already did that)
    
    Returns:
        dict with 'answer', 'citations', 'confidence', 'evidence_sufficient'
    """
    config = config or get_config()
    llm_config = config.main_llm

    # Allow overriding the model (e.g. switching between 70B and 8B in the UI)
    if model_override:
        from dataclasses import replace as dc_replace
        llm_config = dc_replace(llm_config, model=model_override)

    # If evidence is insufficient, return refusal without calling LLM
    if not retrieval.evidence_sufficient:
        return {
            "answer": "I cannot answer this question with sufficient confidence. "
                      "The available evidence does not adequately cover this topic.",
            "citations": [],
            "confidence": retrieval.confidence_score,
            "evidence_sufficient": False,
            "retrieval_results": [r.to_dict() for r in retrieval.results],
        }

    # Assemble the evidence package for the LLM
    evidence_block = _format_evidence_block(retrieval)
    entities_block = _format_entities_block(retrieval)
    paths_block = _format_paths_block(retrieval)

    user_prompt = AGGREGATOR_USER_PROMPT.format(
        query=query,
        sufficiency="SUFFICIENT" if retrieval.evidence_sufficient else "INSUFFICIENT",
        confidence=f"{retrieval.confidence_score:.2f}",
        evidence_block=evidence_block,
        entities_block=entities_block,
        paths_block=paths_block,
    )

    try:
        if stream:
            response_stream = _call_llm(
                llm_config,
                system_msg=AGGREGATOR_SYSTEM_PROMPT,
                user_msg=user_prompt,
                stream=True,
            )
            return {
                "stream": response_stream,
                "citations": _extract_citations(retrieval),
                "confidence": retrieval.confidence_score,
                "evidence_sufficient": True,
            }
        else:
            answer = _call_llm(
                llm_config,
                system_msg=AGGREGATOR_SYSTEM_PROMPT,
                user_msg=user_prompt,
                stream=False,
            )

            return {
                "answer": answer,
                "citations": _extract_citations(retrieval),
                "confidence": retrieval.confidence_score,
                "evidence_sufficient": True,
                "retrieval_results": [r.to_dict() for r in retrieval.results],
            }

    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "citations": [],
            "confidence": 0.0,
            "evidence_sufficient": False,
            "retrieval_results": [],
        }


def _extract_citations(retrieval: RetrievalResponse) -> list[dict]:
    """Extract structured citations from retrieval results."""
    citations = []
    for r in retrieval.results:
        citations.append({
            "doc_name": r.doc_name,
            "section_path": " > ".join(r.section_path) if r.section_path else "root",
            "element_type": r.element_type,
            "relevance_score": round(r.final_score, 4),
        })
    return citations
