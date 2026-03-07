"""
Hybrid retrieval engine.
Combines vector similarity search + graph traversal + policy filtering.
Implements the scoring formula: Final = α * vector + β * graph + γ * policy
"""

from typing import Optional
from dataclasses import dataclass, field
import numpy as np

from config import get_config, Config
from src.ingestion.embedder import VectorStore
from src.graph.kg_builder import KnowledgeGraph
from src.graph.traversal import (
    extract_query_entities_simple,
    graph_retrieve,
    compute_graph_score,
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


@dataclass
class RetrievalResult:
    """A single retrieved item with scoring breakdown."""
    chunk_id: str
    content: str
    doc_name: str
    section_path: list[str]
    element_type: str
    # Scores
    vector_score: float = 0.0       # Embedding similarity
    graph_score: float = 0.0        # Graph connectivity
    policy_score: float = 1.0       # Policy compliance (1.0 = no restrictions)
    final_score: float = 0.0        # Weighted combination
    # Evidence
    evidence_path: list[dict] = field(default_factory=list)
    connected_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "section_path": self.section_path,
            "element_type": self.element_type,
            "content": self.content,
            "vector_score": round(self.vector_score, 4),
            "graph_score": round(self.graph_score, 4),
            "policy_score": round(self.policy_score, 4),
            "final_score": round(self.final_score, 4),
            "connected_entities": self.connected_entities,
            "evidence_path": self.evidence_path,
        }


@dataclass
class RetrievalResponse:
    """Complete retrieval response with metadata."""
    results: list[RetrievalResult]
    query: str
    query_entities: list[str]
    evidence_sufficient: bool
    confidence_score: float
    subgraph: Optional[dict] = None  # Graph neighborhood for visualization

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "query_entities": self.query_entities,
            "evidence_sufficient": self.evidence_sufficient,
            "confidence_score": round(self.confidence_score, 4),
            "num_results": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }


class HybridRetriever:
    """
    Orchestrates the full hybrid retrieval pipeline:
    1. Vector similarity search (Qdrant)
    2. Query entity extraction + graph seeding
    3. Graph traversal expansion
    4. Policy filtering
    5. Hybrid scoring: α * vector + β * graph + γ * policy //// Removed later wtf are these weights
    6. Evidence sufficiency check
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        config: Optional[Config] = None,
    ):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.config = config or get_config()
        self.rc = self.config.retrieval

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_graph: bool = True,
        user_access_level: Optional[int] = None,
        doc_filter: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        Execute the full hybrid retrieval pipeline.
        
        Args:
            query: User's question
            top_k: Number of final results (default from config)
            use_graph: Whether to use graph expansion (set False for baseline comparison)
            user_access_level: Integer access level (1=Public,2=Confidential,3=Restricted).
                               Only chunks with access_level <= this value are returned.
                               None = no filtering (backwards-compatible).
            doc_filter: Filter to specific document
        """
        top_k = top_k or self.rc.rerank_top_k

        # ── Step 1: Vector similarity search ──
        vector_results = self.vector_store.search(
            query=query,
            top_k=self.rc.top_k,
            doc_filter=doc_filter,
            max_access_level=user_access_level,
        )

        # Build initial result objects from vector search
        results_map: dict[str, RetrievalResult] = {}
        for vr in vector_results:
            cid = vr["chunk_id"]
            results_map[cid] = RetrievalResult(
                chunk_id=cid,
                content=vr["content"],
                doc_name=vr["doc_name"],
                section_path=vr.get("section_path", []),
                element_type=vr.get("element_type", "text"),
                vector_score=vr["score"],
            )

        # ── Step 2 & 3: Graph expansion ──
        query_entity_nodes = []
        subgraph = None

        if use_graph and self.kg and self.kg.num_nodes > 0:
            # Extract entities from query
            query_entity_nodes = extract_query_entities_simple(query, self.kg)

            # Seed graph traversal from retrieved chunks + query entities
            seed_chunk_ids = list(results_map.keys())
            graph_result = graph_retrieve(
                kg=self.kg,
                seed_chunk_ids=seed_chunk_ids,
                query_entity_nodes=query_entity_nodes,
                max_hops=self.config.graph.max_hops,
            )

            subgraph = graph_result["subgraph"]

            # Add graph-discovered chunks that vector search missed
            # Use cached embeddings for fair similarity scoring (no extra Ollama calls)
            query_embedding = None  # Lazy: only compute if needed

            for graph_chunk_id in graph_result["chunk_ids"]:
                if graph_chunk_id not in results_map:
                    # Get chunk content from graph
                    node_data = self.kg.graph.nodes.get(graph_chunk_id, {})
                    content = node_data.get("content", "")

                    # RBAC: skip graph-discovered chunks the user cannot access
                    # Uses authoritative DOCUMENT_ACCESS_MAP, not stored metadata
                    if user_access_level is not None:
                        from src.auth.rbac import get_access_level as _get_al
                        doc_name = node_data.get("doc_name", "")
                        if _get_al(doc_name) > user_access_level:
                            continue

                    if content:
                        # Use cached embedding for fast similarity
                        cached_emb = self.vector_store.get_cached_embedding(graph_chunk_id)
                        if cached_emb is not None:
                            if query_embedding is None:
                                query_embedding = self.vector_store._embedder.embed(query)
                            real_vector_score = max(0.0, _cosine_similarity(query_embedding, cached_emb))
                        else:
                            real_vector_score = 0.0

                        results_map[graph_chunk_id] = RetrievalResult(
                            chunk_id=graph_chunk_id,
                            content=content,
                            doc_name=node_data.get("doc_name", ""),
                            section_path=node_data.get("section_path", []),
                            element_type=node_data.get("element_type", "text"),
                            vector_score=real_vector_score,
                        )

            # Compute graph scores for all results
            for cid, result in results_map.items():
                result.graph_score = compute_graph_score(
                    cid, self.kg, query_entity_nodes
                )

                # Find connected entities for citation
                if self.kg.graph.has_node(cid):
                    entity_neighbors = [
                        self.kg.graph.nodes[n].get("name", "")
                        for n in list(self.kg.graph.successors(cid)) + list(self.kg.graph.predecessors(cid))
                        if self.kg.graph.nodes.get(n, {}).get("node_type") == "entity"
                    ]
                    result.connected_entities = entity_neighbors[:5]

        # ── Step 3b: Wire evidence paths ──
        # For each result with connected entities, find shortest graph path
        # from the chunk to its connected entity nodes (auditable reasoning).
        if self.kg and use_graph:
            for cid, result in results_map.items():
                if result.connected_entities and self.kg.graph.has_node(cid):
                    paths: list[dict] = []
                    for entity_name in result.connected_entities:
                        # Resolve entity name → node id (entities are keyed by name)
                        entity_node = None
                        for neighbor in list(self.kg.graph.successors(cid)) + list(self.kg.graph.predecessors(cid)):
                            nd = self.kg.graph.nodes.get(neighbor, {})
                            if nd.get("node_type") == "entity" and nd.get("name") == entity_name:
                                entity_node = neighbor
                                break
                        if entity_node:
                            trail = self.kg.get_evidence_path(cid, entity_node)
                            paths.extend(trail)
                    result.evidence_path = paths

        # ── Step 4: RBAC hard gate ──
        # Authoritative access check using DOCUMENT_ACCESS_MAP.
        # This is the single enforcement point — any chunk whose document
        # requires a higher access level than the user's is REMOVED entirely,
        # regardless of how it was discovered (vector search or graph traversal).
        if user_access_level is not None:
            from src.auth.rbac import get_access_level
            inaccessible = [
                cid for cid, r in results_map.items()
                if get_access_level(r.doc_name) > user_access_level
            ]
            for cid in inaccessible:
                del results_map[cid]

            # Set policy_score for surviving results
            for cid, result in results_map.items():
                result.policy_score = 1.0  # confirmed accessible

        # ── Step 5: Compute final scores ──
        alpha = self.rc.alpha
        beta = self.rc.beta if use_graph else 0.0
        gamma = self.rc.gamma

        # Renormalize weights if graph is disabled
        if not use_graph:
            total = alpha + gamma
            if total > 0:
                alpha = alpha / total
                gamma = gamma / total

        for result in results_map.values():
            result.final_score = (
                alpha * result.vector_score
                + beta * result.graph_score
                + gamma * result.policy_score
            )

        # Sort by final score, take top_k
        sorted_results = sorted(results_map.values(), key=lambda r: r.final_score, reverse=True)[:top_k]

        # ── Step 6: Evidence sufficiency check ──
        confidence = self._compute_confidence(sorted_results)
        evidence_sufficient = (
            confidence >= self.rc.evidence_threshold
            and len(sorted_results) >= self.rc.min_evidence_nodes
        )

        query_entities_names = [
            self.kg.graph.nodes[n].get("name", n) for n in query_entity_nodes
        ] if self.kg and query_entity_nodes else []

        return RetrievalResponse(
            results=sorted_results,
            query=query,
            query_entities=query_entities_names,
            evidence_sufficient=evidence_sufficient,
            confidence_score=confidence,
            subgraph=subgraph,
        )

    def _compute_confidence(self, results: list[RetrievalResult]) -> float:
        """
        Compute overall retrieval confidence.
        Based on: top scores, score distribution, and evidence diversity.
        """
        if not results:
            return 0.0

        # Average of top-3 vector scores (main signal)
        top_vector_scores = sorted([r.vector_score for r in results], reverse=True)[:3]
        avg_top_score = sum(top_vector_scores) / len(top_vector_scores) if top_vector_scores else 0.0

        # Diversity bonus: how many different documents contribute
        unique_docs = len(set(r.doc_name for r in results))
        diversity_bonus = 0.2 * min(unique_docs / 3.0, 1.0)  # Scales: 1 doc=0.067, 2=0.133, 3+=0.2

        return min(avg_top_score + diversity_bonus, 1.0)
