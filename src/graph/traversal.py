"""
Graph traversal for hybrid retrieval.
Seeds traversal from vector-retrieved chunks and query entities,
expands k hops, and scores nodes.
"""

from typing import Optional

from config import get_config, Config
from src.graph.kg_builder import KnowledgeGraph

# Structural edge types to SKIP during semantic BFS traversal.
# Following these floods the traversal with same-document chunks
# (chunk → doc → every other chunk/section in that doc) and adds
# no cross-document signal.
STRUCTURAL_EDGES = frozenset({
    "belongs_to_document",
    "belongs_to_section",
    "child_of_section",
    "table_belongs_to_clause",
})


def _stem(word: str) -> str:
    """Minimal English suffix stripping for better entity matching."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("tion") and len(word) > 5:
        return word[:-4]
    if word.endswith("ment") and len(word) > 5:
        return word[:-4]
    if word.endswith("ness") and len(word) > 5:
        return word[:-4]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def extract_query_entities_simple(query: str, kg: KnowledgeGraph) -> list[str]:
    """
    Entity matching: find KG entity nodes that match terms in the query.
    Uses multi-strategy matching: exact substring, token overlap, and key term matching.
    """
    matches = []
    query_lower = query.lower()
    query_tokens = set(query_lower.split())
    # Remove stop words for better matching
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "shall", "can", "need", "dare", "ought",
                  "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "as", "into", "through", "during", "before", "after", "above", "below",
                  "between", "out", "off", "over", "under", "again", "further", "then",
                  "once", "and", "but", "or", "nor", "not", "so", "yet", "both", "each",
                  "few", "more", "most", "other", "some", "such", "no", "only", "own",
                  "same", "than", "too", "very", "just", "because", "if", "when", "what",
                  "which", "who", "whom", "this", "that", "these", "those", "how", "all",
                  "any", "many", "much"}
    query_keywords = query_tokens - stop_words
    query_stems = {_stem(w) for w in query_keywords}

    for node_id, data in kg.graph.nodes(data=True):
        if data.get("node_type") != "entity":
            continue
        name = data.get("name", "").lower().strip()
        if not name or len(name) < 2:
            continue

        # Strategy 1: Entity name appears as substring in query
        if len(name) > 2 and name in query_lower:
            matches.append(node_id)
            continue

        # Strategy 2: Query keywords match entity name tokens (≥ 50% overlap)
        #             Uses both exact and stemmed forms for robustness
        name_tokens = set(name.split()) - stop_words
        if name_tokens and query_keywords:
            overlap = name_tokens & query_keywords
            # Also try stemmed matching (e.g., "loans" matches "loan")
            name_stems = {_stem(w) for w in name_tokens}
            stem_overlap = name_stems & query_stems
            best_overlap = max(len(overlap), len(stem_overlap))
            if best_overlap >= max(1, len(name_tokens) * 0.5):
                matches.append(node_id)
                continue

        # Strategy 3: Entity type + partial name match for known banking terms
        etype = data.get("entity_type", "")
        matched = False
        if etype in ("organization", "role", "regulation", "product"):
            # Check if any significant word (or its stem) from entity name is in query
            significant_words = name_tokens - stop_words
            for w in significant_words:
                if len(w) > 3 and (w in query_lower or _stem(w) in query_stems):
                    matches.append(node_id)
                    matched = True
                    break
        if matched:
            continue

        # Strategy 4: Check entity mentions / aliases
        for mention in data.get("mentions", []):
            mention_lower = mention.lower().strip()
            if mention_lower != name and len(mention_lower) > 2 and mention_lower in query_lower:
                matches.append(node_id)
                break

    return matches


def graph_retrieve(
    kg: KnowledgeGraph,
    seed_chunk_ids: list[str],
    query_entity_nodes: list[str],
    max_hops: int = 2,
    max_nodes: int = 300,
) -> dict:
    """
    Perform graph-based retrieval starting from seed nodes.
    
    1. Start from vector-retrieved chunk IDs (seed_chunk_ids)
    2. Also start from matched query entities (query_entity_nodes)
    3. Expand k hops
    4. Collect all chunk nodes reachable within the subgraph
    5. Return subgraph + evidence paths
    
    Returns:
        {
            "chunk_ids": list of reachable chunk node IDs,
            "subgraph": {"nodes": [...], "edges": [...]},
            "entity_matches": list of matched entity names,
        }
    """
    all_seed_nodes = set()

    # Add seed chunks that exist in the graph
    for cid in seed_chunk_ids:
        if kg.graph.has_node(cid):
            all_seed_nodes.add(cid)

    # Add query entity nodes
    all_seed_nodes.update(query_entity_nodes)

    if not all_seed_nodes:
        return {"chunk_ids": [], "subgraph": {"nodes": [], "edges": []}, "entity_matches": []}

    # Expand from all seeds
    visited = set()
    frontier = set(all_seed_nodes)
    all_edges = []

    for hop in range(max_hops):
        next_frontier = set()
        for n in frontier:
            if n in visited:
                continue
            visited.add(n)

            # Explore successors — skip structural edges to avoid flooding
            for neighbor in kg.graph.successors(n):
                edge_data = kg.graph.edges[n, neighbor]
                if edge_data.get("relation_type", "") in STRUCTURAL_EDGES:
                    continue
                if neighbor not in visited and len(visited) + len(next_frontier) < max_nodes:
                    next_frontier.add(neighbor)
                    all_edges.append({
                        "source": n,
                        "target": neighbor,
                        "relation": edge_data.get("relation_type", "unknown"),
                    })

            # Explore predecessors — skip structural edges
            for neighbor in kg.graph.predecessors(n):
                edge_data = kg.graph.edges[neighbor, n]
                if edge_data.get("relation_type", "") in STRUCTURAL_EDGES:
                    continue
                if neighbor not in visited and len(visited) + len(next_frontier) < max_nodes:
                    next_frontier.add(neighbor)
                    all_edges.append({
                        "source": neighbor,
                        "target": n,
                        "relation": edge_data.get("relation_type", "unknown"),
                    })

        frontier = next_frontier

    visited.update(frontier)

    # Collect results
    chunk_ids = []
    entity_matches = []
    nodes = []

    for n in visited:
        if not kg.graph.has_node(n):
            continue
        data = dict(kg.graph.nodes[n])
        data["id"] = n
        data.pop("content", None)  # Don't include content in graph results
        nodes.append(data)

        if data.get("node_type") == "chunk":
            chunk_ids.append(n)
        elif data.get("node_type") == "entity" and n in query_entity_nodes:
            entity_matches.append(data.get("name", n))

    return {
        "chunk_ids": chunk_ids,
        "subgraph": {"nodes": nodes, "edges": all_edges},
        "entity_matches": entity_matches,
    }


def compute_graph_score(
    chunk_id: str,
    kg: KnowledgeGraph,
    query_entity_nodes: list[str],
) -> float:
    """
    Compute a graph connectivity score for a chunk.
    
    Score based on:
    - Number of entity connections
    - Whether connected to query entities
    - Cross-document connections (bonus)
    
    Returns a score between 0.0 and 1.0.
    """
    if not kg.graph.has_node(chunk_id):
        return 0.0

    score = 0.0

    # Count entity neighbors
    entity_neighbors = [
        n for n in list(kg.graph.successors(chunk_id)) + list(kg.graph.predecessors(chunk_id))
        if kg.graph.nodes.get(n, {}).get("node_type") == "entity"
    ]
    # More entity connections = more relevant (diminishing returns)
    entity_score = min(len(entity_neighbors) / 5.0, 1.0)
    score += 0.3 * entity_score

    # Connected to query entities? (strong signal)
    # entity_neighbors are already graph node IDs like "entity::abc123"
    entity_neighbor_set = set(entity_neighbors)
    query_entity_connected = bool(entity_neighbor_set & set(query_entity_nodes))
    # Fallback: direct edge check
    if not query_entity_connected:
        for qe in query_entity_nodes:
            if kg.graph.has_edge(chunk_id, qe) or kg.graph.has_edge(qe, chunk_id):
                query_entity_connected = True
                break

    if query_entity_connected:
        score += 0.5

    # Cross-document entity bonus
    for en in entity_neighbors:
        if kg.graph.nodes.get(en, {}).get("cross_document"):
            score += 0.2
            break

    return min(score, 1.0)
