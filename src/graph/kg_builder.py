"""
Knowledge Graph builder using NetworkX.
Builds the dual-layer graph: text-based KG + cross-modal (table) KG.
"""

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import networkx as nx

from config import get_config, Config
from src.ingestion.chunker import Chunk
from src.graph.entity_extractor import Entity, Relation
from src.graph.neo4j_store import Neo4jGraphStore

if TYPE_CHECKING:
    from pyvis.network import Network


class KnowledgeGraph:
    """
    Dual-layer knowledge graph:
    - Text layer: chunks, entities, documents, sections linked by semantic relations
    - Cross-modal layer: table nodes linked to parent clauses / entities
    
    Both layers live in a single NetworkX graph with node/edge type attributes.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.graph = nx.DiGraph()
        self._neo4j: Optional[Neo4jGraphStore] = None
        if self.config.graph.backend == "neo4j":
            self._neo4j = Neo4jGraphStore(self.config)
        self.entity_id_map: dict[str, str] = {}  # original entity_id -> graph node_id
        # Counters for silent-failure visibility
        self._dropped_relations: int = 0
        self._orphan_entities: int = 0
        # Inverted index: stemmed token -> set of entity node IDs
        self._entity_token_index: dict[str, set[str]] = {}
        # Inverted index: full normalized name -> entity node ID
        self._entity_name_index: dict[str, list[str]] = {}

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def add_chunks(self, chunks: list[Chunk]):
        """
        Add chunk nodes and document/section structure.
        This creates the structural backbone of the graph.
        """
        docs_seen = set()
        sections_seen = set()

        for chunk in chunks:
            # Add chunk node
            self.graph.add_node(
                chunk.chunk_id,
                node_type="chunk",
                layer="text" if chunk.element_type == "text" else "cross_modal",
                element_type=chunk.element_type,
                doc_name=chunk.doc_name,
                section_path=chunk.section_path,
                content=chunk.content,
                language=chunk.language,
            )

            # Add document node
            if chunk.doc_name not in docs_seen:
                self.graph.add_node(
                    f"doc::{chunk.doc_name}",
                    node_type="document",
                    layer="text",
                    name=chunk.doc_name,
                )
                docs_seen.add(chunk.doc_name)

            # Link chunk → document
            self.graph.add_edge(
                chunk.chunk_id,
                f"doc::{chunk.doc_name}",
                relation_type="belongs_to_document",
            )

            # Add section nodes and hierarchy
            if chunk.section_path:
                for i, section in enumerate(chunk.section_path):
                    section_id = f"section::{chunk.doc_name}::{' > '.join(chunk.section_path[:i+1])}"
                    if section_id not in sections_seen:
                        self.graph.add_node(
                            section_id,
                            node_type="section",
                            layer="text",
                            name=section,
                            doc_name=chunk.doc_name,
                            depth=i + 1,
                        )
                        sections_seen.add(section_id)

                        # Link section → document
                        self.graph.add_edge(
                            section_id,
                            f"doc::{chunk.doc_name}",
                            relation_type="belongs_to_document",
                        )

                        # Link section → parent section
                        if i > 0:
                            parent_id = f"section::{chunk.doc_name}::{' > '.join(chunk.section_path[:i])}"
                            self.graph.add_edge(
                                section_id,
                                parent_id,
                                relation_type="child_of_section",
                            )

                # Link chunk → its deepest section
                # For table chunks, use specialized edge type to avoid DiGraph overwrite
                deepest_section = f"section::{chunk.doc_name}::{' > '.join(chunk.section_path)}"
                if chunk.element_type == "table":
                    self.graph.add_edge(
                        chunk.chunk_id,
                        deepest_section,
                        relation_type="table_belongs_to_clause",
                        layer="cross_modal",
                    )
                else:
                    self.graph.add_edge(
                        chunk.chunk_id,
                        deepest_section,
                        relation_type="belongs_to_section",
                    )

        print(f"Added {len(chunks)} chunk nodes. Graph: {self.num_nodes} nodes, {self.num_edges} edges")

    def add_entities(
        self,
        entities: list[Entity],
        relations: list[Relation],
        canonical_map: Optional[dict[str, str]] = None,
    ):
        """
        Add entity nodes and relation edges.
        Links entities to their source chunks.
        Applies entity alignment via canonical_map if provided.
        """
        canonical_map = canonical_map or {}

        for entity in entities:
            canonical_id = canonical_map.get(entity.entity_id, entity.entity_id)
            node_id = f"entity::{canonical_id}"
            self.entity_id_map[entity.entity_id] = node_id

            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    node_type="entity",
                    layer="text",
                    entity_type=entity.entity_type,
                    name=entity.name,
                    value=entity.value,
                    mentions=entity.mentions,
                    source_doc=entity.source_doc,
                )
            else:
                # Merge mentions
                existing = self.graph.nodes[node_id].get("mentions", [])
                if entity.name not in existing:
                    existing.append(entity.name)
                    self.graph.nodes[node_id]["mentions"] = existing

            # Link entity → source chunk
            if entity.source_chunk_id and self.graph.has_node(entity.source_chunk_id):
                self.graph.add_edge(
                    node_id,
                    entity.source_chunk_id,
                    relation_type="mentioned_in",
                )
                self.graph.add_edge(
                    entity.source_chunk_id,
                    node_id,
                    relation_type="mentions_entity",
                )
            elif entity.source_chunk_id:
                # Entity's source chunk isn't in the graph — orphan node
                self._orphan_entities += 1

        # Add explicit relations between entities
        for rel in relations:
            source_node = self.entity_id_map.get(rel.source_id)
            target_node = self.entity_id_map.get(rel.target_id)
            if source_node and target_node:
                self.graph.add_edge(
                    source_node,
                    target_node,
                    relation_type=rel.relation_type,
                )
            else:
                self._dropped_relations += 1

        if self._dropped_relations:
            print(f"  ⚠ {self._dropped_relations} relations dropped (missing source/target entity)")
        if self._orphan_entities:
            print(f"  ⚠ {self._orphan_entities} orphan entities (source chunk not in graph)")
        print(f"Added entities and relations. Graph: {self.num_nodes} nodes, {self.num_edges} edges")

    def add_cross_document_links(self):
        """
        Link entities that appear across multiple documents.
        If the same canonical entity is mentioned in chunks from different docs,
        create cross-document edges.
        """
        entity_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "entity"
        ]

        cross_links = 0
        for entity_node in entity_nodes:
            # Find all chunks this entity is mentioned in (both edge directions)
            chunk_neighbors = [
                n for n in set(self.graph.successors(entity_node)) | set(self.graph.predecessors(entity_node))
                if self.graph.nodes.get(n, {}).get("node_type") == "chunk"
            ]

            # Get unique documents
            docs = set()
            for cn in chunk_neighbors:
                docs.add(self.graph.nodes[cn].get("doc_name"))

            if len(docs) > 1:
                # This entity links multiple documents — mark it
                self.graph.nodes[entity_node]["cross_document"] = True
                cross_links += 1

        print(f"Found {cross_links} cross-document entity links")

    def get_neighbors(self, node_id: str, max_hops: int = 2, max_nodes: int = 50) -> dict:
        """
        Get subgraph around a node up to max_hops.
        Returns nodes and edges in the subgraph.
        """
        if not self.graph.has_node(node_id):
            return {"nodes": [], "edges": []}

        visited = set()
        frontier = {node_id}
        all_edges = []

        for hop in range(max_hops):
            next_frontier = set()
            for n in frontier:
                if n in visited:
                    continue
                visited.add(n)
                for neighbor in self.graph.successors(n):
                    if neighbor not in visited and len(visited) + len(next_frontier) < max_nodes:
                        next_frontier.add(neighbor)
                        edge_data = self.graph.edges[n, neighbor]
                        all_edges.append({
                            "source": n,
                            "target": neighbor,
                            "relation": edge_data.get("relation_type", "unknown"),
                        })
                for neighbor in self.graph.predecessors(n):
                    if neighbor not in visited and len(visited) + len(next_frontier) < max_nodes:
                        next_frontier.add(neighbor)
                        edge_data = self.graph.edges[neighbor, n]
                        all_edges.append({
                            "source": neighbor,
                            "target": n,
                            "relation": edge_data.get("relation_type", "unknown"),
                        })
            frontier = next_frontier

        # Include final frontier nodes
        visited.update(frontier)

        nodes = []
        for n in visited:
            if self.graph.has_node(n):
                data = dict(self.graph.nodes[n])
                data["id"] = n
                # Don't include full content in graph traversal results (too large)
                data.pop("content", None)
                nodes.append(data)

        return {"nodes": nodes, "edges": all_edges}

    def find_entity_nodes(self, entity_name: str) -> list[str]:
        """Find entity nodes by name (case-insensitive substring match)."""
        matches = []
        query = entity_name.lower().strip()
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "entity":
                continue
            name = data.get("name", "").lower()
            mentions = [m.lower() for m in data.get("mentions", [])]
            if query in name or any(query in m for m in mentions):
                matches.append(node_id)
        return matches

    def get_chunk_nodes_from_subgraph(self, subgraph: dict) -> list[str]:
        """Extract chunk node IDs from a subgraph result."""
        return [
            n["id"] for n in subgraph.get("nodes", [])
            if n.get("node_type") == "chunk"
        ]

    def get_evidence_path(self, start_node: str, end_node: str) -> list[dict]:
        """
        Find the shortest path between two nodes and return the evidence trail.
        This is the auditable reasoning path.
        """
        try:
            path = nx.shortest_path(self.graph, start_node, end_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        evidence = []
        for i in range(len(path) - 1):
            edge_data = self.graph.edges.get((path[i], path[i + 1]), {})
            evidence.append({
                "from": path[i],
                "to": path[i + 1],
                "relation": edge_data.get("relation_type", "connected"),
                "from_type": self.graph.nodes[path[i]].get("node_type", "unknown"),
                "to_type": self.graph.nodes[path[i + 1]].get("node_type", "unknown"),
            })

        return evidence

    def save(self, path: str | Path):
        """Save graph to JSON and optionally sync to Neo4j backend."""
        if self._neo4j is not None:
            self._neo4j.replace_from_networkx(self.graph)
            print("Synced graph to Neo4j")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved graph to {path}")

    def load(self, path: str | Path):
        """Load graph from Neo4j backend when configured, else JSON."""
        if self._neo4j is not None:
            self.graph = self._neo4j.load_as_networkx()
        else:
            path = Path(path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, directed=True)
        self.build_entity_index()
        print(f"Loaded graph: {self.num_nodes} nodes, {self.num_edges} edges")

    def build_entity_index(self):
        """
        Build inverted indexes for fast entity matching at query time.
        Called after load() or after add_entities().
        """
        self._entity_token_index.clear()
        self._entity_name_index.clear()

        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "entity":
                continue
            name = data.get("name", "").lower().strip()
            if not name or len(name) < 2:
                continue

            # Full name index
            self._entity_name_index.setdefault(name, []).append(node_id)

            # Token-level index (stemmed)
            tokens = set(name.split())
            for token in tokens:
                if len(token) > 1:
                    stem = self._stem_token(token)
                    self._entity_token_index.setdefault(stem, set()).add(node_id)

            # Also index mentions
            for mention in data.get("mentions", []):
                ml = mention.lower().strip()
                if ml and ml != name:
                    self._entity_name_index.setdefault(ml, []).append(node_id)
                    for t in ml.split():
                        if len(t) > 1:
                            self._entity_token_index.setdefault(self._stem_token(t), set()).add(node_id)

    @staticmethod
    def _stem_token(word: str) -> str:
        """Minimal suffix stripping for index building."""
        if len(word) <= 3:
            return word
        for suffix in ("ies", "ing", "tion", "ment", "ness", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)] if suffix != "ies" else word[:-3] + "y"
        return word

    def get_stats(self) -> dict:
        """Get graph statistics."""
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("relation_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "total_nodes": self.num_nodes,
            "total_edges": self.num_edges,
            "node_types": node_types,
            "edge_types": edge_types,
            "dropped_relations": self._dropped_relations,
            "orphan_entities": self._orphan_entities,
        }

    # ── Visualization ──────────────────────────────────────────────

    # Color + shape per node_type for PyVis
    _VIS_STYLE: dict[str, dict] = {
        "entity":   {"color": "#E74C3C", "shape": "dot",      "size": 18},
        "chunk":    {"color": "#3498DB", "shape": "box",       "size": 12},
        "document": {"color": "#2ECC71", "shape": "diamond",   "size": 25},
        "section":  {"color": "#F39C12", "shape": "triangle",  "size": 14},
    }

    def to_pyvis(
        self,
        *,
        filter_node_types: list[str] | None = None,
        max_nodes: int = 500,
        height: str = "750px",
        width: str = "100%",
        notebook: bool = False,
    ) -> "Network":
        """
        Convert the NetworkX KG to an interactive PyVis network.

        Args:
            filter_node_types: show only these types (None = all).
            max_nodes: cap so the browser doesn't freeze.
            height / width: HTML dimensions.
            notebook: True when running inside Jupyter.

        Returns:
            A pyvis.network.Network ready to .show() or .generate_html().
        """
        from pyvis.network import Network

        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=notebook,
            bgcolor="#1a1a2e",
            font_color="#ffffff",
        )

        # Physics for a nice layout
        net.barnes_hut(
            gravity=-4000,
            central_gravity=0.3,
            spring_length=120,
            spring_strength=0.04,
            damping=0.09,
        )

        allowed = set(filter_node_types) if filter_node_types else None
        added_nodes: set[str] = set()

        # ── Nodes ──
        for node_id, data in self.graph.nodes(data=True):
            ntype = data.get("node_type", "unknown")
            if allowed and ntype not in allowed:
                continue
            if len(added_nodes) >= max_nodes:
                break

            style = self._VIS_STYLE.get(ntype, {"color": "#95A5A6", "shape": "dot", "size": 10})

            # Build a readable label (short) and hover title (detailed)
            if ntype == "entity":
                label = data.get("name", node_id)[:40]
                etype = data.get("entity_type", "")
                value = data.get("value") or ""
                cross = " ⬡ cross-doc" if data.get("cross_document") else ""
                title = (
                    f"<b>{data.get('name', node_id)}</b><br>"
                    f"Type: {etype}<br>"
                    f"{'Value: ' + str(value) + '<br>' if value else ''}"
                    f"Source: {data.get('source_doc', '?')}{cross}"
                )
            elif ntype == "chunk":
                label = f"📄 {data.get('doc_name', '?')[:18]}"
                content_preview = (data.get("content") or "")[:200].replace("\n", " ")
                section = " > ".join(data.get("section_path") or []) or "root"
                title = (
                    f"<b>Chunk</b><br>"
                    f"Doc: {data.get('doc_name', '?')}<br>"
                    f"Section: {section}<br>"
                    f"Type: {data.get('element_type', '?')}<br>"
                    f"<hr>{content_preview}…"
                )
            elif ntype == "document":
                label = f"📘 {data.get('name', node_id)[:25]}"
                title = f"<b>Document</b><br>{data.get('name', node_id)}"
            elif ntype == "section":
                label = f"§ {data.get('name', node_id)[:30]}"
                title = (
                    f"<b>Section</b><br>"
                    f"{data.get('name', node_id)}<br>"
                    f"Doc: {data.get('doc_name', '?')}<br>"
                    f"Depth: {data.get('depth', '?')}"
                )
            else:
                label = node_id[:30]
                title = node_id

            net.add_node(
                node_id,
                label=label,
                title=title,
                color=style["color"],
                shape=style["shape"],
                size=style["size"],
            )
            added_nodes.add(node_id)

        # ── Edges ──
        for src, dst, data in self.graph.edges(data=True):
            if src not in added_nodes or dst not in added_nodes:
                continue
            rtype = data.get("relation_type", "")
            # Color structural edges grey, semantic edges white
            is_structural = rtype in (
                "belongs_to_document", "belongs_to_section",
                "child_of_section", "table_belongs_to_clause",
            )
            net.add_edge(
                src,
                dst,
                title=rtype,
                label=rtype if not is_structural else "",
                color="#555555" if is_structural else "#cccccc",
                width=1 if is_structural else 2,
                arrows="to",
            )

        return net
