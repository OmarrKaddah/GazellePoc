"""Neo4j persistence adapter for the knowledge graph.

This module provides a thin bridge between the existing NetworkX graph
representation and a persistent Neo4j store so migration can happen in stages.
"""

from __future__ import annotations

from typing import Any, Optional

import networkx as nx

from config import Config


class Neo4jGraphStore:
    """Persist and load the knowledge graph from Neo4j."""

    def __init__(self, config: Config):
        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Neo4j backend selected but python package 'neo4j' is not installed. "
                "Install it with: pip install neo4j"
            ) from exc

        gc = config.graph
        self._driver = GraphDatabase.driver(
            gc.neo4j_uri,
            auth=(gc.neo4j_user, gc.neo4j_password),
        )
        self._database = gc.neo4j_database

    def close(self):
        self._driver.close()

    def ensure_schema(self):
        """Create core constraints/indexes used by KG reads/writes."""
        statements = [
            "CREATE CONSTRAINT kg_node_id_unique IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX kg_node_type_idx IF NOT EXISTS FOR (n:KGNode) ON (n.node_type)",
            "CREATE INDEX kg_doc_name_idx IF NOT EXISTS FOR (n:KGNode) ON (n.doc_name)",
            "CREATE INDEX kg_entity_name_idx IF NOT EXISTS FOR (n:KGNode) ON (n.name)",
        ]
        with self._driver.session(database=self._database) as session:
            for stmt in statements:
                session.run(stmt)

    def replace_from_networkx(self, graph: nx.DiGraph, batch_size: int = 1000):
        """Replace KG contents in Neo4j from a NetworkX graph snapshot."""
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (n:KGNode) DETACH DELETE n")

            node_rows = []
            for node_id, data in graph.nodes(data=True):
                row = {
                    "id": str(node_id),
                    "node_type": str(data.get("node_type", "unknown")),
                    "props": self._normalize_props(data),
                }
                node_rows.append(row)

            for i in range(0, len(node_rows), batch_size):
                batch = node_rows[i : i + batch_size]
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (n:KGNode {id: row.id})
                    SET n += row.props
                    SET n.node_type = row.node_type
                    """,
                    rows=batch,
                )

            edge_rows = []
            for source, target, data in graph.edges(data=True):
                props = self._normalize_props(data)
                relation_type = str(props.get("relation_type", "connected"))
                edge_rows.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "relation_type": relation_type,
                        "props": props,
                    }
                )

            for i in range(0, len(edge_rows), batch_size):
                batch = edge_rows[i : i + batch_size]
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (s:KGNode {id: row.source})
                    MATCH (t:KGNode {id: row.target})
                    MERGE (s)-[r:KG_REL {
                      source: row.source,
                      target: row.target,
                      relation_type: row.relation_type
                    }]->(t)
                    SET r += row.props
                    """,
                    rows=batch,
                )

    def load_as_networkx(self) -> nx.DiGraph:
        """Load the full KG from Neo4j into a NetworkX DiGraph."""
        graph = nx.DiGraph()
        with self._driver.session(database=self._database) as session:
            node_result = session.run(
                "MATCH (n:KGNode) RETURN n.id AS id, properties(n) AS props"
            )
            for record in node_result:
                node_id = record["id"]
                props = dict(record["props"] or {})
                graph.add_node(node_id, **props)

            edge_result = session.run(
                """
                MATCH (s:KGNode)-[r:KG_REL]->(t:KGNode)
                RETURN s.id AS source, t.id AS target, properties(r) AS props
                """
            )
            for record in edge_result:
                props = dict(record["props"] or {})
                props.pop("source", None)
                props.pop("target", None)
                graph.add_edge(record["source"], record["target"], **props)

        return graph

    @staticmethod
    def _normalize_props(data: dict[str, Any]) -> dict[str, Any]:
        """Keep only Neo4j-serializable property types."""
        out: dict[str, Any] = {}
        for k, v in data.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, list):
                if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                    out[k] = v
                else:
                    out[k] = [str(x) for x in v]
            else:
                out[k] = str(v)
        return out
