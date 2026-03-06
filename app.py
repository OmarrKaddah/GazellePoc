"""
Streamlit Chat UI for the Banking Knowledge AI System.
Displays: answers, citations, evidence paths, graph visualization, confidence scores.
"""

import json
import os
import re
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

# Load .env before importing config
env_path = Path(".env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from config import get_config
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import VectorStore
from src.graph.kg_builder import KnowledgeGraph
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.aggregator import generate_response


# ── Page config ──
st.set_page_config(
    page_title="Banking Knowledge AI",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def init_system():
    """Initialize the full pipeline (cached across reruns)."""
    config = get_config()

    # Check for pre-built index
    graph_path = Path(config.graph_dir) / "knowledge_graph.json"
    chunks_path = Path(config.parsed_dir) / "chunks.json"

    if not (graph_path.exists() and chunks_path.exists()):
        return None, None, None

    # Load graph
    kg = KnowledgeGraph(config)
    kg.load(graph_path)
    print(f"Loaded graph: {kg.get_stats()['total_nodes']} nodes, {kg.get_stats()['total_edges']} edges")

    # Load chunks and embed
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [Chunk(**c) for c in chunks_data]

    vector_store = VectorStore(config)
    vector_store.index_chunks(chunks)

    retriever = HybridRetriever(vector_store, kg, config)
    return retriever, kg, config


def _linkify_citations(text: str, msg_idx: int) -> str:
    """Replace [1], [2], etc. in answer text with clickable anchor links."""
    def _replace_citation(match):
        num = match.group(1)
        return f'<a href="#cite-{msg_idx}-{num}" style="color: #4A90D9; text-decoration: none; font-weight: bold;">[{num}]</a>'
    return re.sub(r'\[(\d+)\]', _replace_citation, text)


def _render_citations(citations: list[dict], msg_idx: int):
    """Render citations with anchor targets so inline links can scroll to them."""
    for i, c in enumerate(citations, 1):
        emoji = "📊" if c.get("element_type") == "table" else "📄"
        st.markdown(
            f'<div id="cite-{msg_idx}-{i}" style="padding: 4px 0;">'
            f'{emoji} <b>[{i}]</b> {c["doc_name"]} | {c["section_path"]} '
            f'(score: {c["relevance_score"]:.3f})</div>',
            unsafe_allow_html=True,
        )


# ── Graph Explorer ──────────────────────────────────────────────

def _render_graph_explorer(kg: KnowledgeGraph):
    """Full-page interactive PyVis graph explorer."""
    st.header("🌐 Knowledge Graph Explorer")

    stats = kg.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", stats["total_nodes"])
    c2.metric("Edges", stats["total_edges"])
    c3.metric("Entities", stats["node_types"].get("entity", 0))
    c4.metric("Cross-doc entities", sum(
        1 for _, d in kg.graph.nodes(data=True)
        if d.get("cross_document")
    ))

    st.divider()

    # ── Filters ──
    col_f1, col_f2, col_f3 = st.columns(3)

    all_node_types = sorted(stats["node_types"].keys())
    with col_f1:
        selected_types = st.multiselect(
            "Node types to show",
            all_node_types,
            default=["entity", "chunk"],
            help="Showing all types on a large graph can be slow",
        )

    with col_f2:
        max_nodes = st.slider("Max nodes to render", 50, 2000, 300, step=50)

    with col_f3:
        # Optional: focus on a specific document
        doc_nodes = [
            d.get("name", n)
            for n, d in kg.graph.nodes(data=True)
            if d.get("node_type") == "document"
        ]
        focus_doc = st.selectbox("Focus on document (optional)", ["All"] + sorted(doc_nodes))

    # ── Build filtered subgraph ──
    if not selected_types:
        st.warning("Select at least one node type.")
        return

    # If focusing on a specific doc, restrict to that doc's subgraph
    filter_types = selected_types if selected_types else None

    if focus_doc != "All":
        # Get nodes belonging to this document
        doc_node_id = f"doc::{focus_doc}"
        focused_nodes = set()
        for n, d in kg.graph.nodes(data=True):
            ntype = d.get("node_type", "")
            if ntype not in (selected_types or all_node_types):
                continue
            if ntype == "document" and d.get("name") == focus_doc:
                focused_nodes.add(n)
            elif d.get("doc_name") == focus_doc:
                focused_nodes.add(n)
            elif d.get("source_doc") == focus_doc:
                focused_nodes.add(n)
            elif ntype == "entity":
                # Include entities connected to chunks from this doc
                for neighbor in list(kg.graph.successors(n)) + list(kg.graph.predecessors(n)):
                    nd = kg.graph.nodes.get(neighbor, {})
                    if nd.get("doc_name") == focus_doc:
                        focused_nodes.add(n)
                        break

        # Build a temporary subgraph
        sub = kg.graph.subgraph(focused_nodes).copy()
        temp_kg = KnowledgeGraph.__new__(KnowledgeGraph)
        temp_kg.config = kg.config
        temp_kg.graph = sub
        temp_kg.entity_id_map = {}
        temp_kg._VIS_STYLE = kg._VIS_STYLE
        net = temp_kg.to_pyvis(filter_node_types=filter_types, max_nodes=max_nodes)
    else:
        net = kg.to_pyvis(filter_node_types=filter_types, max_nodes=max_nodes)

    # ── Legend ──
    st.markdown(
        """
        **Legend:**
        🔴 **Entity** (dot) &nbsp;|&nbsp;
        🔵 **Chunk** (box) &nbsp;|&nbsp;
        🟢 **Document** (diamond) &nbsp;|&nbsp;
        🟠 **Section** (triangle)
        &nbsp;&nbsp;•&nbsp; Grey edges = structural &nbsp;|&nbsp; White edges = semantic relations
        &nbsp;&nbsp;•&nbsp; **Hover** any node or edge for details &nbsp;|&nbsp; **Scroll** to zoom &nbsp;|&nbsp; **Drag** to rearrange
        """,
        unsafe_allow_html=True,
    )

    # ── Render ──
    with st.spinner("Rendering graph..."):
        html = net.generate_html()
        # PyVis generates full HTML page; embed it via an iframe component
        components.html(html, height=780, scrolling=True)

    # ── Edge type breakdown ──
    with st.expander("📊 Edge type breakdown"):
        st.json(stats["edge_types"])


def main():
    st.title("🏦 Banking Knowledge AI")
    st.caption("Graph-grounded, hallucination-resistant knowledge assistant")

    # ── Initialize system ──
    result = init_system()
    if result[0] is None:
        st.error("System not initialized. Run `python build_index.py` first.")
        st.stop()
    retriever, kg, config = result

    # ── Top-level page selector ──
    page = st.sidebar.radio("Page", ["💬 Chat", "🌐 Graph Explorer"], index=0)

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Retrieval Mode",
            ["🔗 Hybrid (Vector + Graph)", "📊 Vector-Only", "⚔️ A/B Comparison"],
            index=0,
            help="Compare both modes side-by-side to see the graph layer's impact",
        )
        use_graph = mode == "🔗 Hybrid (Vector + Graph)"
        ab_mode = mode == "⚔️ A/B Comparison"

        top_k = st.slider("Results to retrieve", 2, 15, 5)

        llm_model = st.selectbox(
            "Answer Generation Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0,
            help="8B model is more retrieval-dependent — graph advantage becomes more visible",
        )

        show_evidence = st.toggle("Show Evidence Details", value=True)
        show_graph_info = st.toggle("Show Graph Info", value=False)

        st.divider()
        st.header("📊 System Info")

        if kg:
            stats = kg.get_stats()
            st.metric("Graph Nodes", stats["total_nodes"])
            st.metric("Graph Edges", stats["total_edges"])
            if stats.get("node_types"):
                st.json(stats["node_types"])

        mode_label = "🔗 Hybrid (Vector + Graph)" if use_graph else "📊 Vector-Only"
        st.info(f"Mode: {mode_label}")

    # ────────────────────────────────────────────────────────────
    # PAGE: Graph Explorer
    # ────────────────────────────────────────────────────────────
    if page == "🌐 Graph Explorer":
        _render_graph_explorer(kg)
        return

    # ────────────────────────────────────────────────────────────
    # PAGE: Chat
    # ────────────────────────────────────────────────────────────

    # ── Chat interface ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("citations"):
                st.markdown(_linkify_citations(msg["content"], msg_idx), unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📎 Citations", expanded=True):
                    _render_citations(msg["citations"], msg_idx)
            if msg.get("evidence_details"):
                with st.expander("🔍 Evidence Details"):
                    st.json(msg["evidence_details"])

    # Chat input
    if prompt := st.chat_input("Ask a question about banking policies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if ab_mode:
            # ── A/B Comparison Mode ──
            col_a, col_b = st.columns(2)

            with st.spinner("Running A/B comparison..."):
                retrieval_vector = retriever.retrieve(query=prompt, top_k=top_k, use_graph=False)
                retrieval_hybrid = retriever.retrieve(query=prompt, top_k=top_k, use_graph=True)
                resp_vector = generate_response(query=prompt, retrieval=retrieval_vector, config=config, model_override=llm_model)
                resp_hybrid = generate_response(query=prompt, retrieval=retrieval_hybrid, config=config, model_override=llm_model)

            with col_a:
                st.subheader("📊 Vector-Only")
                st.markdown(resp_vector.get("answer", "No response."))
                conf_a = resp_vector.get("confidence", 0)
                st.caption(f"Confidence: {conf_a:.2f}")

                with st.expander("Score Breakdown", expanded=True):
                    for i, r in enumerate(retrieval_vector.results, 1):
                        section = " > ".join(r.section_path) if r.section_path else "root"
                        st.markdown(
                            f"**[{i}]** {r.doc_name} | {section}\n\n"
                            f"  `vector={r.vector_score:.3f}` · "
                            f"`graph={r.graph_score:.3f}` · "
                            f"`policy={r.policy_score:.3f}` → "
                            f"**final={r.final_score:.3f}**"
                        )
                        if r.connected_entities:
                            st.caption(f"  Entities: {', '.join(r.connected_entities)}")

            with col_b:
                st.subheader("🔗 Hybrid (Vector + Graph)")
                st.markdown(resp_hybrid.get("answer", "No response."))
                conf_b = resp_hybrid.get("confidence", 0)
                st.caption(f"Confidence: {conf_b:.2f}")

                with st.expander("Score Breakdown", expanded=True):
                    for i, r in enumerate(retrieval_hybrid.results, 1):
                        section = " > ".join(r.section_path) if r.section_path else "root"
                        st.markdown(
                            f"**[{i}]** {r.doc_name} | {section}\n\n"
                            f"  `vector={r.vector_score:.3f}` · "
                            f"`graph={r.graph_score:.3f}` · "
                            f"`policy={r.policy_score:.3f}` → "
                            f"**final={r.final_score:.3f}**"
                        )
                        if r.connected_entities:
                            st.caption(f"  Entities: {', '.join(r.connected_entities)}")

            # Show ranking differences
            vec_chunks = [r.chunk_id for r in retrieval_vector.results]
            hyb_chunks = [r.chunk_id for r in retrieval_hybrid.results]
            only_in_hybrid = [cid for cid in hyb_chunks if cid not in vec_chunks]
            only_in_vector = [cid for cid in vec_chunks if cid not in hyb_chunks]
            reranked = vec_chunks != hyb_chunks

            st.divider()
            st.subheader("📈 Retrieval Analysis")

            # Quantitative metrics
            m1, m2, m3, m4 = st.columns(4)
            vec_docs = set(r.doc_name for r in retrieval_vector.results)
            hyb_docs = set(r.doc_name for r in retrieval_hybrid.results)
            hyb_entities = set()
            for r in retrieval_hybrid.results:
                hyb_entities.update(r.connected_entities)
            vec_entities = set()
            for r in retrieval_vector.results:
                vec_entities.update(r.connected_entities)

            m1.metric("Doc Coverage (Vec)", f"{len(vec_docs)} docs")
            m2.metric("Doc Coverage (Hyb)", f"{len(hyb_docs)} docs",
                       delta=f"+{len(hyb_docs) - len(vec_docs)}" if len(hyb_docs) > len(vec_docs) else None)
            m3.metric("Entities (Vec)", f"{len(vec_entities)}")
            m4.metric("Entities (Hyb)", f"{len(hyb_entities)}",
                       delta=f"+{len(hyb_entities) - len(vec_entities)}" if len(hyb_entities) > len(vec_entities) else None)

            # Query entity detection
            if retrieval_hybrid.query_entities:
                st.info(f"🔍 **Query entities detected:** {', '.join(retrieval_hybrid.query_entities)}")

            if only_in_hybrid:
                st.success(f"**Graph pulled in {len(only_in_hybrid)} chunk(s)** not found by vector search alone")
                for cid in only_in_hybrid:
                    r = next(x for x in retrieval_hybrid.results if x.chunk_id == cid)
                    st.markdown(f"  + **{r.doc_name}** | {' > '.join(r.section_path)} — "
                                f"graph={r.graph_score:.3f}, vector={r.vector_score:.3f}")
                    if r.connected_entities:
                        st.caption(f"    via entities: {', '.join(r.connected_entities)}")
            if only_in_vector:
                st.info(f"**{len(only_in_vector)} chunk(s) in vector-only** were displaced by graph-connected chunks")
            if reranked and not only_in_hybrid:
                st.info("**Same chunks retrieved, but re-ranked** by graph connectivity scores")
                for i, (vc, hc) in enumerate(zip(vec_chunks, hyb_chunks), 1):
                    if vc != hc:
                        st.caption(f"  Position {i}: vector had `{vc[:20]}...` → hybrid has `{hc[:20]}...`")
            if not reranked and not only_in_hybrid:
                st.warning("⚠️ No difference — try a multi-hop question that spans multiple policies")

            # Save to history (hybrid answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"**[A/B Comparison]**\n\n**Vector-Only:** {resp_vector.get('answer', '')}\n\n**Hybrid:** {resp_hybrid.get('answer', '')}",
                "citations": resp_hybrid.get("citations", []),
            })

        else:
            # ── Single mode ──
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    retrieval = retriever.retrieve(
                        query=prompt,
                        top_k=top_k,
                        use_graph=use_graph,
                    )
                    response = generate_response(
                        query=prompt,
                        retrieval=retrieval,
                        config=config,
                        model_override=llm_model,
                    )

                # Display answer
                answer = response.get("answer", "No response generated.")
                citations = response.get("citations", [])
                msg_idx = len(st.session_state.messages)

                if citations:
                    st.markdown(_linkify_citations(answer, msg_idx), unsafe_allow_html=True)
                else:
                    st.markdown(answer)

                # Confidence badge
                conf = response.get("confidence", 0)
                if conf > 0.7:
                    st.success(f"Confidence: {conf:.2f} ✓")
                elif conf > 0.4:
                    st.warning(f"Confidence: {conf:.2f} ⚠")
                else:
                    st.error(f"Confidence: {conf:.2f} — Low evidence")

                # Citations
                if citations and show_evidence:
                    with st.expander("📎 Citations", expanded=True):
                        _render_citations(citations, msg_idx)

                # Evidence details
                if show_evidence:
                    retrieval_results = response.get("retrieval_results", [])
                    if retrieval_results:
                        with st.expander("🔍 Retrieved Evidence"):
                            for i, r in enumerate(retrieval_results, 1):
                                st.markdown(f"**Evidence {i}** (Score: {r['final_score']:.3f})")
                                st.markdown(f"*Source: {r['doc_name']} | {' > '.join(r['section_path'])}*")
                                st.code(r["content"][:300] + ("..." if len(r["content"]) > 300 else ""))
                                if r.get("connected_entities"):
                                    st.caption(f"Connected entities: {', '.join(r['connected_entities'])}")
                                st.divider()

                # Graph info
                if show_graph_info and use_graph:
                    with st.expander("🔗 Graph Traversal Info"):
                        st.markdown(f"**Query entities detected:** {', '.join(retrieval.query_entities) or 'None'}")
                        st.markdown(f"**Evidence sufficient:** {retrieval.evidence_sufficient}")
                        if retrieval.subgraph:
                            st.metric("Subgraph nodes", len(retrieval.subgraph.get("nodes", [])))
                            st.metric("Subgraph edges", len(retrieval.subgraph.get("edges", [])))

                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "evidence_details": retrieval_results[:3] if show_evidence else None,
                })


if __name__ == "__main__":
    main()
